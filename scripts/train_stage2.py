from __future__ import annotations

import argparse
from functools import partial

import torch
from tqdm import tqdm

from qkd.modeling import load_causal_lm, load_tokenizer
from qkd.photonic.checkpoint import load_stage_one_checkpoints
from qkd.photonic.model import make_compressed_student
from qkd.training.artifacts import STAGE2_LOG_FIELDS, TrainingArtifacts, best_probe_payload, capture_mlp_outputs, resolve_checkpoint_dir
from qkd.training.spsa import SPSA
from qkd.training.stage2_loss import stage_two_loss
from qkd.training.tools import apply_overrides, load_config, make_loader, next_batch, provider_factory, section, stage_two_validation_objective, training_device


def arguments() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="阶段二：端到端光子条件化知识蒸馏")
    parser.add_argument("--config", required=True, help="阶段二 YAML 配置文件")
    parser.add_argument("--set", action="append", default=[], metavar="键路径=值", help="临时覆盖 YAML；可重复使用。")
    return parser.parse_args()


def main() -> None:
    args = arguments()

    # 1. 配置与 Stage 1 初始化权重
    config = apply_overrides(load_config(args.config), args.set)
    experiment, data = section(config, "experiment"), section(config, "data")
    model, compression = section(config, "model"), section(config, "compression")
    photonic, initialization = section(config, "photonic"), section(config, "initialization")
    optimization = section(config, "optimization")
    target_layers = tuple(int(index) for index in model["target_layers"])
    checkpoint_references = initialization.get("stage1_checkpoints")
    if not isinstance(checkpoint_references, list) or not checkpoint_references:
        raise ValueError("initialization.stage1_checkpoints 必须提供覆盖全部目标层的目录列表")
    checkpoint_dirs = [resolve_checkpoint_dir(path) for path in checkpoint_references]
    initialization["stage1_checkpoints"] = [str(path) for path in checkpoint_dirs]
    torch.manual_seed(int(experiment["seed"]))
    device = training_device()
    teacher_name = str(model["teacher"])
    artifacts = TrainingArtifacts.create(config, "stage2", device, STAGE2_LOG_FIELDS)

    # 2. 数据、教师模型与多层学生模型
    tokenizer = load_tokenizer(teacher_name)
    train_loader = make_loader(str(data["train_data"]), tokenizer, int(data["batch_size"]), True)
    validation_loader = make_loader(str(data["validation_data"]), tokenizer, int(data["batch_size"]), False)
    probe_batch = {key: value.to(device) for key, value in next(iter(validation_loader)).items()}
    teacher = load_causal_lm(teacher_name, trainable=False).to(device).eval()
    rank = int(compression["rank"])
    z_dim = int(compression["z_dim"])
    kappa = float(compression["kappa"])
    n_modes, n_layers = int(photonic["modes"]), int(photonic["layers"])
    student, replacements = make_compressed_student(
        teacher, provider_factory(str(photonic["provider"]), z_dim, photonic.get("ema_decay"), n_modes, n_layers),
        rank, z_dim, kappa, target_layers,
    )
    for replacement in replacements:
        replacement.shots = photonic.get("shots")
    load_stage_one_checkpoints(checkpoint_dirs, replacements, rank, z_dim, kappa, target_layers, str(photonic["provider"]))
    student.to(device).train()
    optimizer = torch.optim.Adam(
        (parameter for replacement in replacements for parameter in (*replacement.adam_parameters(), *replacement.photonic_parameters())),
        lr=float(optimization["adam_learning_rate"]),
    )
    # spsa = SPSA(
    #     perturbation=float(optimization["spsa_perturbation"]),
    #     learning_rate=float(optimization["spsa_learning_rate"]),
    #     seed=int(experiment["seed"]),
    # )
    early_stop_loss = optimization.get("early_stop_loss")
    if early_stop_loss is not None and float(early_stop_loss) < 0:
        raise ValueError("optimization.early_stop_loss 必须为非负数或 null")
    stopped_early = False
    best_loss = float("inf")
    best_step: int | None = None
    final_loss: float | None = None
    iterator = iter(train_loader)

    # 3. 最佳 checkpoint 的固定验证 probe
    @torch.no_grad()
    def save_best_probe(step: int, loss: float) -> None:
        student.eval()
        try:
            with capture_mlp_outputs(teacher, student, target_layers) as (teacher_y, student_y):
                teacher(input_ids=probe_batch["input_ids"], attention_mask=probe_batch["attention_mask"])
                student(input_ids=probe_batch["input_ids"], attention_mask=probe_batch["attention_mask"])
            artifacts.save_best_probe(best_probe_payload("stage2", step, "train_loss", loss, target_layers, probe_batch, teacher_y, student_y))
        finally:
            student.train()

    # 4. 端到端蒸馏：CE + Top-K KD；P/B 冻结，光路参数通过 autograd 优化。
    layer_label = ",".join(str(index) for index in target_layers)
    progress = tqdm(range(1, int(optimization["steps"]) + 1), desc=f"Stage 2 · layers {layer_label}", unit="step")
    for step in progress:
        batch, iterator = next_batch(iterator, train_loader)
        batch = {key: value.to(device) for key, value in batch.items()}
        with torch.no_grad():
            teacher_logits = teacher(input_ids=batch["input_ids"], attention_mask=batch["attention_mask"]).logits  # 跑不带梯度的 Teacher 获得教师 logits
        student_logits = student(input_ids=batch["input_ids"], attention_mask=batch["attention_mask"]).logits  # 跑全网 Student 获得学生 logits
        terms = stage_two_loss(student_logits, teacher_logits, batch["labels"], temperature=float(optimization["temperature"]), top_k=int(optimization["top_k"]))
        terms["loss"].backward()
        optimizer.step()
        optimizer.zero_grad()
        final_loss = terms["loss"].detach().float().item()
        row = {
            "step": step,
            "elapsed_seconds": artifacts.elapsed_seconds,
            "loss": final_loss,
            "ce": terms["ce"].detach().float().item(),
            "kd": terms["kd"].detach().float().item(),
            "is_best": False,
            "spsa_applied": False,
            "early_stopped": False,
        }

        if final_loss < best_loss:
            best_loss = final_loss
            best_step = step
            row["is_best"] = True
            artifacts.save_checkpoint(replacements, rank, z_dim, kappa, target_layers, teacher_name, str(photonic["provider"]))
            save_best_probe(step, final_loss)

        if early_stop_loss is not None and final_loss <= float(early_stop_loss):
            stopped_early = True
            row["early_stopped"] = True
            artifacts.log_step(row)
            progress.set_postfix(loss=f"{final_loss:.4f}", stopped=True)
            break

        # 保留 SPSA 模块与配置以便后续非可微硬件后端；当前可微模拟不调用它。
        artifacts.log_step(row)
        progress.set_postfix(loss=f"{final_loss:.4f}", ce=f"{row['ce']:.4f}", kd=f"{row['kd']:.4f}")

    artifacts.finish(status="early_stopped" if stopped_early else "completed", final_step=step, final_loss=final_loss, best_step=best_step, best_train_loss=best_loss, early_stop_loss=early_stop_loss)


if __name__ == "__main__":
    main()
