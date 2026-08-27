from __future__ import annotations

import argparse
import math
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
    parser.add_argument("--pb-only", action="store_true", help="仅训练 P/B，固定 C 与 provider，并严格使用 g=1")
    parser.add_argument("--random-init", action="store_true", help="不加载 Stage 1 checkpoint，直接以 SVD P/B 与新建条件化参数开始 Stage 2")
    return parser.parse_args()


def parameter_norm(parameters) -> float:
    values = [parameter.grad.detach().float().norm().square() for parameter in parameters if parameter.grad is not None]
    return torch.stack(values).sum().sqrt().item() if values else 0.0


def stage_two_gradient_norms(replacements) -> dict[str, float]:
    c = [parameter for replacement in replacements for parameter in replacement.c_parameters()]
    pb = [parameter for replacement in replacements for parameter in replacement.pb_parameters()]
    providers = [parameter for replacement in replacements for parameter in replacement.photonic_parameters()]
    theta = [getattr(provider, "theta") for replacement in replacements for provider in (replacement.gate_provider, replacement.up_provider, replacement.down_provider) if hasattr(provider, "theta")]
    phi = [getattr(provider, "phi") for replacement in replacements for provider in (replacement.gate_provider, replacement.up_provider, replacement.down_provider) if hasattr(provider, "phi")]
    return {
        "c_grad_norm": parameter_norm(c),
        "pb_grad_norm": parameter_norm(pb),
        "feature_grad_norm": parameter_norm(providers),
        "theta_grad_norm": parameter_norm(theta),
        "phi_grad_norm": parameter_norm(phi),
    }


def main() -> None:
    args = arguments()

    # 1. 配置与初始化方式
    config = apply_overrides(load_config(args.config), args.set)
    experiment, data = section(config, "experiment"), section(config, "data")
    model, compression = section(config, "model"), section(config, "compression")
    photonic, initialization = section(config, "photonic"), section(config, "initialization")
    optimization, validation_settings = section(config, "optimization"), section(config, "validation")
    target_layers = tuple(int(index) for index in model["target_layers"])
    if args.random_init:
        checkpoint_dirs = []
        initialization["mode"] = "random"
        initialization["stage1_checkpoints"] = []
    else:
        checkpoint_references = initialization.get("stage1_checkpoints")
        if not isinstance(checkpoint_references, list) or not checkpoint_references:
            raise ValueError("initialization.stage1_checkpoints 必须提供覆盖全部目标层的目录列表；若要跳过 Stage 1，请传 --random-init")
        checkpoint_dirs = [resolve_checkpoint_dir(path) for path in checkpoint_references]
        initialization["mode"] = "stage1"
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
        teacher, provider_factory(str(photonic["provider"]), z_dim, photonic.get("ema_decay"), n_modes, n_layers, float(photonic.get("theta_init_std", 0.1)), float(photonic.get("phi_init_std", 0.1)), int(photonic.get("meshes", 1))),
        rank, z_dim, kappa, target_layers, gate_scale=float(compression.get("gate_scale", 0.5)),
        c_init_std=float(compression.get("c_init_std", 0.1)),
        encoded_input_mode=str(compression.get("encoded_input_mode", "input_dependent")),
        fixed_encoded_std=float(compression.get("fixed_encoded_std", 0.1)),
    )
    for replacement in replacements:
        replacement.shots = photonic.get("shots")
    if checkpoint_dirs:
        load_stage_one_checkpoints(checkpoint_dirs, replacements, rank, z_dim, kappa, target_layers, str(photonic["provider"]))
    student.to(device).train()
    c_parameters = [parameter for replacement in replacements for parameter in replacement.c_parameters()]
    photonic_parameters = [parameter for replacement in replacements for parameter in replacement.photonic_parameters()]
    pb_parameters = [parameter for replacement in replacements for parameter in replacement.pb_parameters()]
    if args.pb_only and not bool(compression.get("train_pb", False)):
        raise ValueError("--pb-only 需要 compression.train_pb=true")
    parameter_groups = []
    if args.pb_only:
        for parameter in c_parameters + photonic_parameters:
            parameter.requires_grad_(False)
        for replacement in replacements:
            replacement.disable_conditioning()
    else:
        parameter_groups.extend([
            {"params": c_parameters, "lr": float(optimization["c_learning_rate"])},
            {"params": photonic_parameters, "lr": float(optimization["photonic_learning_rate"])},
        ])
    if bool(compression.get("train_pb", False)):
        for parameter in pb_parameters:
            parameter.requires_grad_(True)
        parameter_groups.append({"params": pb_parameters, "lr": float(optimization["pb_learning_rate"])})
    if not parameter_groups:
        raise ValueError("没有可训练参数；请启用 P/B 或 C/provider")
    optimizer = torch.optim.Adam(parameter_groups)
    schedule_name = str(optimization.get("lr_schedule", "constant")).lower()
    min_lr_scale = float(optimization.get("min_lr_scale", 0.1))
    if schedule_name == "cosine":
        total_steps = int(optimization["steps"])
        scheduler = torch.optim.lr_scheduler.LambdaLR(
            optimizer,
            lr_lambda=lambda step: min_lr_scale + (1 - min_lr_scale) * 0.5 * (1 + math.cos(math.pi * min(step, total_steps) / total_steps)),
        )
    elif schedule_name == "constant":
        scheduler = None
    else:
        raise ValueError("optimization.lr_schedule 仅支持 constant 或 cosine")
    # spsa = SPSA(
    #     perturbation=float(optimization["spsa_perturbation"]),
    #     learning_rate=float(optimization["spsa_learning_rate"]),
    #     seed=int(experiment["seed"]),
    # )
    early_stop_loss = optimization.get("early_stop_loss")
    if early_stop_loss is not None and float(early_stop_loss) < 0:
        raise ValueError("optimization.early_stop_loss 必须为非负数或 null")
    stopped_early = False
    validation_every = int(validation_settings["every"])
    if validation_every < 1:
        raise ValueError("validation.every 必须为正整数")
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
            artifacts.save_best_probe(best_probe_payload("stage2", step, "validation_loss", loss, target_layers, probe_batch, teacher_y, student_y))
        finally:
            student.train()

    # 4. 端到端蒸馏：CE + Top-K KD；可选解冻 P/B，所有参数通过 autograd 优化。
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
        gradient_norms = stage_two_gradient_norms(replacements)
        optimizer.step()
        if scheduler is not None:
            scheduler.step()
        optimizer.zero_grad()
        final_loss = terms["loss"].detach().float().item()
        row = {
            "step": step,
            "elapsed_seconds": artifacts.elapsed_seconds,
            "loss": final_loss,
            "ce": terms["ce"].detach().float().item(),
            "kd": terms["kd"].detach().float().item(),
            **gradient_norms,
            "validation_loss": "",
            "validation_ce": "",
            "validation_kd": "",
            "is_best": False,
            "spsa_applied": False,
            "early_stopped": False,
        }

        if step % validation_every == 0:
            validation = stage_two_validation_objective(
                student, teacher, validation_loader, device,
                float(optimization["temperature"]), int(optimization["top_k"]),
            )
            row["validation_loss"] = validation["loss"]
            row["validation_ce"] = validation["ce"]
            row["validation_kd"] = validation["kd"]
            if validation["loss"] < best_loss:
                best_loss = validation["loss"]
                best_step = step
                row["is_best"] = True
                artifacts.save_checkpoint(replacements, rank, z_dim, kappa, target_layers, teacher_name, str(photonic["provider"]))
                save_best_probe(step, validation["loss"])

        if early_stop_loss is not None and final_loss <= float(early_stop_loss):
            stopped_early = True
            row["early_stopped"] = True
            artifacts.log_step(row)
            progress.set_postfix(loss=f"{final_loss:.4f}", stopped=True)
            break

        # 保留 SPSA 模块与配置以便后续非可微硬件后端；当前可微模拟不调用它。
        artifacts.log_step(row)
        progress.set_postfix(loss=f"{final_loss:.4f}", val=row["validation_loss"] or "-", ce=f"{row['ce']:.4f}")

    artifacts.finish(status="early_stopped" if stopped_early else "completed", final_step=step, final_loss=final_loss, best_step=best_step, best_validation_loss=best_loss, early_stop_loss=early_stop_loss)


if __name__ == "__main__":
    main()
