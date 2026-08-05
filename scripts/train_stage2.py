from __future__ import annotations

import argparse
from contextlib import ExitStack

import torch

from qkd.modeling import load_causal_lm, load_tokenizer
from qkd.photonic.checkpoint import load_stage_one_checkpoints
from qkd.photonic.model import find_decoder_layers, make_compressed_student
from qkd.training.artifacts import STAGE2_LOG_FIELDS, TrainingArtifacts, resolve_checkpoint_dir
from qkd.training.spsa import SPSA
from qkd.training.stage2_loss import stage_two_loss
from qkd.training.tools import (
    apply_overrides,
    load_config,
    make_loader,
    next_batch,
    provider_factory,
    section,
    training_device,
)


def arguments() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="阶段二：端到端光子条件化知识蒸馏")
    parser.add_argument("--config", required=True, help="阶段二 YAML 配置文件")
    parser.add_argument("--set", action="append", default=[], metavar="键路径=值", help="临时覆盖 YAML；可重复使用。")
    return parser.parse_args()


def main() -> None:
    args = arguments()
    config = apply_overrides(load_config(args.config), args.set)
    experiment, data = section(config, "experiment"), section(config, "data")
    model, compression = section(config, "model"), section(config, "compression")
    photonic, initialization = section(config, "photonic"), section(config, "initialization")
    optimization, logging = section(config, "optimization"), section(config, "logging")
    target_layers = tuple(int(index) for index in model["target_layers"])
    checkpoint_references = initialization.get("stage1_checkpoints")
    if not isinstance(checkpoint_references, list) or not checkpoint_references:
        raise ValueError("initialization.stage1_checkpoints 必须提供覆盖全部目标层的目录列表")
    checkpoint_dirs = [resolve_checkpoint_dir(path) for path in checkpoint_references]
    initialization["stage1_checkpoints"] = [str(path) for path in checkpoint_dirs]

    torch.manual_seed(int(experiment["seed"]))
    device = training_device()
    teacher_name = str(model["teacher"])
    print(f"运行设备：{device}")
    artifacts = TrainingArtifacts.create(config, "stage2", device, STAGE2_LOG_FIELDS, teacher_name)
    output = artifacts.output

    tokenizer = load_tokenizer(teacher_name)
    train_loader = make_loader(str(data["train_data"]), tokenizer, int(data["batch_size"]), True)
    validation_loader = make_loader(str(data["validation_data"]), tokenizer, int(data["batch_size"]), False)
    probe_batch = {key: value.to(device) for key, value in next(iter(validation_loader)).items()}
    teacher = load_causal_lm(teacher_name, trainable=False).to(device).eval()
    rank = int(compression["rank"])
    z_dim = int(compression["z_dim"])
    kappa = float(compression["kappa"])
    student, replacements = make_compressed_student(
        teacher, provider_factory(str(photonic["provider"]), z_dim, photonic.get("ema_decay")),
        rank, z_dim, kappa, target_layers,
    )
    for replacement in replacements:
        replacement.shots = photonic.get("shots")
    load_stage_one_checkpoints(checkpoint_dirs, replacements, rank, z_dim, kappa, target_layers, str(photonic["provider"]))
    print(f"已合并阶段一 checkpoint：{checkpoint_dirs}")
    student.to(device).train()
    optimizer = torch.optim.Adam(
        (parameter for replacement in replacements for parameter in replacement.adam_parameters()),
        lr=float(optimization["adam_learning_rate"]),
    )
    spsa = SPSA(perturbation=float(optimization["spsa_perturbation"]), learning_rate=float(optimization["spsa_learning_rate"]), seed=int(experiment["seed"]))
    early_stop_loss = optimization.get("early_stop_loss")
    if early_stop_loss is not None and float(early_stop_loss) < 0:
        raise ValueError("optimization.early_stop_loss 必须为非负数或 null")
    stopped_early = False
    best_loss = float("inf")
    best_step: int | None = None
    final_loss: float | None = None
    iterator = iter(train_loader)

    @torch.no_grad()
    def save_best_probe(step: int, loss: float) -> None:
        teacher_layers, student_layers = find_decoder_layers(teacher), find_decoder_layers(student)
        teacher_y: dict[int, torch.Tensor] = {}
        student_y: dict[int, torch.Tensor] = {}

        def capture(values: dict[int, torch.Tensor], index: int):
            def hook(module, inputs, output_value) -> None:
                values[index] = output_value.detach()
            return hook

        student.eval()
        try:
            with ExitStack() as stack:
                for index in target_layers:
                    stack.callback(teacher_layers[index].mlp.register_forward_hook(capture(teacher_y, index)).remove)
                    stack.callback(student_layers[index].mlp.register_forward_hook(capture(student_y, index)).remove)
                teacher(input_ids=probe_batch["input_ids"], attention_mask=probe_batch["attention_mask"])
                student(input_ids=probe_batch["input_ids"], attention_mask=probe_batch["attention_mask"])
            artifacts.save_best_probe({
                "format": "qkd-best-probe-v1",
                "stage": "stage2",
                "step": step,
                "selection": {"metric": "train_loss", "value": loss},
                "target_layers": list(target_layers),
                "input_ids": probe_batch["input_ids"].detach().cpu(),
                "attention_mask": probe_batch["attention_mask"].detach().cpu(),
                "labels": probe_batch["labels"].detach().cpu(),
                "layers": {
                    str(index): {
                        "teacher_y": teacher_y[index].to("cpu", torch.float16),
                        "student_y": student_y[index].to("cpu", torch.float16),
                    }
                    for index in target_layers
                },
            })
        finally:
            student.train()

    for step in range(1, int(optimization["steps"]) + 1):
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
            "step": step, "elapsed_seconds": artifacts.elapsed_seconds, "loss": final_loss,
            "ce": terms["ce"].detach().float().item(), "kd": terms["kd"].detach().float().item(),
            "is_best": False, "spsa_applied": False, "early_stopped": False,
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
            print(f"stage2 提前停止：step={step} loss={final_loss:.6f} <= {float(early_stop_loss):.6f}")
            break

        if step % int(logging["log_every"]) == 0:
            print(f"stage2 step={step} loss={final_loss:.4f} ce={row['ce']:.4f} kd={row['kd']:.4f}")

        if step % int(optimization["spsa_every"]) == 0:
            def validation_objective() -> torch.Tensor:
                student.eval()
                values = []
                with torch.no_grad():
                    for validation in validation_loader:
                        validation = {key: value.to(device) for key, value in validation.items()}
                        teacher_logits = teacher(input_ids=validation["input_ids"], attention_mask=validation["attention_mask"]).logits
                        student_logits = student(input_ids=validation["input_ids"], attention_mask=validation["attention_mask"]).logits
                        values.append(stage_two_loss(
                            student_logits, teacher_logits, validation["labels"],
                            temperature=float(optimization["temperature"]), top_k=int(optimization["top_k"]),
                        )["loss"])
                student.train()
                return torch.stack(values).mean()

            for replacement in replacements:
                for theta in (replacement.theta_gate, replacement.theta_up, replacement.theta_down):
                    spsa.step(theta, validation_objective)
            row["spsa_applied"] = True
        artifacts.log_step(row)

    artifacts.finish(status="early_stopped" if stopped_early else "completed", final_step=step, final_loss=final_loss, best_step=best_step, best_train_loss=best_loss, early_stop_loss=early_stop_loss)
    print(f"已保存阶段二条件化模块至 {output}")


if __name__ == "__main__":
    main()
