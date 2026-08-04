from __future__ import annotations

import argparse
from pathlib import Path

import torch

from qkd.modeling import load_causal_lm, load_tokenizer
from qkd.photonic.checkpoint import load_stage_one_checkpoints, save_compressed_checkpoint
from qkd.photonic.model import make_compressed_student
from qkd.training.tools import apply_overrides, load_config, make_loader, next_batch, provider_factory, section, training_device, write_run_config
from qkd.training.spsa import SPSA
from qkd.training.stage2_loss import stage_two_loss


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
    checkpoint_dirs = initialization.get("stage1_checkpoints")
    if not isinstance(checkpoint_dirs, list) or not checkpoint_dirs:
        raise ValueError("initialization.stage1_checkpoints 必须提供覆盖全部目标层的目录列表")

    torch.manual_seed(int(experiment["seed"]))
    device = training_device()
    output = Path(experiment["output_dir"])
    teacher_name = str(model["teacher"])
    print(f"运行设备：{device}")
    write_run_config(output, config, "stage2", device, {"status": "running", "teacher": teacher_name})

    tokenizer = load_tokenizer(teacher_name)
    train_loader = make_loader(str(data["train_data"]), tokenizer, int(data["batch_size"]), True)
    validation_loader = make_loader(str(data["validation_data"]), tokenizer, int(data["batch_size"]), False)
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
    iterator = iter(train_loader)

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

        if early_stop_loss is not None and terms["loss"].detach().float().item() <= float(early_stop_loss):
            stopped_early = True
            print(f"stage2 提前停止：step={step} loss={terms['loss'].detach().float().item():.6f} <= {float(early_stop_loss):.6f}")
            break

        if step % int(logging["log_every"]) == 0:
            print(f"stage2 step={step} loss={terms['loss'].detach().float().item():.4f} ce={terms['ce'].detach().float().item():.4f} kd={terms['kd'].detach().float().item():.4f}")

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

    save_compressed_checkpoint(output, replacements, rank, z_dim, kappa, target_layers, teacher_name, str(photonic["provider"]))
    tokenizer.save_pretrained(output)
    write_run_config(output, config, "stage2", device, {"status": "early_stopped" if stopped_early else "completed", "teacher": teacher_name, "early_stop_loss": early_stop_loss})
    print(f"已保存阶段二条件化模块至 {output}")


if __name__ == "__main__":
    main()
