from __future__ import annotations

import argparse
import math
from functools import partial
from typing import Any

import torch
from tqdm import tqdm

from qkd.modeling import load_causal_lm, load_tokenizer
from qkd.photonic.model import find_decoder_layers, make_compressed_student
from qkd.training.artifacts import STAGE1_LOG_FIELDS, TrainingArtifacts, best_probe_payload
from qkd.training.spsa import SPSA
from qkd.training.tools import StageOneReference, apply_overrides, load_config, make_loader, next_batch, provider_factory, section, training_device


def arguments() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="阶段一：单层光子条件化低秩 MLP 重建")
    parser.add_argument("--config", required=True, help="阶段一 YAML 配置文件")
    parser.add_argument("--set", action="append", default=[], metavar="键路径=值", help="临时覆盖 YAML；可重复，例如 'model.target_layers=[22]'。")
    parser.add_argument("--freeze-photonic", action="store_true", help="固定光路 theta/phi，只训练 C（C-only 消融）")
    parser.add_argument("--pb-only", action="store_true", help="仅训练 P/B，固定 C 与 provider，并严格使用 g=1")
    parser.add_argument("--freeze-pb-after", type=int, default=None, metavar="STEP", help="在完成 STEP 次 P/B 更新后冻结 P/B；C 与光路继续训练")
    parser.add_argument("--validate-initial", action="store_true", help="在 step 0 验证并保存初始 SVD/条件化 checkpoint")
    return parser.parse_args()


def stage_one_gradient_norms(replacement) -> dict[str, float]:
    """在 optimizer.step 前记录各投影 C/theta/phi 的 L2 梯度范数。"""
    def norm(parameter: torch.Tensor) -> float:
        return 0.0 if parameter.grad is None else parameter.grad.detach().float().norm().item()

    def provider_norm(provider: torch.nn.Module) -> float:
        values = [norm(parameter) ** 2 for parameter in provider.parameters()]
        return sum(values) ** 0.5

    values: dict[str, float] = {}
    for name, projection, provider in (
        ("gate", replacement.gate, replacement.gate_provider),
        ("up", replacement.up, replacement.up_provider),
        ("down", replacement.down, replacement.down_provider),
    ):
        values[f"{name}_c_grad_norm"] = norm(projection.C)
        values[f"{name}_p_grad_norm"] = norm(projection.P)
        values[f"{name}_b_grad_norm"] = norm(projection.B)
        values[f"{name}_theta_grad_norm"] = norm(provider.theta) if hasattr(provider, "theta") else 0.0
        values[f"{name}_phi_grad_norm"] = norm(provider.phi) if hasattr(provider, "phi") else 0.0
        values[f"{name}_feature_grad_norm"] = provider_norm(provider)
    return values


def main() -> None:
    args = arguments()

    # 1. 配置与实验目录
    config: dict[str, Any] = apply_overrides(load_config(args.config), args.set)
    experiment, data = section(config, "experiment"), section(config, "data")
    model, compression = section(config, "model"), section(config, "compression")
    photonic, optimization, validation_settings = (
        section(config, "photonic"),
        section(config, "optimization"),
        section(config, "validation"),
    )
    target_layers = tuple(int(index) for index in model["target_layers"])
    if len(target_layers) != 1:
        raise ValueError("阶段一配置一次只能指定一个 target_layers，例如 [21]")
    torch.manual_seed(int(experiment["seed"]))
    device = training_device()
    teacher_name = str(model["teacher"])
    n_modes, n_layers = int(photonic["modes"]), int(photonic["layers"])
    artifacts = TrainingArtifacts.create(config, "stage1_old", device, STAGE1_LOG_FIELDS)

    # 2. 数据、教师模型与待训练的单层学生 MLP
    tokenizer = load_tokenizer(teacher_name)
    train_loader = make_loader(str(data["train_data"]), tokenizer, int(data["batch_size"]), True)
    validation_loader = make_loader(str(data["validation_data"]), tokenizer, int(data["batch_size"]), False)
    probe_batch = {key: value.to(device) for key, value in next(iter(validation_loader)).items()}
    teacher = load_causal_lm(teacher_name, trainable=False).to(device).eval()
    teacher_layers = find_decoder_layers(teacher)
    target = target_layers[0]
    if target < 0 or target >= len(teacher_layers):
        raise ValueError(f"目标层必须在 0 到 {len(teacher_layers) - 1} 之间")
    rank = int(compression["rank"])
    z_dim = int(compression["z_dim"])
    kappa = float(compression["kappa"])
    gate_scale = float(compression.get("gate_scale", 0.5))
    student, replacements = make_compressed_student(
        teacher, provider_factory(str(photonic["provider"]), z_dim, photonic.get("ema_decay"), n_modes, n_layers, float(photonic.get("theta_init_std", 0.1)), float(photonic.get("phi_init_std", 0.1)), int(photonic.get("meshes", 1))),
        rank, z_dim, kappa, target_layers,
        gate_scale=gate_scale,
        c_init_std=float(compression.get("c_init_std", 0.1)),
        encoded_input_mode=str(compression.get("encoded_input_mode", "input_dependent")),
        fixed_encoded_std=float(compression.get("fixed_encoded_std", 0.1)),
    )
    replacement = replacements[0]
    replacement.shots = photonic.get("shots")
    student.to(device).train()
    c_parameters = list(replacement.c_parameters())
    photonic_parameters = list(replacement.photonic_parameters())
    pb_parameters = list(replacement.pb_parameters())
    if args.pb_only and not bool(compression.get("train_pb", False)):
        raise ValueError("--pb-only 需要 compression.train_pb=true")
    if args.freeze_pb_after is not None:
        if args.freeze_pb_after < 1:
            raise ValueError("--freeze-pb-after 必须为正整数")
        if not bool(compression.get("train_pb", False)) or args.pb_only:
            raise ValueError("--freeze-pb-after 需要 compression.train_pb=true 且不能与 --pb-only 同用")
    parameter_groups: list[dict[str, object]] = []
    if args.pb_only:
        for parameter in c_parameters + photonic_parameters:
            parameter.requires_grad_(False)
        replacement.disable_conditioning()
    else:
        parameter_groups.append({"params": c_parameters, "lr": float(optimization["c_learning_rate"])})
    if not args.freeze_photonic and not args.pb_only:
        parameter_groups.append(
            {
                "params": photonic_parameters,
                "lr": float(optimization["photonic_learning_rate"]),
            }
        )
    else:
        for parameter in photonic_parameters:
            parameter.requires_grad_(False)
    if bool(compression.get("train_pb", False)):
        for parameter in pb_parameters:
            parameter.requires_grad_(True)
        parameter_groups.append(
            {
                "params": pb_parameters,
                "lr": float(optimization["pb_learning_rate"]),
            }
        )
    if not parameter_groups:
        raise ValueError("没有可训练参数；请启用 P/B 或 C/provider")
    optimizer = torch.optim.Adam(parameter_groups)
    schedule_name = str(optimization.get("lr_schedule", "constant")).lower()
    min_lr_scale = float(optimization.get("min_lr_scale", 0.1))
    if not 0 < min_lr_scale <= 1:
        raise ValueError("optimization.min_lr_scale 必须在 (0, 1] 内")
    if schedule_name == "cosine":
        total_steps = int(optimization["steps"])
        scheduler = torch.optim.lr_scheduler.LambdaLR(
            optimizer,
            lr_lambda=lambda step: min_lr_scale
            + (1 - min_lr_scale) * 0.5 * (1 + math.cos(math.pi * min(step, total_steps) / total_steps)),
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
    best_validation_loss = float("inf")
    best_step: int | None = None
    final_loss: float | None = None

    # 3. 局部重建损失、验证与最佳 probe
    with StageOneReference(
        teacher,
        teacher_layers[target].mlp,
        replacement,
        auxiliary_weight=float(optimization.get("auxiliary_loss_weight", 0.0)),
        output_weight=float(optimization.get("output_loss_weight", 1.0)),
        loss_scale=float(optimization.get("loss_scale", 1.0)),
    ) as reference:

        @torch.no_grad()
        def save_best_probe(step: int, metric_name: str, metric_value: float | None) -> None:
            student.eval()
            try:
                details = reference.terms(probe_batch)
                if reference.values is None:
                    raise RuntimeError("教师 MLP hook 未捕获到输出")
                _, teacher_y = reference.values
                artifacts.save_best_probe(best_probe_payload("stage1_old", step, metric_name, metric_value, target_layers, probe_batch, {target: teacher_y}, {target: details["output"]}))
            finally:
                student.train()

        if args.validate_initial:
            # With C=0 and frozen P/B this is exactly the g=1 truncated-SVD
            # baseline.  Recording it makes later conditioning improvements
            # causal rather than inferred from separate runs.
            initial_loss, initial_metrics = reference.validate(validation_loader, student, device)
            best_validation_loss = initial_loss
            best_step = 0
            artifacts.save_checkpoint(replacements, rank, z_dim, kappa, target_layers, teacher_name, str(photonic["provider"]))
            save_best_probe(0, "validation_loss", initial_loss)
            initial_row = {field: "" for field in STAGE1_LOG_FIELDS}
            initial_row.update({
                "step": 0,
                "elapsed_seconds": artifacts.elapsed_seconds,
                "validation_loss": initial_loss,
                "validation_output_loss": initial_metrics["output_loss"],
                "validation_gate_loss": initial_metrics["gate_loss"],
                "validation_up_loss": initial_metrics["up_loss"],
                "validation_down_loss": initial_metrics["down_loss"],
                "validation_y_nmse": initial_metrics["y_nmse"],
                "is_best": True,
                "spsa_applied": False,
                "early_stopped": False,
            })
            artifacts.log_step(initial_row)

        # 4. 训练循环：默认更新 C/provider；PB-only 消融仅更新 P/B。
        iterator = iter(train_loader)
        progress = tqdm(range(1, int(optimization["steps"]) + 1), desc=f"Stage 1 · layer {target}", unit="step")
        for step in progress:
            batch, iterator = next_batch(iterator, train_loader)
            batch = {key: value.to(device) for key, value in batch.items()}
            details = reference.terms(batch)
            details["loss"].backward()
            gradient_norms = stage_one_gradient_norms(replacement)
            optimizer.step()
            # Keep the first N P/B updates, then make the remainder of Stage 1
            # a conditioning-only refinement without rebuilding Adam/scheduler.
            if args.freeze_pb_after is not None and step == args.freeze_pb_after:
                for parameter in pb_parameters:
                    parameter.requires_grad_(False)
                print(f"冻结 P/B：已完成前 {args.freeze_pb_after} 次更新", flush=True)
            if scheduler is not None:
                scheduler.step()
            optimizer.zero_grad()
            final_loss = details["loss"].detach().float().item()
            diagnostics = reference.diagnostics(details, batch["attention_mask"])
            row = {
                "step": step,
                "elapsed_seconds": artifacts.elapsed_seconds,
                "train_loss": final_loss,
                "gate_loss": details["gate_loss"].detach().float().item(),
                "up_loss": details["up_loss"].detach().float().item(),
                "down_loss": details["down_loss"].detach().float().item(),
                "output_loss": details["output_loss"].detach().float().item(),
                "auxiliary_loss": details["auxiliary_loss"].detach().float().item(),
                "unscaled_loss": details["unscaled_loss"].detach().float().item(),
                **gradient_norms,
                **{name: value for name, value in diagnostics.items() if name != "valid_tokens"},
                "validation_loss": "",
                "validation_output_loss": "",
                "validation_gate_loss": "",
                "validation_up_loss": "",
                "validation_down_loss": "",
                "validation_y_nmse": "",
                "is_best": False,
                "spsa_applied": False,
                "early_stopped": False,
            }

            if early_stop_loss is not None and final_loss <= float(early_stop_loss):
                stopped_early = True
                row["early_stopped"] = True
                artifacts.log_step(row)
                progress.set_postfix(loss=f"{final_loss:.4f}", stopped=True)
                break

            # 保留 spsa 配置与实现以便后续硬件实验；当前可微模拟不调用 SPSA。
            postfix = {"loss": f"{final_loss:.4f}"}
            if int(validation_settings["every"]) and step % int(validation_settings["every"]) == 0:
                validation_loss, metrics = reference.validate(validation_loader, student, device)
                row["validation_loss"] = validation_loss
                row["validation_output_loss"] = metrics["output_loss"]
                row["validation_gate_loss"] = metrics["gate_loss"]
                row["validation_up_loss"] = metrics["up_loss"]
                row["validation_down_loss"] = metrics["down_loss"]
                row["validation_y_nmse"] = metrics["y_nmse"]
                postfix["val_loss"] = f"{validation_loss:.4f}"
                if validation_loss < best_validation_loss:
                    best_validation_loss = validation_loss
                    best_step = step
                    row["is_best"] = True
                    artifacts.save_checkpoint(replacements, rank, z_dim, kappa, target_layers, teacher_name, str(photonic["provider"]))
                    save_best_probe(step, "validation_loss", validation_loss)
            artifacts.log_step(row)
            progress.set_postfix(postfix)

        if best_step is None:
            artifacts.save_checkpoint(replacements, rank, z_dim, kappa, target_layers, teacher_name, str(photonic["provider"]))
            best_step = step
            best_validation_loss = None
            save_best_probe(step, "final_train_loss", final_loss)
    artifacts.finish(status="early_stopped" if stopped_early else "completed", final_step=step, final_train_loss=final_loss, best_step=best_step, best_validation_loss=best_validation_loss, early_stop_loss=early_stop_loss)


if __name__ == "__main__":
    main()
