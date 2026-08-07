from __future__ import annotations

import argparse
from functools import partial
from typing import Any

import torch
from tqdm import tqdm

from qkd.modeling import load_causal_lm, load_tokenizer
from qkd.photonic.model import find_decoder_layers, make_compressed_student
from qkd.training.artifacts import TrainingArtifacts, best_probe_payload, stage1_photonic_log_fields
from qkd.training.spsa import SPSA
from qkd.training.tools import StageOneReference, apply_overrides, load_config, make_loader, next_batch, provider_factory, section, training_device


def arguments() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="阶段一：单层光子条件化低秩 MLP 重建")
    parser.add_argument("--config", required=True, help="阶段一 YAML 配置文件")
    parser.add_argument("--set", action="append", default=[], metavar="键路径=值", help="临时覆盖 YAML；可重复，例如 'model.target_layers=[22]'。")
    return parser.parse_args()


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
    artifacts = TrainingArtifacts.create(config, "stage1", device, stage1_photonic_log_fields(n_modes, n_layers))

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
    student, replacements = make_compressed_student(
        teacher, provider_factory(str(photonic["provider"]), z_dim, photonic.get("ema_decay"), n_modes, n_layers),
        rank, z_dim, kappa, target_layers,
    )
    replacement = replacements[0]
    replacement.shots = photonic.get("shots")
    student.to(device).train()
    optimizer = torch.optim.Adam(
        (*replacement.adam_parameters(), *replacement.photonic_parameters()),
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
    best_validation_loss = float("inf")
    best_step: int | None = None
    final_loss: float | None = None

    # 3. 局部重建损失、验证与最佳 probe
    with StageOneReference(teacher, teacher_layers[target].mlp, replacement) as reference:

        @torch.no_grad()
        def save_best_probe(step: int, metric_name: str, metric_value: float | None) -> None:
            student.eval()
            try:
                details = reference.terms(probe_batch)
                if reference.values is None:
                    raise RuntimeError("教师 MLP hook 未捕获到输出")
                _, teacher_y = reference.values
                artifacts.save_best_probe(best_probe_payload("stage1", step, metric_name, metric_value, target_layers, probe_batch, {target: teacher_y}, {target: details["output"]}))
            finally:
                student.train()

        # 4. 训练循环：Adam 仅更新 C/b 与 DeepQuantum 可微光路参数；P/B 保持冻结。
        iterator = iter(train_loader)
        progress = tqdm(range(1, int(optimization["steps"]) + 1), desc=f"Stage 1 · layer {target}", unit="step")
        for step in progress:
            batch, iterator = next_batch(iterator, train_loader)
            batch = {key: value.to(device) for key, value in batch.items()}
            details = reference.terms(batch)
            details["loss"].backward()
            optimizer.step()
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
                **{name: value for name, value in diagnostics.items() if name != "valid_tokens"},
                "validation_loss": "",
                "validation_y_nmse": "",
                "is_best": False,
                "spsa_applied": False,
                "early_stopped": False,
                **replacement.photonic_parameter_values(),
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
