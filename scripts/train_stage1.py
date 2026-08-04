from __future__ import annotations

import argparse
from contextlib import ExitStack
from pathlib import Path

import torch

from qkd.modeling import load_causal_lm, load_tokenizer
from qkd.photonic.checkpoint import save_compressed_checkpoint
from qkd.photonic.model import find_decoder_layers, make_compressed_student
from qkd.training.tools import apply_overrides, load_config, make_loader, next_batch, provider_factory, section, training_device, write_run_config
from qkd.training.spsa import SPSA
from qkd.training.stage1_loss import local_diagnostics, stage_one_loss


def arguments() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="阶段一：单层光子条件化低秩 MLP 重建")
    parser.add_argument("--config", required=True, help="阶段一 YAML 配置文件")
    parser.add_argument("--set", action="append", default=[], metavar="键路径=值", help="临时覆盖 YAML；可重复，例如 'model.target_layers=[22]'。")
    return parser.parse_args()


def main() -> None:
    args = arguments()
    config = apply_overrides(load_config(args.config), args.set)
    experiment, data = section(config, "experiment"), section(config, "data")
    model, compression = section(config, "model"), section(config, "compression")
    photonic, optimization, logging = section(config, "photonic"), section(config, "optimization"), section(config, "logging")
    target_layers = tuple(int(index) for index in model["target_layers"])
    if len(target_layers) != 1:
        raise ValueError("阶段一配置一次只能指定一个 target_layers，例如 [21]")

    torch.manual_seed(int(experiment["seed"]))
    device = training_device()
    output = Path(experiment["output_dir"])
    teacher_name = str(model["teacher"])
    print(f"运行设备：{device}")
    write_run_config(output, config, "stage1", device, {"status": "running", "teacher": teacher_name})

    tokenizer = load_tokenizer(teacher_name)
    train_loader = make_loader(str(data["train_data"]), tokenizer, int(data["batch_size"]), True)
    validation_loader = make_loader(str(data["validation_data"]), tokenizer, int(data["batch_size"]), False)
    teacher = load_causal_lm(teacher_name, trainable=False).to(device).eval()
    teacher_layers = find_decoder_layers(teacher)
    target = target_layers[0]
    if target < 0 or target >= len(teacher_layers):
        raise ValueError(f"目标层必须在 0 到 {len(teacher_layers) - 1} 之间")
    rank = int(compression["rank"])
    z_dim = int(compression["z_dim"])
    kappa = float(compression["kappa"])
    student, replacements = make_compressed_student(
        teacher, provider_factory(str(photonic["provider"]), z_dim, photonic.get("ema_decay")),
        rank, z_dim, kappa, target_layers,
    )
    replacement = replacements[0]
    replacement.shots = photonic.get("shots")
    student.to(device).train()
    optimizer = torch.optim.Adam(replacement.adam_parameters(), lr=float(optimization["adam_learning_rate"]))
    spsa = SPSA(perturbation=float(optimization["spsa_perturbation"]), learning_rate=float(optimization["spsa_learning_rate"]), seed=int(experiment["seed"]))
    captured: dict[str, tuple[torch.Tensor, torch.Tensor]] = {}
    best_validation_loss = float("inf")

    with ExitStack() as stack:
        old_mlp = teacher_layers[target].mlp

        def save_input(module, inputs, output_value) -> None:
            captured["values"] = (inputs[0].detach(), output_value.detach())

        stack.callback(old_mlp.register_forward_hook(save_input).remove)

        def details_for(batch: dict[str, torch.Tensor]) -> dict[str, torch.Tensor]:
            captured.clear()
            with torch.no_grad():
                teacher(input_ids=batch["input_ids"], attention_mask=batch["attention_mask"])
                x, teacher_output = captured["values"]
                gate, up = old_mlp.gate_proj(x), old_mlp.up_proj(x)
            return stage_one_loss(replacement, x, gate, up, teacher_output, batch["attention_mask"])

        @torch.no_grad()
        def validate() -> tuple[float, dict[str, float]]:
            student.eval()
            weighted_loss, token_count, totals = 0.0, 0.0, {}
            for batch in validation_loader:
                batch = {key: value.to(device) for key, value in batch.items()}
                details = details_for(batch)
                x, teacher_output = captured["values"]
                gate, up = old_mlp.gate_proj(x), old_mlp.up_proj(x)
                diagnostics = local_diagnostics(details, gate, up, teacher_output, batch["attention_mask"])
                valid = diagnostics["valid_tokens"]
                weighted_loss += details["loss"].float().item() * valid
                token_count += valid
                for name, value in diagnostics.items():
                    if name != "valid_tokens":
                        totals[name] = totals.get(name, 0.0) + value * valid
            student.train()
            return weighted_loss / token_count, {name: value / token_count for name, value in totals.items()}

        iterator = iter(train_loader)
        for step in range(1, int(optimization["steps"]) + 1):
            batch, iterator = next_batch(iterator, train_loader)
            batch = {key: value.to(device) for key, value in batch.items()}
            details = details_for(batch)
            details["loss"].backward()
            optimizer.step()
            optimizer.zero_grad()

            if step % int(logging["visualize_every"]) == 0:
                x, teacher_output = captured["values"]
                with torch.no_grad():
                    gate, up = old_mlp.gate_proj(x), old_mlp.up_proj(x)
                metrics = local_diagnostics(details, gate, up, teacher_output, batch["attention_mask"])
                print(
                    f"stage1 layer={target} step={step} y_loss={details['loss'].detach().float().item():.4f} "
                    f"gate[nmse={metrics['gate_nmse']:.4f},cos={metrics['gate_cos']:.4f}] "
                    f"up[nmse={metrics['up_nmse']:.4f},cos={metrics['up_cos']:.4f}] "
                    f"down[nmse={metrics['down_nmse']:.4f},cos={metrics['down_cos']:.4f}] "
                    f"y[nmse={metrics['y_nmse']:.4f},cos={metrics['y_cos']:.4f},mae={metrics['y_mae']:.4f}]"
                )

            if step % int(optimization["spsa_every"]) == 0:
                validation = next(iter(validation_loader))
                validation = {key: value.to(device) for key, value in validation.items()}

                def objective() -> torch.Tensor:
                    student.eval()
                    try:
                        return details_for(validation)["loss"]
                    finally:
                        student.train()

                for theta in (replacement.theta_gate, replacement.theta_up, replacement.theta_down):
                    spsa.step(theta, objective)

            if int(logging["validate_every"]) and step % int(logging["validate_every"]) == 0:
                validation_loss, metrics = validate()
                print(f"stage1 validation layer={target} step={step} y_loss={validation_loss:.4f} y_nmse={metrics['y_nmse']:.4f}")
                if validation_loss < best_validation_loss:
                    best_validation_loss = validation_loss
                    best_output = output / "best"
                    save_compressed_checkpoint(best_output, replacements, rank, z_dim, kappa, target_layers, teacher_name, str(photonic["provider"]))
                    tokenizer.save_pretrained(best_output)
                    write_run_config(best_output, config, "stage1", device, {
                        "status": "best_validation_checkpoint", "teacher": teacher_name,
                        "step": step, "validation_y_loss": validation_loss,
                    })

    save_compressed_checkpoint(output, replacements, rank, z_dim, kappa, target_layers, teacher_name, str(photonic["provider"]))
    tokenizer.save_pretrained(output)
    write_run_config(output, config, "stage1", device, {"status": "completed", "teacher": teacher_name, "best_validation_y_loss": best_validation_loss,})
    print(f"已保存 layer {target} 的阶段一模块至 {output}")


if __name__ == "__main__":
    main()
