"""阶段一局部 MLP 重建损失与 gate/up/down 诊断指标。"""

from __future__ import annotations

import torch
import torch.nn.functional as F
from torch import Tensor

from qkd.photonic.model import PhotonicLowRankMLP


def stage_one_loss(student_mlp: PhotonicLowRankMLP, x: Tensor, teacher_gate: Tensor, teacher_up: Tensor, teacher_output: Tensor, attention_mask: Tensor) -> dict[str, Tensor]:
    """返回第一阶段 MLP 损失及实时诊断所需的低维汇总依据。"""
    mask = attention_mask.bool()
    gate = student_mlp.gate(x, student_mlp.gate_provider, student_mlp.theta_gate, student_mlp.shots)
    up = student_mlp.up(x, student_mlp.up_provider, student_mlp.theta_up, student_mlp.shots)
    value = torch.nn.functional.silu(gate) * up
    down = student_mlp.down(value, student_mlp.down_provider, student_mlp.theta_down, student_mlp.shots)
    teacher_value = torch.nn.functional.silu(teacher_gate) * teacher_up
    teacher_down = student_mlp.down(teacher_value, student_mlp.down_provider, student_mlp.theta_down, student_mlp.shots)
    gate_loss = relative_mse_cosine(gate, teacher_gate, mask)
    up_loss = relative_mse_cosine(up, teacher_up, mask)
    output_loss = relative_mse_cosine(down, teacher_output, mask)
    down_loss = relative_mse_cosine(teacher_down, teacher_output, mask)
    return {  # 阶段一只优化最终 MLP 输出；gate/up/down 的单独误差仅用于监测。
        "loss": output_loss,
        "gate_loss": gate_loss,
        "up_loss": up_loss,
        "output_loss": output_loss,
        "down_loss": down_loss,
        "gate": gate,
        "up": up,
        "down": teacher_down,
        "output": down,
    }


def relative_mse_cosine(prediction: torch.Tensor, target: torch.Tensor, mask: torch.Tensor, epsilon: float = 1e-8) -> torch.Tensor:
    """掩码归一化 MSE 加余弦重建惩罚。"""
    prediction, target = prediction[mask].float(), target[mask].float()  # 模型权重可为 bf16，但损失必须在 fp32 中完成，避免平台期被低精度量化。
    if target.numel() == 0:
        return prediction.new_zeros(())
    mse = (prediction - target).square().sum() / (target.square().sum() + epsilon)
    cosine = F.cosine_similarity(prediction, target, dim=-1).mean()
    return mse + 0.1 * (1 - cosine)


@torch.no_grad()
def local_diagnostics(details: dict[str, Tensor], teacher_gate: Tensor, teacher_up: Tensor, teacher_output: Tensor, attention_mask: Tensor) -> dict[str, float]:
    """将长向量压缩为终端显示用的误差、余弦和均值统计量。"""
    mask = attention_mask.bool()

    def summarize(name: str, prediction: Tensor, target: Tensor) -> dict[str, float]:
        prediction, target = prediction[mask].float(), target[mask].float()
        delta = prediction - target
        return {
            f"{name}_mae": delta.abs().mean().item(),
            f"{name}_nmse": (delta.square().sum() / target.square().sum().clamp_min(1e-8)).item(),
            f"{name}_cos": torch.nn.functional.cosine_similarity(prediction, target, dim=-1).mean().item(),
            f"{name}_student_mean": prediction.mean().item(),
            f"{name}_teacher_mean": target.mean().item(),
        }

    result = {"valid_tokens": float(mask.sum().item())}
    result.update(summarize("gate", details["gate"], teacher_gate))
    result.update(summarize("up", details["up"], teacher_up))
    result.update(summarize("down", details["down"], teacher_output))
    result.update(summarize("y", details["output"], teacher_output))
    return result
