"""阶段一局部 MLP 重建的纯张量损失与诊断指标。"""

from __future__ import annotations

import torch
import torch.nn.functional as F
from torch import Tensor


def stage_one_loss(
    student_gate: Tensor,
    student_up: Tensor,
    student_output: Tensor,
    projected_teacher_value: Tensor,
    teacher_gate: Tensor,
    teacher_up: Tensor,
    teacher_output: Tensor,
    attention_mask: Tensor,
) -> dict[str, Tensor]:
    """计算第一阶段局部重建损失及诊断所需的张量。"""
    mask = attention_mask.bool()
    gate_loss = relative_mse_cosine(student_gate, teacher_gate, mask)
    up_loss = relative_mse_cosine(student_up, teacher_up, mask)
    output_loss = relative_mse_cosine(student_output, teacher_output, mask)
    down_loss = relative_mse_cosine(projected_teacher_value, teacher_output, mask)
    return {  # 阶段一只优化最终 MLP 输出；gate/up/down 的单独误差仅用于监测。
        "loss": output_loss,
        "gate_loss": gate_loss,
        "up_loss": up_loss,
        "output_loss": output_loss,
        "down_loss": down_loss,
        "gate": student_gate,
        "up": student_up,
        "down": projected_teacher_value,
        "output": student_output,
    }


def relative_mse_cosine(
    prediction: torch.Tensor, target: torch.Tensor, mask: torch.Tensor, epsilon: float = 1e-8
) -> torch.Tensor:
    """掩码归一化 MSE 加余弦重建惩罚。"""
    prediction, target = (
        prediction[mask].float(),
        target[mask].float(),
    )  # 模型权重可为 bf16，但损失必须在 fp32 中完成，避免平台期被低精度量化。
    if target.numel() == 0:
        return prediction.new_zeros(())
    mse = (prediction - target).square().sum() / (target.square().sum() + epsilon)
    cosine = F.cosine_similarity(prediction, target, dim=-1).mean()
    return mse + 0.1 * (1 - cosine)


@torch.no_grad()
def local_diagnostics(
    details: dict[str, Tensor],
    teacher_gate: Tensor,
    teacher_up: Tensor,
    teacher_output: Tensor,
    attention_mask: Tensor,
) -> dict[str, float]:
    """将长向量压缩为终端显示用的误差、余弦和均值统计量。"""
    mask = attention_mask.bool()

    def summarize(name: str, prediction: Tensor, target: Tensor) -> dict[str, float]:
        prediction, target = prediction[mask].float(), target[mask].float()
        delta = prediction - target
        return {
            f"{name}_mae": delta.abs().mean().item(),
            f"{name}_nmse": (delta.square().sum() / target.square().sum().clamp_min(1e-8)).item(),
            f"{name}_cos": torch.nn.functional.cosine_similarity(prediction, target, dim=-1)
            .mean()
            .item(),
            f"{name}_student_mean": prediction.mean().item(),
            f"{name}_teacher_mean": target.mean().item(),
        }

    result = {"valid_tokens": float(mask.sum().item())}
    result.update(summarize("gate", details["gate"], teacher_gate))
    result.update(summarize("up", details["up"], teacher_up))
    result.update(summarize("down", details["down"], teacher_output))
    result.update(summarize("y", details["output"], teacher_output))
    return result
