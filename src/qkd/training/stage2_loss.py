"""阶段二端到端知识蒸馏的 CE 与 Top-K KD 联合损失。"""

from __future__ import annotations

import torch
import torch.nn.functional as F
from torch import Tensor


def stage_two_loss(
    student_logits: Tensor, teacher_logits: Tensor, labels: Tensor, temperature: float, top_k: int
) -> dict[str, Tensor]:
    """计算 0.5 CE + 0.5 tau² KD。"""
    ce = torch.nn.functional.cross_entropy(
        student_logits[:, :-1].reshape(-1, student_logits.size(-1)),
        labels[:, 1:].reshape(-1),
        ignore_index=-100,
    )
    kd = expanded_topk_logit_kl(student_logits, teacher_logits, labels, temperature, top_k)
    return {"loss": 0.5 * ce + 0.5 * kd, "ce": ce, "kd": kd}


def expanded_topk_indices(
    teacher_logits: torch.Tensor, labels: torch.Tensor, top_k: int
) -> torch.Tensor:
    """返回教师 Top-K 再加真实 token 的 K+1 个候选索引。"""
    teacher = teacher_logits[:, :-1]
    targets = labels[:, 1:]
    indices = teacher.topk(min(top_k, teacher.size(-1)), dim=-1).indices
    target_indices = targets.clamp_min(0).unsqueeze(-1)
    return torch.cat([indices, target_indices], dim=-1)


def expanded_topk_logit_kl(
    student_logits: torch.Tensor,
    teacher_logits: torch.Tensor,
    labels: torch.Tensor,
    temperature: float = 2.0,
    top_k: int = 64,
) -> torch.Tensor:
    """在扩展教师 Top-K logits 上计算右移且掩码化的 KD。"""
    teacher_positions = teacher_logits[:, :-1]
    indices = expanded_topk_indices(teacher_logits, labels, top_k)
    # 先收集 Top-K + 真实 token，再转 FP32。对全词表 logits 先 ``.float()``
    # 会为 [batch, sequence, vocab] 额外复制数 GB 显存；27B 教师与全 32 层
    # 学生同卡时容易 OOM。gather 对 FP16/BF16 logits 等价，之后的 FP32 用于
    # softmax / KL 的数值稳定性。
    student = student_logits[:, :-1].gather(-1, indices).float() / temperature
    teacher = teacher_positions.gather(-1, indices).float() / temperature
    valid = labels[:, 1:].ne(-100)
    # 真实 token 已在 Top-K 中时，末尾的重复候选必须被屏蔽，不能改变原 Top-K 分布。
    true_token = labels[:, 1:].clamp_min(0).unsqueeze(-1)
    topk_size = indices.size(-1) - 1
    present = indices[..., :topk_size].eq(true_token).any(dim=-1, keepdim=True)
    candidate_mask = torch.cat(
        [torch.ones_like(present).expand_as(indices[..., :topk_size]), ~present], dim=-1
    )
    # 使用有限的最小值而非 -inf，避免某些后端在 KL 内部出现 0 * inf 的 NaN。
    minimum = torch.finfo(student.dtype).min
    student = student.masked_fill(~candidate_mask, minimum)
    teacher = teacher.masked_fill(~candidate_mask, minimum)
    per_token = F.kl_div(F.log_softmax(student, -1), F.softmax(teacher, -1), reduction="none").sum(
        -1
    )
    return (per_token * valid).sum() / valid.sum().clamp_min(1) * temperature**2
