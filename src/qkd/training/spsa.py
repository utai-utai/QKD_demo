"""光子低秩压缩蒸馏的损失函数基础组件。"""

from __future__ import annotations

import torch
from torch import Tensor, nn
from typing import Callable


class SPSA:
    """阶段一和阶段二共用的 theta 无梯度随机扰动优化器。"""
    def __init__(self, perturbation: float, learning_rate: float, seed: int) -> None:
        self.perturbation = perturbation
        self.learning_rate = learning_rate
        self.generator = torch.Generator(device="cpu").manual_seed(seed)

    @torch.no_grad()
    def step(self, theta: nn.Parameter, objective: Callable[[], Tensor]) -> tuple[Tensor, Tensor]:
        if theta.requires_grad:
            raise ValueError("theta 必须冻结，SPSA 是其唯一更新方式")
        delta = torch.randint(0, 2, theta.shape, generator=self.generator, device="cpu", dtype=torch.int64)
        delta = (delta.mul_(2).sub_(1)).to(theta)
        original = theta.detach().clone()  # 在 CPU 上生成 Rademacher 随机扰动向量 delta (取值为 +1 或 -1)
        theta.copy_(original + self.perturbation * delta)
        plus = objective().detach()  # 正向扰动 (+c * delta) 并评估 Target Objective Loss
        theta.copy_(original - self.perturbation * delta)
        minus = objective().detach()  # 负向扰动 (-c * delta) 并评估 Target Objective Loss
        gradient = (plus - minus) / (2 * self.perturbation) * delta  # 双边有限差分估计数值梯度：g ≈ (L+ - L-) / (2 * c * delta)
        theta.copy_(original - self.learning_rate * gradient)
        return plus, minus
