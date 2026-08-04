"""供输入条件化低秩 MLP 使用的光子特征提供器。"""

from __future__ import annotations

from abc import ABC, abstractmethod

import torch
from torch import Tensor, nn

from .circuit import eight_mode_ring_photon_numbers


class PhotonicFeatureProvider(nn.Module, ABC):
    @abstractmethod
    def sample(self, encoded: Tensor, theta: Tensor, shots: int | None, device: torch.device) -> Tensor:
        """返回与输入末维相同的输入条件化特征。"""


class MockPhotonicFeatureProvider(PhotonicFeatureProvider):
    """用于 CPU 测试与早期实验的确定性八模连续变量光子替代器。"""
    def __init__(self, z_dim: int = 16, ema_decay: float | None = 0.9) -> None:
        super().__init__()
        if z_dim != 16:
            raise ValueError("the default photonic observable layout requires z_dim=16")
        self.z_dim = z_dim
        self.ema_decay = ema_decay
        # 固定形状且写入 checkpoint，保证中断恢复后 EMA 统计连续。
        self.register_buffer("ema", torch.zeros(z_dim), persistent=True)
        self.register_buffer("ema_initialized", torch.tensor(False), persistent=True)

    def _postprocess(self, number: Tensor, phase: Tensor, shots: int | None, device: torch.device) -> Tensor:
        centered = number - number.mean(dim=-1, keepdim=True)
        correlation = centered * centered.roll(shifts=-1, dims=-1)
        features = torch.cat([number, correlation], dim=-1)
        if shots is not None:
            if shots < 1:
                raise ValueError("shots must be positive or None")
            noise = torch.sin(features.detach() * 12.9898 + phase.detach() * 78.233)
            features = features + noise / float(shots) ** 0.5
        if self.ema_decay is not None:
            batch_mean = features.detach().mean(dim=tuple(range(features.ndim - 1)))
            # 验证与 SPSA 目标评估必须是无副作用的；只有训练模式更新 EMA。
            if self.training:
                if not self.ema_initialized:
                    self.ema.copy_(batch_mean)
                    self.ema_initialized.fill_(True)
                else:
                    self.ema.lerp_(batch_mean, 1 - self.ema_decay)
            center = self.ema if self.ema_initialized else batch_mean
            features = center.view(*([1] * (features.ndim - 1)), -1) + (features - batch_mean)
        return features.to(device)

    def sample(self, encoded: Tensor, theta: Tensor, shots: int | None, device: torch.device) -> Tensor:
        if encoded.shape[-1] != self.z_dim:
            raise ValueError(f"期望 encoded[..., {self.z_dim}]，实际为 {tuple(encoded.shape)}")
        theta_term = theta.detach().reshape(-1)
        phase = theta_term.mean() if theta_term.numel() else encoded.new_zeros(())
        displacement = torch.tanh(encoded + phase)
        return self._postprocess(displacement[..., :8].square(), phase, shots, device)


class DeepQuantumCVFeatureProvider(MockPhotonicFeatureProvider):
    """使用八模高斯环形连续变量电路的 DeepQuantum 特征提供器。"""
    def __init__(self, z_dim: int = 16, ema_decay: float | None = 0.9) -> None:
        super().__init__(z_dim=z_dim, ema_decay=ema_decay)
        import deepquantum  # noqa: F401 -- 在训练开始时确认依赖可用。

    def sample(self, encoded: Tensor, theta: Tensor, shots: int | None, device: torch.device) -> Tensor:
        number = eight_mode_ring_photon_numbers(encoded, theta)
        return self._postprocess(number, theta.detach().mean(), shots, device)
