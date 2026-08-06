"""供输入条件化低秩 MLP 使用的光子特征提供器。"""

from __future__ import annotations

from abc import ABC, abstractmethod

import torch
from torch import Tensor, nn

from .circuit import circuit_layout, layered_photon_numbers


class PhotonicFeatureProvider(nn.Module, ABC):
    @abstractmethod
    def sample(self, encoded: Tensor, shots: int | None, device: torch.device) -> Tensor:
        """返回与输入末维相同的输入条件化特征。"""


class MockPhotonicFeatureProvider(PhotonicFeatureProvider):
    """用于 CPU 测试与早期实验的确定性可微分层光子替代器。"""
    def __init__(self, z_dim: int = 16, ema_decay: float | None = 0.9, n_modes: int = 8, n_layers: int = 4) -> None:
        super().__init__()
        if z_dim != 2 * n_modes:
            raise ValueError("compression.z_dim 必须等于 2 * photonic.modes")
        pairs, n_theta, n_phi = circuit_layout(n_modes, n_layers)
        self.z_dim = z_dim
        self.n_modes = n_modes
        self.n_layers = n_layers
        self.pairs = pairs
        self.ema_decay = ema_decay
        # 每个 provider 都拥有一套独立的分层 BS(theta)→PS(phi) 参数。
        self.theta = nn.Parameter(torch.zeros(n_theta))
        self.phi = nn.Parameter(torch.zeros(n_phi))
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
            batch_mean = features.detach().float().mean(dim=tuple(range(features.ndim - 1)))
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

    def _mock_photon_numbers(self, encoded: Tensor) -> Tensor:
        """与 DeepQuantum 拓扑一致的可微快速替代器，用于 CPU 冒烟测试。"""
        amplitudes = torch.complex(torch.tanh(encoded[..., : self.n_modes]), torch.zeros_like(encoded[..., : self.n_modes]))
        theta_index = 0
        for layer in range(self.n_layers):
            for wire_a, wire_b in self.pairs:
                angle = self.theta[theta_index]
                theta_index += 1
                values = list(amplitudes.unbind(dim=-1))
                source_a, source_b = values[wire_a], values[wire_b]
                values[wire_a] = torch.cos(angle) * source_a - torch.sin(angle) * source_b
                values[wire_b] = torch.sin(angle) * source_a + torch.cos(angle) * source_b
                amplitudes = torch.stack(values, dim=-1)
            amplitudes = amplitudes * torch.exp(1j * self.phi[layer * self.n_modes : (layer + 1) * self.n_modes])
        return amplitudes.abs().square()

    def sample(self, encoded: Tensor, shots: int | None, device: torch.device) -> Tensor:
        if encoded.shape[-1] != self.z_dim:
            raise ValueError(f"期望 encoded[..., {self.z_dim}]，实际为 {tuple(encoded.shape)}")
        number = self._mock_photon_numbers(encoded)
        return self._postprocess(number, self.phi.mean(), shots, device)


class DeepQuantumCVFeatureProvider(MockPhotonicFeatureProvider):
    """使用 DeepQuantum 高斯模拟器的可微分层光子特征提供器。"""
    def __init__(self, z_dim: int = 16, ema_decay: float | None = 0.9, n_modes: int = 8, n_layers: int = 4) -> None:
        super().__init__(z_dim=z_dim, ema_decay=ema_decay, n_modes=n_modes, n_layers=n_layers)
        import deepquantum  # noqa: F401 -- 在训练开始时确认依赖可用。

    def sample(self, encoded: Tensor, shots: int | None, device: torch.device) -> Tensor:
        number = layered_photon_numbers(encoded, self.theta, self.phi, self.n_modes, self.n_layers)
        return self._postprocess(number, self.phi.mean(), shots, device)
