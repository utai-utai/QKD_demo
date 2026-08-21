"""供输入条件化低秩 MLP 使用的光子特征提供器。"""

from __future__ import annotations

from abc import ABC, abstractmethod

import torch
from torch import Tensor, nn

from .circuit import clements_layout, clements_single_photon_probabilities


class PhotonicFeatureProvider(nn.Module, ABC):
    @abstractmethod
    def sample(self, encoded: Tensor, shots: int | None, device: torch.device) -> Tensor:
        """返回与输入末维相同的输入条件化特征。"""


class MockPhotonicFeatureProvider(PhotonicFeatureProvider):
    """用于 CPU 测试与早期实验的确定性可微分层光子替代器。"""
    def __init__(self, z_dim: int = 16, ema_decay: float | None = 0.9, n_modes: int = 16, n_layers: int = 16, n_meshes: int = 1, theta_init_std: float = 0.1, phi_init_std: float = 0.1) -> None:
        super().__init__()
        if z_dim != n_modes:
            raise ValueError("Clements 相移编码要求 compression.z_dim 等于 photonic.modes")
        pairs, n_mzi = clements_layout(n_modes)
        if n_layers != n_modes:
            raise ValueError("Clements 网格固定为 photonic.layers == photonic.modes")
        if n_meshes not in (1, 2):
            raise ValueError("当前仅支持 1 或 2 个串联 Clements 网格")
        self.z_dim = z_dim
        self.n_modes = n_modes
        self.n_layers = n_layers
        self.n_meshes = n_meshes
        self.pairs = pairs
        self.ema_decay = ema_decay
        # 每个 provider 由 1 或 2 个 U(N) Clements 网格串联组成。
        # 50:50 MZI 起点确保输入相位从第一步就能通过干涉改变探测概率；
        # theta=0 会使网格退化为路径置换，导致 phase encoding 的梯度消失。
        if theta_init_std < 0 or phi_init_std < 0:
            raise ValueError("theta_init_std 与 phi_init_std 必须非负")
        n_mzi_total = n_meshes * n_mzi
        self.theta = nn.Parameter(torch.full((n_mzi_total,), torch.pi / 4) + torch.randn(n_mzi_total) * theta_init_std)
        self.phi = nn.Parameter(torch.randn(n_mzi_total) * phi_init_std)
        # For two meshes, this is the first mesh's output PS and genuinely
        # changes the second mesh's interference.  The final output PS remains
        # a buffer because photon-number probabilities cannot observe it.
        self.intermediate_phase = nn.Parameter(torch.randn(n_meshes - 1, n_modes) * phi_init_std)
        # 探测器前的末端 PS 属于完整 U(N) 网格的物理组成，但 photon-number
        # 概率对它不变（|exp(i alpha) b|^2 = |b|^2），因此将其保留为硬件布局
        # 参数而不交给优化器，避免无意义的零梯度参数。
        self.register_buffer("output_phase", torch.zeros(n_modes), persistent=True)

    def _postprocess(self, number: Tensor, phase: Tensor, shots: int | None, device: torch.device) -> Tensor:
        # Clements 单光子测量本身就是所需的 16 维概率向量，严格满足 sum=1。
        features = number
        if shots is not None:
            if shots < 1:
                raise ValueError("shots must be positive or None")
            # 当前训练使用概率期望值（shots=None）。有限 shots 时用可微的近似
            # 扰动仅作硬件噪声实验，并重新归一化为概率向量。
            noise = torch.sin(features.detach() * 12.9898 + phase.detach() * 78.233)
            features = (features + noise / float(shots) ** 0.5).clamp_min(0)
            features = features / features.sum(dim=-1, keepdim=True).clamp_min(1e-8)
        return features.to(device)

    def _mock_photon_numbers(self, encoded: Tensor) -> Tensor:
        """与 DeepQuantum 拓扑一致的可微快速替代器，用于 CPU 冒烟测试。"""
        # 快速 mock 与 Fock Clements 拓扑一致：均匀单光子路径态、输入相移、
        # 120 个 MZI、输出相移，最后取 |amplitude|^2。
        amplitudes = torch.ones_like(encoded, dtype=torch.complex64) / self.n_modes**0.5
        amplitudes = amplitudes * torch.exp(1j * torch.tanh(encoded).to(torch.complex64) * torch.pi)
        n_mzi = len(self.pairs)
        for mesh_index in range(self.n_meshes):
            offset = mesh_index * n_mzi
            for pair_index, (wire_a, wire_b) in enumerate(self.pairs):
                angle, phase = self.theta[offset + pair_index], self.phi[offset + pair_index]
                values = list(amplitudes.unbind(dim=-1))
                source_a, source_b = values[wire_a], values[wire_b]
                scale = 1j * torch.exp(0.5j * angle)
                values[wire_a] = scale * (torch.exp(1j * phase) * torch.sin(angle / 2) * source_a + torch.cos(angle / 2) * source_b)
                values[wire_b] = scale * (torch.exp(1j * phase) * torch.cos(angle / 2) * source_a - torch.sin(angle / 2) * source_b)
                amplitudes = torch.stack(values, dim=-1)
            if mesh_index < self.n_meshes - 1:
                amplitudes = amplitudes * torch.exp(1j * self.intermediate_phase[mesh_index].to(torch.complex64))
        return (amplitudes * torch.exp(1j * self.output_phase)).abs().square()

    def sample(self, encoded: Tensor, shots: int | None, device: torch.device) -> Tensor:
        if encoded.shape[-1] != self.z_dim:
            raise ValueError(f"期望 encoded[..., {self.z_dim}]，实际为 {tuple(encoded.shape)}")
        number = self._mock_photon_numbers(encoded)
        return self._postprocess(number, self.output_phase.mean(), shots, device)


class DeepQuantumCVFeatureProvider(MockPhotonicFeatureProvider):
    """使用 DeepQuantum 单光子 Clements Fock 模拟器的可微分特征提供器。"""
    def __init__(self, z_dim: int = 16, ema_decay: float | None = 0.9, n_modes: int = 16, n_layers: int = 16, n_meshes: int = 1, theta_init_std: float = 0.1, phi_init_std: float = 0.1) -> None:
        super().__init__(z_dim=z_dim, ema_decay=ema_decay, n_modes=n_modes, n_layers=n_layers, n_meshes=n_meshes, theta_init_std=theta_init_std, phi_init_std=phi_init_std)
        import deepquantum  # noqa: F401 -- 在训练开始时确认依赖可用。

    def sample(self, encoded: Tensor, shots: int | None, device: torch.device) -> Tensor:
        probability = clements_single_photon_probabilities(
            encoded, self.theta, self.phi, self.output_phase, self.n_modes, self.intermediate_phase
        )
        return self._postprocess(probability, self.output_phase.mean(), shots, device)


class ClassicalFCFeatureProvider(PhotonicFeatureProvider):
    """Simplex-aligned parameter-matched classical conditioning baseline."""

    def __init__(self, z_dim: int = 16, ema_decay: float | None = 0.9, n_modes: int = 16, n_layers: int = 16, n_meshes: int = 1, theta_init_std: float = 0.1, phi_init_std: float = 0.1) -> None:
        super().__init__()
        del ema_decay, n_layers, theta_init_std, phi_init_std
        if z_dim != 16 or n_modes != 16:
            raise ValueError("参数匹配经典基线当前固定为 16 维输入/输出")
        self.z_dim = z_dim
        if n_meshes == 1:
            # 16*7 + 7*16 + 16 = 240, matching one mesh theta/phi.
            self.fc_in = nn.Linear(16, 7, bias=False)
            self.fc_out = nn.Linear(7, 16, bias=True)
            self.logit_bias = None
            self.simplex_output = False  # preserve previously trained 1-mesh checkpoints
        elif n_meshes == 2:
            # 16*15 + 15*16 + 16 = 496.  This matches the observable
            # parameters of two meshes: 480 MZI theta/phi + 16 inter-mesh PS.
            self.fc_in = nn.Linear(16, 15, bias=False)
            self.fc_out = nn.Linear(15, 16, bias=False)
            self.logit_bias = nn.Parameter(torch.zeros(16))
            self.simplex_output = True
        else:
            raise ValueError("当前经典基线仅支持 1 或 2 个 Clements 网格")

    def sample(self, encoded: Tensor, shots: int | None, device: torch.device) -> Tensor:
        if shots is not None:
            raise ValueError("classical provider does not support photon shots")
        if encoded.shape[-1] != self.z_dim:
            raise ValueError(f"期望 encoded[..., {self.z_dim}]，实际为 {tuple(encoded.shape)}")
        # 4B 教师的 hidden states 通常是 bf16，而新建的经典基线权重保持 fp32；
        # 在线性层内部统一 fp32，随后 ConditionedLowRankLinear 会转回其模型 dtype。
        features = self.fc_out(torch.nn.functional.gelu(self.fc_in(encoded.to(self.fc_in.weight.dtype))))
        if self.logit_bias is not None:
            features = features + self.logit_bias
        if self.simplex_output:
            # Match the two-mesh photonic provider exactly: non-negative
            # 16-port features summing to one for every token.
            features = torch.softmax(features, dim=-1)
        return features.to(device)
