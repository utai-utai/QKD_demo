"""用光子条件化低秩 MLP 替换 Qwen 的 SwiGLU MLP。"""

from __future__ import annotations

from collections.abc import Iterator
from copy import deepcopy
from typing import Callable

import torch
import torch.nn.functional as F
from torch import Tensor, nn

from .provider import PhotonicFeatureProvider


def find_decoder_layers(model: nn.Module) -> nn.ModuleList:
    """动态查找实际 decoder 的 ``ModuleList``。"""
    candidates: list[nn.ModuleList] = []
    for module in model.modules():
        for child in module.children():
            if isinstance(child, nn.ModuleList) and child and all(hasattr(layer, "mlp") for layer in child):
                candidates.append(child)
    if not candidates:
        raise ValueError("could not locate decoder layers: no ModuleList of layers with an mlp attribute")
    expected = getattr(getattr(model, "config", None), "num_hidden_layers", None)
    matching = [layers for layers in candidates if len(layers) == expected] if expected else candidates
    return max(matching or candidates, key=len)


def _svd_pb(linear: nn.Module | Tensor, rank: int) -> tuple[Tensor, Tensor]:
    weight = linear.weight.detach().float().cpu()
    usable_rank = min(rank, *weight.shape)
    u, singular, vh = torch.linalg.svd(weight, full_matrices=False)
    root = singular[:usable_rank].sqrt()
    p = u[:, :usable_rank] * root.unsqueeze(0)
    b = root.unsqueeze(1) * vh[:usable_rank]
    # 若请求 rank 大于可用 rank，自动做零填充
    if usable_rank < rank:
        p = F.pad(p, (0, rank - usable_rank))
        b = F.pad(b, (0, 0, 0, rank - usable_rank))
    return p.to(linear.weight), b.to(linear.weight)


class ConditionedLowRankLinear(nn.Module):
    """以教师线性层的截断 SVD 初始化 ``P (g(z) ⊙ (B s))``。"""
    def __init__(
        self,
        teacher_target: nn.Module | Tensor,
        rank: int,
        z_dim: int,
        kappa: float,
        seed: int,
        gate_scale: float = 0.5,
        c_init_std: float = 0.1,
        encoded_input_mode: str = "input_dependent",
        fixed_encoded_std: float = 0.1,
    ) -> None:
        super().__init__()
        if rank < z_dim:
            raise ValueError("当前光子设计要求 rank 不小于 z_dim")
        p, b = _svd_pb(teacher_target, rank)
        # 截断 SVD 提供固定低秩基；训练仅学习无偏置的条件化门控 C。
        self.P = nn.Parameter(p, requires_grad=False)
        self.B = nn.Parameter(b, requires_grad=False)
        generator = torch.Generator(device="cpu").manual_seed(seed)
        if c_init_std < 0:
            raise ValueError("c_init_std 必须非负")
        c_initial = torch.randn(rank, z_dim, generator=generator) * c_init_std
        # 非零小随机 C 使第一步就有 dL/dz，从而把梯度传到量子 theta/phi；
        # 仍以很小方差保持初始门控接近 g=1。
        self.C = nn.Parameter(c_initial.to(dtype=p.dtype, device=p.device))
        # 先生成 rank×rank 正交矩阵，再取前 z_dim 行，确保 R 始终为 [z_dim, rank]，即使 rank 大于 z_dim 也保持行正交。
        orthogonal, _ = torch.linalg.qr(torch.randn(rank, rank, generator=generator))
        self.register_buffer("R", orthogonal[:z_dim].to(dtype=p.dtype, device=p.device))
        if encoded_input_mode not in {"input_dependent", "constant_zero", "constant_gaussian"}:
            raise ValueError("encoded_input_mode 必须是 input_dependent、constant_zero 或 constant_gaussian")
        if fixed_encoded_std < 0:
            raise ValueError("fixed_encoded_std 必须非负")
        self.encoded_input_mode = encoded_input_mode
        # 常量输入消融：每个投影有一个由其固定 seed 生成的向量，但它对所有
        # token 都相同。这样测试的是“是否依赖输入”，而不是每次前向重采样的噪声。
        fixed_encoded = torch.zeros(z_dim) if encoded_input_mode == "constant_zero" else torch.randn(z_dim, generator=generator) * fixed_encoded_std
        self.register_buffer("fixed_encoded", fixed_encoded.to(dtype=p.dtype, device=p.device))
        self.kappa = kappa
        self.gate_scale = gate_scale
        # PB-only 消融时固定 C/线路且直接取 g=1；跳过无意义的特征线路计算。
        self.conditioning_enabled = True

    def forward(self, x: Tensor, provider: PhotonicFeatureProvider, shots: int | None) -> Tensor:
        t = F.linear(x, self.B)
        if not self.conditioning_enabled:
            return F.linear(t, self.P)
        if self.encoded_input_mode == "input_dependent":
            encoded = self.kappa * torch.tanh(F.linear(t, self.R))
        else:
            encoded = self.fixed_encoded.expand(*t.shape[:-1], self.fixed_encoded.shape[0])
        z = provider.sample(encoded, shots, x.device)
        z = z.to(device=t.device, dtype=self.C.dtype)
        gate = 1 + self.gate_scale * torch.tanh(F.linear(z, self.C))
        return F.linear(gate * t, self.P)


class PhotonicLowRankMLP(nn.Module):
    """三个投影各自使用独立 provider、theta 与 EMA 的 SwiGLU 替换模块。"""
    def __init__(
        self,
        old_mlp: nn.Module,
        provider_factory: Callable[[], PhotonicFeatureProvider],
        rank: int,
        z_dim: int,
        kappa: float,
        layer_index: int,
        gate_scale: float = 0.5,
        c_init_std: float = 0.1,
        encoded_input_mode: str = "input_dependent",
        fixed_encoded_std: float = 0.1,
    ) -> None:
        super().__init__()
        for name in ("gate_proj", "up_proj", "down_proj"):
            if not isinstance(getattr(old_mlp, name, None), nn.Linear):
                raise TypeError(f"MLP lacks a linear {name}")
        # 三个 W 完全独立：不共享线路实例、光路参数或 EMA。
        self.gate = ConditionedLowRankLinear(old_mlp.gate_proj, rank, z_dim, kappa, layer_index * 11 + 1, gate_scale, c_init_std, encoded_input_mode, fixed_encoded_std)
        self.up = ConditionedLowRankLinear(old_mlp.up_proj, rank, z_dim, kappa, layer_index * 11 + 2, gate_scale, c_init_std, encoded_input_mode, fixed_encoded_std)
        self.down = ConditionedLowRankLinear(old_mlp.down_proj, 2 * rank, z_dim, kappa, layer_index * 11 + 3, gate_scale, c_init_std, encoded_input_mode, fixed_encoded_std)
        self.gate_provider = provider_factory()
        self.up_provider = provider_factory()
        self.down_provider = provider_factory()
        self.shots: int | None = None

    def forward(self, hidden_states: Tensor) -> Tensor:
        gate = self.gate(hidden_states, self.gate_provider, self.shots)
        up = self.up(hidden_states, self.up_provider, self.shots)
        return self.down(F.silu(gate) * up, self.down_provider, self.shots)

    def c_parameters(self) -> Iterator[nn.Parameter]:
        for projection in (self.gate, self.up, self.down):
            yield projection.C

    def photonic_parameters(self) -> Iterator[nn.Parameter]:
        for provider in (self.gate_provider, self.up_provider, self.down_provider):
            yield from provider.parameters()

    def pb_parameters(self) -> Iterator[nn.Parameter]:
        """返回冻结 SVD 低秩 P/B，供可选的低学习率微调。"""
        for projection in (self.gate, self.up, self.down):
            yield projection.P
            yield projection.B

    def disable_conditioning(self) -> None:
        """PB-only baseline: exactly use g=1 and do not execute the feature provider."""
        for projection in (self.gate, self.up, self.down):
            projection.conditioning_enabled = False

    @torch.no_grad()
    def photonic_parameter_values(self) -> dict[str, float]:
        """返回更新后的三套光路 theta/phi，供逐 step CSV 记录。"""
        values: dict[str, float] = {}
        for projection, provider in (
            ("gate", self.gate_provider),
            ("up", self.up_provider),
            ("down", self.down_provider),
        ):
            for parameter_name in ("theta", "phi"):
                parameter = getattr(provider, parameter_name).detach().float().cpu().reshape(-1)
                values.update({f"{projection}_{parameter_name}_{index:02d}": value for index, value in enumerate(parameter.tolist())})
        return values


class TruncatedSVDLinear(nn.Module):
    """无条件化的截断 SVD 基线：``W ≈ P B``。

    该模块没有门控、C 或特征提供器；它用于与同 rank 的条件化模型做公平的
    纯低秩比较。down 投影的 rank 由调用者显式传入（本项目为 ``2 * rank``）。
    """

    def __init__(self, teacher_target: nn.Module | Tensor, rank: int) -> None:
        super().__init__()
        p, b = _svd_pb(teacher_target, rank)
        self.register_buffer("P", p)
        self.register_buffer("B", b)

    def forward(self, x: Tensor) -> Tensor:
        return F.linear(F.linear(x, self.B), self.P)


class TruncatedSVDMLP(nn.Module):
    """Qwen SwiGLU MLP 的纯截断-SVD 基线。"""

    def __init__(self, old_mlp: nn.Module, rank: int) -> None:
        super().__init__()
        for name in ("gate_proj", "up_proj", "down_proj"):
            if not isinstance(getattr(old_mlp, name, None), nn.Linear):
                raise TypeError(f"MLP lacks a linear {name}")
        self.gate = TruncatedSVDLinear(old_mlp.gate_proj, rank)
        self.up = TruncatedSVDLinear(old_mlp.up_proj, rank)
        self.down = TruncatedSVDLinear(old_mlp.down_proj, 2 * rank)

    def forward(self, hidden_states: Tensor) -> Tensor:
        return self.down(F.silu(self.gate(hidden_states)) * self.up(hidden_states))


def freeze_non_mlp_modules(model: nn.Module) -> None:
    for parameter in model.parameters():
        parameter.requires_grad = False


def replace_final_mlps(
    model: nn.Module,
    provider_factory: Callable[[], PhotonicFeatureProvider],
    rank: int,
    z_dim: int,
    kappa: float,
    target_layers: tuple[int, ...],
    gate_scale: float = 0.5,
    c_init_std: float = 0.1,
    encoded_input_mode: str = "input_dependent",
    fixed_encoded_std: float = 0.1,
) -> list[PhotonicLowRankMLP]:
    """完整删除并替换 ``target_layers`` 指定的 MLP。"""
    layers = find_decoder_layers(model)
    if not target_layers:
        raise ValueError("target_layers must not be empty")
    if len(set(target_layers)) != len(target_layers):
        raise ValueError("target_layers must not contain duplicates")
    if any(index < 0 or index >= len(layers) for index in target_layers):
        raise ValueError(f"invalid target layers for {len(layers)} decoder layers: {target_layers}")
    freeze_non_mlp_modules(model)
    replacements: list[PhotonicLowRankMLP] = []
    for index in target_layers:
        # 每层维护独立的线路实例与 EMA；避免层间的特征统计相互污染。
        replacement = PhotonicLowRankMLP(
            layers[index].mlp, provider_factory, rank, z_dim, kappa, index, gate_scale, c_init_std,
            encoded_input_mode, fixed_encoded_std,
        )
        layers[index].mlp = replacement
        replacements.append(replacement)
    return replacements


def make_compressed_student(
    teacher: nn.Module,
    provider_factory: Callable[[], PhotonicFeatureProvider],
    rank: int,
    z_dim: int,
    kappa: float,
    target_layers: tuple[int, ...],
    gate_scale: float = 0.5,
    c_init_std: float = 0.1,
    encoded_input_mode: str = "input_dependent",
    fixed_encoded_std: float = 0.1,
) -> tuple[nn.Module, list[PhotonicLowRankMLP]]:
    """复制冻结教师，并在副本中删除对应 MLP。"""
    student = deepcopy(teacher)
    return student, replace_final_mlps(
        student, provider_factory, rank, z_dim, kappa, target_layers, gate_scale, c_init_std,
        encoded_input_mode, fixed_encoded_std,
    )


def make_truncated_svd_student(
    teacher: nn.Module, rank: int, target_layers: tuple[int, ...]
) -> nn.Module:
    """复制教师并以纯 ``W≈PB`` 模块替换指定层 MLP，且不训练任何新参数。"""
    student = deepcopy(teacher)
    freeze_non_mlp_modules(student)
    layers = find_decoder_layers(student)
    if not target_layers or len(set(target_layers)) != len(target_layers):
        raise ValueError("target_layers must be non-empty and contain no duplicates")
    if any(index < 0 or index >= len(layers) for index in target_layers):
        raise ValueError(f"invalid target layers for {len(layers)} decoder layers: {target_layers}")
    for index in target_layers:
        layers[index].mlp = TruncatedSVDMLP(layers[index].mlp, rank)
    return student
