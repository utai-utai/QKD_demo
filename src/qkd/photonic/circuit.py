"""DeepQuantum 后端使用的可配置分层线性光学电路。"""

from __future__ import annotations

import torch
from torch import Tensor


def circuit_layout(n_modes: int, n_layers: int) -> tuple[tuple[tuple[int, int], ...], int, int]:
    """推导交错相邻 BS→全 mode PS 电路的连线与参数数目。"""
    if n_modes < 2 or n_modes % 2:
        raise ValueError("photonic.modes 必须是大于等于 2 的偶数")
    if n_layers < 1:
        raise ValueError("photonic.layers 必须大于等于 1")
    even_pairs = tuple((mode, mode + 1) for mode in range(0, n_modes, 2))
    odd_pairs = tuple((mode, mode + 1) for mode in range(1, n_modes - 1, 2))
    pairs = even_pairs + odd_pairs
    return pairs, n_layers * len(pairs), n_layers * n_modes


def layered_photon_numbers(
    encoded: Tensor, theta: Tensor, phi: Tensor, n_modes: int, n_layers: int
) -> Tensor:
    """返回可微分层 BS→PS 高斯光路的各 mode 平均光子数。"""
    pairs, n_theta, n_phi = circuit_layout(n_modes, n_layers)
    if encoded.shape[-1] != 2 * n_modes:
        raise ValueError(f"期望 encoded[..., {2 * n_modes}]，实际为 {tuple(encoded.shape)}")
    if theta.numel() != n_theta or phi.numel() != n_phi:
        raise ValueError(
            f"{n_layers} 层 {n_modes} mode 线路需要 theta={n_theta}、phi={n_phi}，"
            f"实际为 {theta.numel()}、{phi.numel()}"
        )

    import deepquantum as dq

    values = encoded.reshape(-1, 2 * n_modes).float()
    circuit = dq.QumodeCircuit(nmode=n_modes, init_state="vac", backend="gaussian")
    theta_index = 0
    for layer in range(n_layers):
        for wire_a, wire_b in pairs:
            # 每个 BS 仅训练混合角 theta；内部相位固定为 0。
            circuit.bs(wires=[wire_a, wire_b], inputs=[theta[theta_index], 0.0])
            theta_index += 1
        for mode in range(n_modes):
            circuit.ps(wires=mode, inputs=phi[layer * n_modes + mode])
    circuit.to(values.device)

    batch_size = values.shape[0]
    covariance = torch.eye(2 * n_modes, dtype=values.dtype, device=values.device).expand(batch_size, -1, -1)
    mean = values.new_zeros(batch_size, 2 * n_modes, 1)
    # xxpp 排列：前 n_modes 项是 x 象限。tanh 与旧版位移编码保持同一数值范围。
    mean[:, :n_modes, 0] = torch.tanh(values[:, :n_modes])
    circuit(state=[covariance, mean])
    photon_numbers, _ = circuit.photon_number_mean_var(wires=list(range(n_modes)))
    return photon_numbers.reshape(*encoded.shape[:-1], n_modes)
