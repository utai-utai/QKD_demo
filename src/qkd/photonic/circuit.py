"""DeepQuantum 后端使用的八模环形连续变量光子电路。"""

from __future__ import annotations

import torch
from torch import Tensor


def eight_mode_ring_photon_numbers(encoded: Tensor, theta: Tensor) -> Tensor:
    """运行八模高斯环形光路，并返回每个模式的平均光子数。
    前八个 ``encoded`` 分量控制模式位移，``theta[:8]`` 为位移偏置，``theta[8:16]`` 为相移。
    相邻模式经 50:50 分束器耦合，末尾与首个模式相连形成环。
    DeepQuantum 采用高斯后端，返回解析的光子数期望而非 shot 样本。
    """
    if encoded.shape[-1] != 16:
        raise ValueError(f"期望 encoded[..., 16]，实际为 {tuple(encoded.shape)}")
    theta_values = theta.detach().reshape(-1).float().cpu()
    if theta_values.numel() != 16:
        raise ValueError(f"期望 16 个 theta 参数，实际为 {theta_values.numel()}")

    import deepquantum as dq

    values = encoded.detach().float().reshape(-1, 16).cpu()
    photon_numbers = []
    for value in values:
        circuit = dq.QumodeCircuit(nmode=8, init_state="vac", backend="gaussian")
        for mode in range(8):
            circuit.d(wires=mode, r=float(torch.tanh(value[mode] + theta_values[mode])))
            circuit.ps(wires=mode, inputs=float(theta_values[8 + mode]))
        for mode in range(8):
            circuit.bs(wires=[mode, (mode + 1) % 8], inputs=[torch.pi / 4, 0.0])
        circuit()
        mean, _ = circuit.photon_number_mean_var(wires=list(range(8)))
        photon_numbers.append(mean.reshape(8).float())
    return torch.stack(photon_numbers).reshape(*encoded.shape[:-1], 8).to(encoded)
