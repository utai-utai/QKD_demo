"""16-mode single-photon path encoding and Clements interferometer."""

from __future__ import annotations

import torch
from torch import Tensor


def clements_layout(n_modes: int) -> tuple[tuple[tuple[int, int], ...], int]:
    """Return the 16-layer alternating Clements mesh connectivity.

    For 16 modes this is 8 even layers with 8 MZIs and 8 odd layers with 7
    MZIs: 120 MZIs total.  A universal U(16) mesh also has 16 output phases.
    """
    if n_modes < 2 or n_modes % 2:
        raise ValueError("photonic.modes must be an even integer >= 2")
    even_pairs = tuple((mode, mode + 1) for mode in range(0, n_modes, 2))
    odd_pairs = tuple((mode, mode + 1) for mode in range(1, n_modes - 1, 2))
    pairs = tuple(pair for layer in range(n_modes) for pair in (even_pairs if layer % 2 == 0 else odd_pairs))
    return pairs, len(pairs)


def clements_parameter_counts(n_modes: int) -> tuple[int, int, int]:
    """Return number of MZIs, MZI parameters, and output PS parameters."""
    _, n_mzi = clements_layout(n_modes)
    return n_mzi, 2 * n_mzi, n_modes


def uniform_single_photon_state(n_modes: int):
    """Return the uniform path state for DeepQuantum drawing/inspection only."""
    import deepquantum as dq

    amplitude = 1 / n_modes**0.5
    return dq.FockState(
        state=[(amplitude, [int(mode == index) for mode in range(n_modes)]) for index in range(n_modes)],
        basis=False,
    )


def clements_single_photon_probabilities(
    encoded: Tensor, theta: Tensor, phi: Tensor, output_phase: Tensor, n_modes: int,
    intermediate_phase: Tensor | None = None,
) -> Tensor:
    """Exact Fock evolution in the single-photon path subspace.

    The full Fock tensor has ``2**n_modes`` entries even though this circuit
    always contains exactly one photon.  The physically reachable Fock sector
    is only ``{|1_0>, ..., |1_(N-1)>}``, so tracking its N complex amplitudes
    is mathematically exact and makes 16-mode training practical.  The MZI
    convention is identical to DeepQuantum's ``mzi(..., phi_first=True)``.
    """
    pairs, n_mzi = clements_layout(n_modes)
    if encoded.shape[-1] != n_modes:
        raise ValueError(f"Clements phase encoding expects encoded[..., {n_modes}], got {tuple(encoded.shape)}")
    n_meshes = 1 if intermediate_phase is None else intermediate_phase.shape[0] + 1
    if theta.numel() != n_meshes * n_mzi or phi.numel() != n_meshes * n_mzi or output_phase.numel() != n_modes:
        raise ValueError(
            f"{n_meshes} × {n_modes}-mode Clements meshes require theta/phi={n_meshes * n_mzi} each and output_phase={n_modes}; got "
            f"{theta.numel()}, {phi.numel()}, {output_phase.numel()}"
        )
    if intermediate_phase is not None and intermediate_phase.shape != (n_meshes - 1, n_modes):
        raise ValueError(f"intermediate_phase must be [{n_meshes - 1}, {n_modes}], got {tuple(intermediate_phase.shape)}")

    values = encoded.reshape(-1, n_modes).float()
    # |psi_in> = 1/sqrt(N) sum_i exp(i*pi*tanh(e_i)) |1_i>.
    amplitudes = torch.exp(1j * torch.tanh(values).to(torch.complex64) * torch.pi) / n_modes**0.5
    for mesh_index in range(n_meshes):
        offset = mesh_index * n_mzi
        for pair_index, (wire_a, wire_b) in enumerate(pairs):
            # DeepQuantum MZI(phi_first=True):
            # i exp(i theta/2) [[exp(i phi) sin(theta/2), cos(theta/2)],
            #                    [exp(i phi) cos(theta/2), -sin(theta/2)]].
            angle, phase = theta[offset + pair_index].float(), phi[offset + pair_index].float()
            scale = 1j * torch.exp(0.5j * angle)
            sine, cosine = torch.sin(angle / 2), torch.cos(angle / 2)
            phase_factor = torch.exp(1j * phase)
            values_by_mode = list(amplitudes.unbind(dim=-1))
            source_a, source_b = values_by_mode[wire_a], values_by_mode[wire_b]
            values_by_mode[wire_a] = scale * (phase_factor * sine * source_a + cosine * source_b)
            values_by_mode[wire_b] = scale * (phase_factor * cosine * source_a - sine * source_b)
            amplitudes = torch.stack(values_by_mode, dim=-1)
        # The first mesh's output PS is physically between the two meshes and
        # therefore affects interference in mesh 2.  Only the final PS cancels
        # under photon-number detection.
        if mesh_index < n_meshes - 1:
            amplitudes = amplitudes * torch.exp(1j * intermediate_phase[mesh_index].to(torch.complex64))
    # End phases are retained in the hardware drawing but cancel from |b_i|^2.
    del output_phase
    probabilities = amplitudes.abs().square()
    return probabilities.reshape(*encoded.shape[:-1], n_modes)
