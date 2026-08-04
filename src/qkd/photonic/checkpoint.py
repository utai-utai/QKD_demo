"""光子条件化 MLP 的专用 checkpoint 保存、读取与阶段一合并。"""

from __future__ import annotations

import json
from pathlib import Path

import torch

from .model import PhotonicLowRankMLP


def save_compressed_checkpoint( output_dir: str | Path, replacements: list[PhotonicLowRankMLP], rank: int, z_dim: int, kappa: float, target_layers: tuple[int, ...], teacher_name: str, provider_name: str) -> None:
    """仅保存替换 MLP，避免复制冻结的基础模型权重。"""
    output = Path(output_dir)
    output.mkdir(parents=True, exist_ok=True)
    metadata = {
        "format": "qkd-photonic-low-rank-v1",
        "teacher": teacher_name,
        "provider": provider_name,
        "spec": {
            "rank": rank,
            "z_dim": z_dim,
            "kappa": kappa,
            "target_layers": list(target_layers),
        },
    }
    (output / "photonic_config.json").write_text(json.dumps(metadata, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
    torch.save({str(index): module.state_dict() for index, module in zip(target_layers, replacements)}, output / "photonic_modules.pt",)


def read_compressed_checkpoint_config(checkpoint_dir: str | Path) -> dict[str, object]:
    """读取 checkpoint 的结构配置。"""
    checkpoint = Path(checkpoint_dir)
    metadata = json.loads((checkpoint / "photonic_config.json").read_text(encoding="utf-8"))
    if metadata.get("format") != "qkd-photonic-low-rank-v1":
        raise ValueError(f"不支持的光子 checkpoint 格式：{metadata.get('format')}")
    return metadata


def _checkpoint_values(metadata: dict[str, object]) -> tuple[int, int, float, tuple[int, ...]]:
    values = metadata["spec"]
    if not isinstance(values, dict):
        raise ValueError("checkpoint 缺少 spec 映射")
    return (
        int(values["rank"]), int(values["z_dim"]), float(values["kappa"]),
        tuple(int(index) for index in values["target_layers"]),
    )


def load_compressed_modules(checkpoint_dir: str | Path, replacements: list[PhotonicLowRankMLP], rank: int, z_dim: int, kappa: float, target_layers: tuple[int, ...]) -> None:
    """将同一结构的 checkpoint 恢复到替换模块。"""
    saved_values = _checkpoint_values(read_compressed_checkpoint_config(checkpoint_dir))
    current_values = (rank, z_dim, kappa, target_layers)
    if saved_values != current_values:
        raise ValueError(f"checkpoint 结构为 {saved_values}，当前结构为 {current_values}，两者不一致")
    weights = torch.load(Path(checkpoint_dir) / "photonic_modules.pt", map_location="cpu", weights_only=True)
    for index, module in zip(target_layers, replacements):
        module.load_state_dict(weights[str(index)])


def load_stage_one_checkpoints(checkpoint_dirs: list[str | Path], replacements: list[PhotonicLowRankMLP], rank: int, z_dim: int, kappa: float, target_layers: tuple[int, ...], provider_name: str) -> None:
    """按层合并多个阶段一 checkpoint，作为阶段二的联合初始化。"""
    modules_by_layer = dict(zip(target_layers, replacements))
    loaded_layers: set[int] = set()
    for checkpoint_dir in checkpoint_dirs:
        metadata = read_compressed_checkpoint_config(checkpoint_dir)
        saved_rank, saved_z_dim, saved_kappa, saved_layers = _checkpoint_values(metadata)
        if metadata.get("provider") != provider_name:
            raise ValueError(f"{checkpoint_dir} 的 provider 与当前 --provider 不一致")
        if (saved_rank, saved_z_dim, saved_kappa) != (rank, z_dim, kappa):
            raise ValueError(f"{checkpoint_dir} 的 rank/z_dim/kappa 与当前实验不一致")
        if not set(saved_layers).issubset(modules_by_layer):
            raise ValueError(f"{checkpoint_dir} 含有不在当前 --target-layers 中的层")
        weights = torch.load(Path(checkpoint_dir) / "photonic_modules.pt", map_location="cpu", weights_only=True)
        for index in saved_layers:
            if index in loaded_layers:
                raise ValueError(f"layer {index} 在多个阶段一 checkpoint 中重复出现")
            modules_by_layer[index].load_state_dict(weights[str(index)])
            loaded_layers.add(index)
    expected_layers = set(target_layers)
    if loaded_layers != expected_layers:
        raise ValueError(f"阶段一 checkpoint 覆盖层为 {sorted(loaded_layers)}，期望 {sorted(expected_layers)}")
