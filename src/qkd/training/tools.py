"""阶段一与阶段二共用的配置、设备和数据加载工具。"""

from __future__ import annotations

from copy import deepcopy
from datetime import datetime, timezone
from functools import partial
from pathlib import Path
import sys
from typing import Any

import torch
import yaml
from torch.utils.data import DataLoader

from qkd.data import TokenizedChatDataset, collate_tokenized
from qkd.photonic import DeepQuantumCVFeatureProvider, MockPhotonicFeatureProvider


def load_config(path: str | Path) -> dict[str, Any]:
    """读取 YAML 配置，并在缺失或非映射结构时尽早报错。"""
    config_path = Path(path)
    config = yaml.safe_load(config_path.read_text(encoding="utf-8"))
    if not isinstance(config, dict):
        raise ValueError(f"{config_path} 必须是 YAML 映射")
    return config


def apply_overrides(config: dict[str, Any], assignments: list[str]) -> dict[str, Any]:
    """将 ``段.键=YAML值`` 形式的命令行覆盖写入配置。"""
    for assignment in assignments:
        if "=" not in assignment:
            raise ValueError(f"--set 必须使用 键路径=值 形式：{assignment}")
        path, raw_value = assignment.split("=", 1)
        keys = path.split(".")
        if not path or any(not key for key in keys):
            raise ValueError(f"无效的配置路径：{path}")
        target: dict[str, Any] = config
        for key in keys[:-1]:
            value = target.get(key)
            if not isinstance(value, dict):
                raise ValueError(f"--set 路径不存在或不是配置段：{path}")
            target = value
        if keys[-1] not in target:
            raise ValueError(f"--set 不允许新增未知配置项：{path}")
        target[keys[-1]] = yaml.safe_load(raw_value)
    return config


def section(config: dict[str, Any], name: str) -> dict[str, Any]:
    """取得必需的配置段。"""
    value = config.get(name)
    if not isinstance(value, dict):
        raise ValueError(f"配置缺少映射段：{name}")
    return value


def make_loader(path: str, tokenizer: Any, batch_size: int, shuffle: bool) -> DataLoader:
    return DataLoader(
        TokenizedChatDataset(path), batch_size=batch_size, shuffle=shuffle,
        collate_fn=partial(collate_tokenized, pad_token_id=tokenizer.pad_token_id),
    )


def next_batch(iterator: Any, loader: DataLoader) -> tuple[dict[str, torch.Tensor], Any]:
    try:
        return next(iterator), iterator
    except StopIteration:
        iterator = iter(loader)
        return next(iterator), iterator


def training_device() -> torch.device:
    """macOS 固定 CPU；其他系统优先 CUDA。"""
    if sys.platform == "darwin":
        return torch.device("cpu")
    return torch.device("cuda" if torch.cuda.is_available() else "cpu")


def provider_factory(name: str, z_dim: int, ema_decay: float | None):
    """为每个 W 返回独立 provider 的构造器。"""
    if name == "mock":
        return lambda: MockPhotonicFeatureProvider(z_dim=z_dim, ema_decay=ema_decay)
    if name == "deepquantum":
        return lambda: DeepQuantumCVFeatureProvider(z_dim=z_dim, ema_decay=ema_decay)
    raise ValueError(f"不支持的光子 provider：{name}")


def write_run_config(output_dir: str | Path, config: dict[str, Any], stage: str, device: torch.device, extra: dict[str, Any] | None = None) -> Path:
    """保存可复现实验的完整配置快照与运行元数据。"""
    snapshot = deepcopy(config)
    snapshot["run_metadata"] = {
        "stage": stage,
        "device": str(device),
        "started_at_utc": datetime.now(timezone.utc).isoformat(),
        "checkpoint_format": "qkd-photonic-low-rank-v1",
        **(extra or {}),
    }
    output = Path(output_dir)
    output.mkdir(parents=True, exist_ok=True)
    path = output / "run_config.yaml"
    path.write_text(yaml.safe_dump(snapshot, allow_unicode=True, sort_keys=False), encoding="utf-8")
    return path
