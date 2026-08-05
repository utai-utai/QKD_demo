"""训练运行目录、元数据、逐步日志与 checkpoint 的统一管理。"""

from __future__ import annotations

import csv
import json
from copy import deepcopy
from dataclasses import dataclass, field
from datetime import datetime, timezone
from glob import glob
from pathlib import Path
from time import monotonic
from typing import Any

import torch

from qkd.photonic.checkpoint import checkpoint_metadata, save_compressed_modules
from qkd.photonic.model import PhotonicLowRankMLP

STAGE1_LOG_FIELDS = [
    "step", "elapsed_seconds", "train_loss", "gate_loss", "up_loss", "down_loss", "output_loss",
    "gate_mae", "gate_nmse", "gate_cos", "gate_student_mean", "gate_teacher_mean",
    "up_mae", "up_nmse", "up_cos", "up_student_mean", "up_teacher_mean",
    "down_mae", "down_nmse", "down_cos", "down_student_mean", "down_teacher_mean",
    "y_mae", "y_nmse", "y_cos", "y_student_mean", "y_teacher_mean",
    "validation_loss", "validation_y_nmse", "is_best", "spsa_applied", "early_stopped",
]
STAGE2_LOG_FIELDS = ["step", "elapsed_seconds", "loss", "ce", "kd", "is_best", "spsa_applied", "early_stopped"]


def resolve_output_dir(template: str | Path) -> Path:
    """展开输出目录中的 ``{timestamp}``，为每次实验创建可区分的名称。"""
    timestamp = datetime.now().astimezone().strftime("%Y%m%d-%H%M%S")
    return Path(str(template).replace("{timestamp}", timestamp))


def resolve_checkpoint_dir(reference: str | Path) -> Path:
    """解析 checkpoint 路径；``{latest}`` 会选择按时间命名的最新一次运行。"""
    value = str(reference)
    if "{latest}" not in value:
        return Path(value)
    directories = [Path(path) for path in sorted(glob(value.replace("{latest}", "*"))) if Path(path).is_dir()]
    if not directories:
        raise FileNotFoundError(f"未找到匹配 {reference} 的 Stage 1 checkpoint")
    return directories[-1]


@dataclass
class TrainingArtifacts:
    """单次训练运行的全部落盘产物；训练代码只需调用其公开方法。"""

    output: Path
    config: dict[str, Any]
    stage: str
    device: torch.device
    fieldnames: list[str]
    started: float = field(default_factory=monotonic)
    started_at_utc: str = field(default_factory=lambda: datetime.now(timezone.utc).isoformat())
    run: dict[str, Any] = field(default_factory=dict)
    checkpoint: dict[str, Any] | None = None
    _file: Any = field(init=False, repr=False)
    _writer: csv.DictWriter = field(init=False, repr=False)

    @classmethod
    def create(
        cls, config: dict[str, Any], stage: str, device: torch.device, fieldnames: list[str], teacher: str,
    ) -> "TrainingArtifacts":
        experiment = config.get("experiment")
        if not isinstance(experiment, dict):
            raise ValueError("配置缺少 experiment 映射")
        output = resolve_output_dir(experiment["output_dir"])
        experiment["output_dir"] = str(output)
        output.mkdir(parents=True, exist_ok=True)
        artifacts = cls(output, config, stage, device, fieldnames)
        artifacts.run = {"stage": stage, "device": str(device), "started_at_utc": artifacts.started_at_utc, "status": "running", "teacher": teacher}
        artifacts._file = (output / "training_log.csv").open("w", newline="", encoding="utf-8")
        artifacts._writer = csv.DictWriter(artifacts._file, fieldnames=fieldnames, extrasaction="raise")
        artifacts._writer.writeheader()
        artifacts._file.flush()
        artifacts._write_metadata()
        return artifacts

    @property
    def elapsed_seconds(self) -> float:
        return monotonic() - self.started

    def log_step(self, values: dict[str, Any]) -> None:
        unknown = set(values) - set(self.fieldnames)
        if unknown:
            raise ValueError(f"CSV 包含未声明字段：{sorted(unknown)}")
        self._writer.writerow(values)
        self._file.flush()

    def save_checkpoint(
        self, replacements: list[PhotonicLowRankMLP], rank: int, z_dim: int, kappa: float,
        target_layers: tuple[int, ...], teacher_name: str, provider_name: str,
    ) -> None:
        self.checkpoint = checkpoint_metadata(rank, z_dim, kappa, target_layers, teacher_name, provider_name)
        save_compressed_modules(self.output, replacements, target_layers)
        self._write_metadata()

    def save_best_probe(self, payload: dict[str, Any]) -> None:
        """保存最佳权重在固定验证 probe 上的教师/学生输出。"""
        torch.save(payload, self.output / "best_probe.pt")

    def finish(self, **summary: Any) -> None:
        self.run.update(summary)
        self.run["duration_seconds"] = self.elapsed_seconds
        self._file.close()
        self._write_metadata()

    def _write_metadata(self) -> None:
        payload: dict[str, Any] = {
            "format": "qkd-photonic-low-rank-v1",
            "config": deepcopy(self.config),
            "run": self.run,
        }
        if self.checkpoint is not None:
            payload["checkpoint"] = self.checkpoint
        (self.output / "run.json").write_text(json.dumps(payload, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
