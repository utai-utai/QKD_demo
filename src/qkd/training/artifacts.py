"""训练运行目录、元数据、逐步日志与 checkpoint 的统一管理。"""

from __future__ import annotations

import csv
import json
from contextlib import ExitStack, contextmanager
from copy import deepcopy
from dataclasses import dataclass, field
from datetime import datetime, timezone
from glob import glob
from pathlib import Path
from time import monotonic
from typing import Any, Iterator

import torch
from torch import nn

from qkd.photonic.checkpoint import checkpoint_metadata, save_compressed_modules
from qkd.photonic.circuit import clements_layout
from qkd.photonic.model import PhotonicLowRankMLP, find_decoder_layers

STAGE1_LOG_FIELDS = [
    "step",
    "elapsed_seconds",
    "train_loss",
    "gate_loss",
    "up_loss",
    "down_loss",
    "output_loss",
    "auxiliary_loss",
    "unscaled_loss",
    "gate_c_grad_norm",
    "gate_p_grad_norm",
    "gate_b_grad_norm",
    "gate_theta_grad_norm",
    "gate_phi_grad_norm",
    "gate_feature_grad_norm",
    "up_c_grad_norm",
    "up_p_grad_norm",
    "up_b_grad_norm",
    "up_theta_grad_norm",
    "up_phi_grad_norm",
    "up_feature_grad_norm",
    "down_c_grad_norm",
    "down_p_grad_norm",
    "down_b_grad_norm",
    "down_theta_grad_norm",
    "down_phi_grad_norm",
    "down_feature_grad_norm",
    "gate_mae",
    "gate_nmse",
    "gate_cos",
    "gate_student_mean",
    "gate_teacher_mean",
    "up_mae",
    "up_nmse",
    "up_cos",
    "up_student_mean",
    "up_teacher_mean",
    "down_mae",
    "down_nmse",
    "down_cos",
    "down_student_mean",
    "down_teacher_mean",
    "y_mae",
    "y_nmse",
    "y_cos",
    "y_student_mean",
    "y_teacher_mean",
    "validation_loss",
    "validation_output_loss",
    "validation_gate_loss",
    "validation_up_loss",
    "validation_down_loss",
    "validation_y_nmse",
    "is_best",
    "spsa_applied",
    "early_stopped",
]


def stage1_photonic_log_fields(n_modes: int, n_layers: int) -> list[str]:
    """为三个独立光路生成逐参数 CSV 列名。"""
    if n_layers != n_modes:
        raise ValueError("Clements 网格要求 photonic.layers 等于 photonic.modes")
    _, n_mzi = clements_layout(n_modes)
    fields = list(STAGE1_LOG_FIELDS)
    for projection in ("gate", "up", "down"):
        fields.extend(f"{projection}_theta_{index:03d}" for index in range(n_mzi))
        fields.extend(f"{projection}_phi_{index:03d}" for index in range(n_mzi))
    return fields
STAGE2_LOG_FIELDS = [
    "step",
    "elapsed_seconds",
    "loss",
    "ce",
    "kd",
    "c_grad_norm",
    "pb_grad_norm",
    "feature_grad_norm",
    "theta_grad_norm",
    "phi_grad_norm",
    "validation_loss",
    "validation_ce",
    "validation_kd",
    "is_best",
    "spsa_applied",
    "early_stopped",
]


def best_probe_payload(
    stage: str,
    step: int,
    metric: str,
    value: float | None,
    target_layers: tuple[int, ...],
    batch: dict[str, torch.Tensor],
    teacher_y: dict[int, torch.Tensor],
    student_y: dict[int, torch.Tensor],
) -> dict[str, object]:
    """构造可持久化的 probe 数据，并将大型输出压缩为 float16 CPU 张量。"""
    return {
        "format": "qkd-best-probe-v1",
        "stage": stage,
        "step": step,
        "selection": {"metric": metric, "value": value},
        "target_layers": list(target_layers),
        "input_ids": batch["input_ids"].detach().cpu(),
        "attention_mask": batch["attention_mask"].detach().cpu(),
        "labels": batch["labels"].detach().cpu(),
        "layers": {
            str(index): {
                "teacher_y": teacher_y[index].detach().to("cpu", torch.float16),
                "student_y": student_y[index].detach().to("cpu", torch.float16),
            }
            for index in target_layers
        },
    }


@contextmanager
def capture_mlp_outputs(
    teacher: nn.Module, student: nn.Module, target_layers: tuple[int, ...]
) -> Iterator[tuple[dict[int, torch.Tensor], dict[int, torch.Tensor]]]:
    """临时捕获教师和学生指定 MLP 层的输出。"""
    teacher_y: dict[int, torch.Tensor] = {}
    student_y: dict[int, torch.Tensor] = {}

    def hook(values: dict[int, torch.Tensor], index: int):
        def save(module, inputs, output_value) -> None:
            values[index] = output_value.detach()

        return save

    teacher_layers, student_layers = find_decoder_layers(teacher), find_decoder_layers(student)
    with ExitStack() as stack:
        for index in target_layers:
            stack.callback(
                teacher_layers[index].mlp.register_forward_hook(hook(teacher_y, index)).remove
            )
            stack.callback(
                student_layers[index].mlp.register_forward_hook(hook(student_y, index)).remove
            )
        yield teacher_y, student_y


def resolve_output_dir(template: str | Path) -> Path:
    """展开输出目录中的 ``{timestamp}``，为每次实验创建可区分的名称。"""
    timestamp = datetime.now().astimezone().strftime("%Y%m%d-%H%M%S")
    return Path(str(template).replace("{timestamp}", timestamp))


def resolve_checkpoint_dir(reference: str | Path) -> Path:
    """解析 checkpoint 路径；``{latest}`` 会选择按时间命名的最新一次运行。"""
    value = str(reference)
    if "{latest}" not in value:
        return Path(value)
    directories = [
        Path(path) for path in sorted(glob(value.replace("{latest}", "*"))) if Path(path).is_dir()
    ]
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
        cls,
        config: dict[str, Any],
        stage: str,
        device: torch.device,
        fieldnames: list[str],
    ) -> "TrainingArtifacts":
        experiment = config.get("experiment")
        if not isinstance(experiment, dict):
            raise ValueError("配置缺少 experiment 映射")
        output = resolve_output_dir(experiment["output_dir"])
        experiment["output_dir"] = str(output)
        output.mkdir(parents=True, exist_ok=True)
        artifacts = cls(output, config, stage, device, fieldnames)
        artifacts.run = {
            "stage": stage,
            "device": str(device),
            "started_at_utc": artifacts.started_at_utc,
            "status": "running",
        }
        artifacts._file = (output / "training_log.csv").open("w", newline="", encoding="utf-8")
        artifacts._writer = csv.DictWriter(
            artifacts._file, fieldnames=fieldnames, extrasaction="raise"
        )
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
        self,
        replacements: list[PhotonicLowRankMLP],
        rank: int,
        z_dim: int,
        kappa: float,
        target_layers: tuple[int, ...],
        teacher_name: str,
        provider_name: str,
    ) -> None:
        self.checkpoint = checkpoint_metadata(
            rank, z_dim, kappa, target_layers, teacher_name, provider_name
        )
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
            "config": self._metadata_config(),
            "run": self.run,
        }
        if self.checkpoint is not None:
            payload["checkpoint"] = self.checkpoint
        (self.output / "run.json").write_text(
            json.dumps(payload, ensure_ascii=False, indent=2) + "\n", encoding="utf-8"
        )

    def _metadata_config(self) -> dict[str, Any]:
        """返回 run.json 的精简配置快照。

        保存 checkpoint 后，模型结构与 provider 由 ``checkpoint`` 唯一描述；避免在
        配置快照中重复保存。未保存 checkpoint 的运行保留完整配置，便于排查中断。
        """
        config = deepcopy(self.config)
        if self.checkpoint is None:
            return config

        experiment = config.get("experiment")
        if isinstance(experiment, dict):
            experiment.pop("output_dir", None)

        config.pop("model", None)

        # rank/z_dim/kappa/target_layers 已由 checkpoint 唯一描述；但门控与初始化
        # 会影响训练动力学，必须保留在 run.json 以复现实验。
        compression = config.get("compression")
        if isinstance(compression, dict):
            config["compression"] = {
                key: compression[key]
                for key in ("gate_scale", "c_init_std", "train_pb")
                if key in compression
            }

        photonic = config.get("photonic")
        if isinstance(photonic, dict):
            photonic.pop("provider", None)
            if not photonic:
                config.pop("photonic")
        return config
