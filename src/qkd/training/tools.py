"""阶段一与阶段二共用的配置、设备和数据加载工具。"""

from __future__ import annotations

import sys
from contextlib import AbstractContextManager
from functools import partial
from pathlib import Path
from typing import Any

import torch
import yaml
from tqdm import tqdm
from torch.utils.data import DataLoader

from qkd.data import TokenizedChatDataset, collate_tokenized
from qkd.photonic import DeepQuantumCVFeatureProvider, MockPhotonicFeatureProvider
from qkd.photonic.model import PhotonicLowRankMLP
from qkd.training.stage1_loss import local_diagnostics, stage_one_loss
from qkd.training.stage2_loss import stage_two_loss


class StageOneReference(AbstractContextManager["StageOneReference"]):
    """管理教师 MLP Hook，并提供阶段一训练与验证计算。"""

    def __init__(
        self,
        teacher: torch.nn.Module,
        teacher_mlp: torch.nn.Module,
        student_mlp: PhotonicLowRankMLP,
        auxiliary_weight: float = 0.0,
        loss_scale: float = 1.0,
    ) -> None:
        self.teacher = teacher
        self.teacher_mlp = teacher_mlp
        self.student_mlp = student_mlp
        self.auxiliary_weight = auxiliary_weight
        self.loss_scale = loss_scale
        self.values: tuple[torch.Tensor, torch.Tensor] | None = None
        self._hook: Any = None

    def __enter__(self) -> "StageOneReference":  # noqa: PYI034, UP037
        self._hook = self.teacher_mlp.register_forward_hook(self._capture)
        return self

    def __exit__(self, *args: object) -> None:
        if self._hook is not None:
            self._hook.remove()

    def _capture(
        self,
        module: torch.nn.Module,
        inputs: tuple[torch.Tensor, ...],
        output: torch.Tensor,
    ) -> None:
        del module
        self.values = (inputs[0].detach(), output.detach())

    def terms(self, batch: dict[str, torch.Tensor]) -> dict[str, torch.Tensor]:
        """运行教师并计算当前批次的局部重建项。"""
        self.values = None
        with torch.no_grad():
            self.teacher(input_ids=batch["input_ids"], attention_mask=batch["attention_mask"])
            if self.values is None:
                raise RuntimeError("教师 MLP hook 未捕获到输出")
            x, teacher_output = self.values
            teacher_gate = self.teacher_mlp.gate_proj(x)
            teacher_up = self.teacher_mlp.up_proj(x)
        student_gate = self.student_mlp.gate(
            x,
            self.student_mlp.gate_provider,
            self.student_mlp.shots,
        )
        student_up = self.student_mlp.up(
            x,
            self.student_mlp.up_provider,
            self.student_mlp.shots,
        )
        student_value = torch.nn.functional.silu(student_gate) * student_up
        student_output = self.student_mlp.down(
            student_value,
            self.student_mlp.down_provider,
            self.student_mlp.shots,
        )
        teacher_value = torch.nn.functional.silu(teacher_gate) * teacher_up
        projected_teacher_value = self.student_mlp.down(
            teacher_value,
            self.student_mlp.down_provider,
            self.student_mlp.shots,
        )
        return stage_one_loss(
            student_gate,
            student_up,
            student_output,
            projected_teacher_value,
            teacher_gate,
            teacher_up,
            teacher_output,
            batch["attention_mask"],
            self.auxiliary_weight,
            self.loss_scale,
        )

    def diagnostics(
        self, terms: dict[str, torch.Tensor], attention_mask: torch.Tensor
    ) -> dict[str, float]:
        """取得最近一次教师前向对应的终端诊断值。"""
        if self.values is None:
            raise RuntimeError("请先调用 terms")
        x, teacher_output = self.values
        with torch.no_grad():
            teacher_gate = self.teacher_mlp.gate_proj(x)
            teacher_up = self.teacher_mlp.up_proj(x)
        return local_diagnostics(terms, teacher_gate, teacher_up, teacher_output, attention_mask)

    @torch.no_grad()
    def validate(
        self, loader: DataLoader, student: torch.nn.Module, device: torch.device
    ) -> tuple[float, dict[str, float]]:
        """在完整验证集上计算 token 加权局部重建损失与诊断。"""
        was_training = student.training
        student.eval()
        loss_sum, token_count, totals, term_sums = 0.0, 0.0, {}, {}
        try:
            progress = tqdm(loader, desc="Stage 1 validation", unit="batch", leave=False)
            for batch in progress:
                batch = {key: value.to(device) for key, value in batch.items()}
                terms = self.terms(batch)
                metrics = self.diagnostics(terms, batch["attention_mask"])
                valid = metrics.pop("valid_tokens")
                loss_sum += terms["loss"].float().item() * valid
                token_count += valid
                for name in ("output_loss", "gate_loss", "up_loss", "down_loss"):
                    term_sums[name] = term_sums.get(name, 0.0) + terms[name].float().item() * valid
                for name, value in metrics.items():
                    totals[name] = totals.get(name, 0.0) + value * valid
                progress.set_postfix(loss=f"{loss_sum / token_count:.4f}")
        finally:
            student.train(was_training)
        metrics = {name: value / token_count for name, value in totals.items()}
        metrics.update({name: value / token_count for name, value in term_sums.items()})
        return loss_sum / token_count, metrics

    @torch.no_grad()
    def spsa_objective(self, batch: dict[str, torch.Tensor], student: torch.nn.Module) -> torch.Tensor:
        """以评估模式计算单个验证批次的 SPSA 目标。"""
        was_training = student.training
        student.eval()
        try:
            return self.terms(batch)["loss"]
        finally:
            student.train(was_training)


@torch.no_grad()
def stage_two_validation_objective(
    student: torch.nn.Module,
    teacher: torch.nn.Module,
    loader: DataLoader,
    device: torch.device,
    temperature: float,
    top_k: int,
) -> torch.Tensor:
    """计算完整验证集的 Stage 2 蒸馏目标，供 SPSA 调用。"""
    was_training = student.training
    student.eval()
    try:
        values = []
        for batch in loader:
            batch = {key: value.to(device) for key, value in batch.items()}
            teacher_logits = teacher(
                input_ids=batch["input_ids"], attention_mask=batch["attention_mask"]
            ).logits
            student_logits = student(
                input_ids=batch["input_ids"], attention_mask=batch["attention_mask"]
            ).logits
            values.append(
                stage_two_loss(student_logits, teacher_logits, batch["labels"], temperature, top_k)[
                    "loss"
                ]
            )
        return torch.stack(values).mean()
    finally:
        student.train(was_training)


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


def provider_factory(name: str, z_dim: int, ema_decay: float | None, n_modes: int, n_layers: int):
    """为每个 W 返回独立 provider 的构造器。"""
    if name == "mock":
        return lambda: MockPhotonicFeatureProvider(
            z_dim=z_dim, ema_decay=ema_decay, n_modes=n_modes, n_layers=n_layers
        )
    if name == "deepquantum":
        return lambda: DeepQuantumCVFeatureProvider(
            z_dim=z_dim, ema_decay=ema_decay, n_modes=n_modes, n_layers=n_layers
        )
    raise ValueError(f"不支持的光子 provider：{name}")
