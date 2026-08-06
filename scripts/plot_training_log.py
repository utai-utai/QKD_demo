"""从 outputs/runs 中选择训练日志，并绘制指定的指标曲线。"""

from __future__ import annotations

import argparse
import csv
import math
import os
import re
import tempfile
from collections.abc import Iterable
from pathlib import Path

_PROJECT_ROOT = Path(__file__).resolve().parents[1]
_RUNS_ROOT = _PROJECT_ROOT / "outputs" / "runs"
_ANALYSIS_ROOT = _PROJECT_ROOT / "outputs" / "analysis"
_CACHE_DIR = Path(tempfile.gettempdir()) / "qkd-matplotlib"
_CACHE_DIR.mkdir(parents=True, exist_ok=True)
os.environ.setdefault("MPLCONFIGDIR", str(_CACHE_DIR))
os.environ.setdefault("XDG_CACHE_HOME", str(_CACHE_DIR))

import matplotlib
from matplotlib.lines import Line2D

matplotlib.use("Agg")
import matplotlib.pyplot as plt

_LINE_STYLES = ("-", "--", ":", "-.")


def plot_training_logs(
    run_selectors: Iterable[str], metrics: list[str], log_y: bool = False
) -> Path:
    """绘制同一 Stage 的指定运行与指标，并返回保存至 analysis 的图片路径。"""
    selections = _expand_selectors(run_selectors)
    if not selections:
        raise ValueError("至少指定一个 --runs 运行选择器")
    if not metrics:
        raise ValueError("至少指定一个 --metrics 指标")

    runs = [(_normalize_selector(selector), _resolve_run(selector)) for selector in selections]
    stages = {run.parent.name for _, run in runs}
    if len(stages) != 1:
        raise ValueError("不能在同一张图中比较 Stage 1 和 Stage 2 的训练日志")
    stage = stages.pop()

    logs = [(selector, run, *_read_log(run / "training_log.csv")) for selector, run in runs]
    for _, run, _, fields in logs:
        missing = [metric for metric in metrics if metric not in fields]
        if missing:
            raise ValueError(f"{run.name} 不包含指标 {missing}；可用列：{', '.join(fields)}")

    figure, axis = plt.subplots(figsize=(10, 5.5), constrained_layout=True)
    if len(logs) == 1:
        _plot_one_run(axis, logs[0], metrics, log_y)
    else:
        _plot_multiple_runs(axis, logs, metrics, log_y)
    axis.set_title(f"{stage} training metrics")
    axis.set_xlabel("step")
    axis.set_ylabel("metric value")
    if log_y:
        axis.set_yscale("log")
    axis.grid(alpha=0.25)

    output = _output_path(stage, [selector for selector, _ in runs], metrics, log_y)
    output.parent.mkdir(parents=True, exist_ok=True)
    figure.savefig(output, dpi=160)
    plt.close(figure)
    return output


def _plot_one_run(
    axis: plt.Axes,
    log: tuple[str, Path, list[dict[str, str]], list[str]],
    metrics: list[str],
    log_y: bool,
) -> None:
    _, _, rows, _ = log
    steps = [float(row["step"]) for row in rows]
    colors = plt.get_cmap("tab10")
    for index, metric in enumerate(metrics):
        values = [_float_or_nan(row[metric]) for row in rows]
        _validate_log_values(values, metric, log_y)
        axis.plot(
            steps,
            values,
            color=colors(index % colors.N),
            marker="o",
            markersize=3,
            linewidth=1.5,
            label=metric,
        )
    axis.legend(title="metric")


def _plot_multiple_runs(
    axis: plt.Axes,
    logs: list[tuple[str, Path, list[dict[str, str]], list[str]]],
    metrics: list[str],
    log_y: bool,
) -> None:
    colors = plt.get_cmap("tab10")
    for run_index, (_, run, rows, _) in enumerate(logs):
        steps = [float(row["step"]) for row in rows]
        color = colors(run_index % colors.N)
        for metric_index, metric in enumerate(metrics):
            values = [_float_or_nan(row[metric]) for row in rows]
            _validate_log_values(values, f"{run.name} · {metric}", log_y)
            axis.plot(
                steps,
                values,
                color=color,
                linestyle=_LINE_STYLES[metric_index % len(_LINE_STYLES)],
                marker="o",
                markersize=2.5,
                linewidth=1.5,
            )
    axis.legend(
        handles=[
            Line2D([0], [0], color=colors(index % colors.N), label=_legend_label(run.name))
            for index, (_, run, _, _) in enumerate(logs)
        ],
        title="run",
    )
    style_key = "  ".join(
        f"{metric}: {_LINE_STYLES[index % len(_LINE_STYLES)]}" for index, metric in enumerate(metrics)
    )
    axis.text(0.01, 0.01, f"line style  {style_key}", transform=axis.transAxes, va="bottom")


def _normalize_selector(selector: str) -> str:
    if Path(selector).name != selector:
        raise ValueError("--runs 只接受 outputs/runs 下的运行名，不接受文件路径")
    match = re.fullmatch(r"layer(\d+)(.*)", selector)
    return f"layer-{match.group(1)}{match.group(2)}" if match else selector


def _expand_selectors(selectors: Iterable[str]) -> list[str]:
    """将 ``layer-[0,1]`` 或 ``layer-[0:24]`` 展开为单层运行选择器。"""
    expanded = []
    for selector in selectors:
        match = re.fullmatch(r"layer-?\[([0-9]+(?::[0-9]+)?(?:,[0-9]+(?::[0-9]+)?)*)\]", selector)
        if match is None:
            expanded.append(selector)
            continue
        for item in match.group(1).split(","):
            if ":" not in item:
                expanded.append(f"layer-{item}")
                continue
            start, end = (int(value) for value in item.split(":"))
            if start > end:
                raise ValueError(f"层范围起点不能大于终点：{selector}")
            expanded.extend(f"layer-{index}" for index in range(start, end + 1))
    return expanded


def _resolve_run(selector: str) -> Path:
    normalized = _normalize_selector(selector)
    candidates = [
        path
        for stage in ("stage1", "stage2")
        # 目录格式为 ``layer-<index>-...``。必须保留 index 后的连字符
        # 边界，否则 ``layer-1`` 会错误匹配 layer-10 到 layer-19。
        for path in (_RUNS_ROOT / stage).glob(f"{normalized}-*")
        if path.is_dir() and (path / "training_log.csv").is_file()
    ]
    if not candidates:
        raise FileNotFoundError(f"未找到 outputs/runs 下匹配 {selector!r} 的 training_log.csv")
    return max(candidates, key=lambda path: path.name)


def _read_log(source: Path) -> tuple[list[dict[str, str]], list[str]]:
    with source.open(newline="", encoding="utf-8") as file:
        reader = csv.DictReader(file)
        rows = list(reader)
        fields = reader.fieldnames or []
    if not rows:
        raise ValueError(f"训练日志为空：{source}")
    if "step" not in fields:
        raise ValueError(f"{source} 缺少 step 列")
    return rows, fields


def _output_path(stage: str, selectors: list[str], metrics: list[str], log_y: bool) -> Path:
    selection = _layer_filename(selectors)
    metric_part = "-".join(_filename_part(metric) for metric in metrics)
    suffix = "-log-y" if log_y else ""
    return _ANALYSIS_ROOT / stage / f"{selection}_{metric_part}{suffix}.png"


def _layer_filename(selectors: list[str]) -> str:
    """将 layer-21 与 layers-21-22 等选择器统一为紧凑层索引名称。"""
    layers = []
    for selector in selectors:
        match = re.match(r"layers?-(\d+(?:-\d+)*)", _normalize_selector(selector))
        if match is None:
            return "-vs-".join(_filename_part(value) for value in selectors)
        layers.extend(match.group(1).split("-"))
    return f"layer-{'-'.join(dict.fromkeys(layers))}"


def _legend_label(run_name: str) -> str:
    """从运行目录名移除 rank 与时间戳，只保留层索引。"""
    match = re.match(r"layers?-(\d+(?:-\d+)*)", run_name)
    return f"layer-{match.group(1)}" if match else run_name


def _filename_part(value: str) -> str:
    return re.sub(r"[^A-Za-z0-9._-]+", "-", value).strip("-")


def _float_or_nan(value: str | None) -> float:
    return float("nan") if value in (None, "") else float(value)


def _validate_log_values(values: list[float], label: str, log_y: bool) -> None:
    if not log_y:
        return
    invalid = [value for value in values if math.isfinite(value) and value <= 0]
    if invalid:
        raise ValueError(f"对数纵轴要求指标为正数：{label}")


def arguments() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="可视化 outputs/runs 中同一 Stage 的训练日志")
    parser.add_argument(
        "--runs",
        nargs="+",
        required=True,
        metavar="RUN",
        help="运行名，如 layer-21、layer-[21,22]、layer-[0:23] 或 layers-21-22-23；省略 rank/日期时自动选择最新运行",
    )
    parser.add_argument(
        "--metrics",
        nargs="+",
        required=True,
        help="要绘制的 training_log.csv 列名，例如 train_loss y_nmse validation_loss",
    )
    parser.add_argument("--log-y", action="store_true", help="将纵坐标设为对数刻度")
    return parser.parse_args()


def main() -> None:
    args = arguments()
    output = plot_training_logs(args.runs, args.metrics, args.log_y)
    print(f"已生成训练曲线：{output}")


if __name__ == "__main__":
    main()
