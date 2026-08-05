"""将一个或多个 training_log.csv 中指定的指标绘制为训练曲线。"""

from __future__ import annotations

import argparse
import csv
import os
import tempfile
from collections.abc import Iterable
from pathlib import Path

_CACHE_DIR = Path(tempfile.gettempdir()) / "qkd-matplotlib"
_CACHE_DIR.mkdir(parents=True, exist_ok=True)
os.environ.setdefault("MPLCONFIGDIR", str(_CACHE_DIR))
os.environ.setdefault("XDG_CACHE_HOME", str(_CACHE_DIR))

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt


def plot_training_logs(
    series: Iterable[tuple[str | Path, str]], output_path: str | Path | None = None
) -> Path:
    """绘制 ``(日志路径, 指标名)`` 序列，并返回生成图片的路径。"""
    requested = [(Path(path), metric) for path, metric in series]
    if not requested:
        raise ValueError("至少指定一个要绘制的日志指标")

    figure, axis = plt.subplots(figsize=(10, 5.5), constrained_layout=True)
    colors = plt.get_cmap("tab20")
    for index, (source, metric) in enumerate(requested):
        rows, fields = _read_log(source)
        if metric not in fields:
            raise ValueError(f"{source} 不包含指标 {metric}；可用列：{', '.join(fields)}")
        steps = [float(row["step"]) for row in rows]
        values = [_float_or_nan(row[metric]) for row in rows]
        axis.plot(
            steps,
            values,
            color=colors(index % colors.N),
            marker="o",
            markersize=3,
            linewidth=1.5,
            label=f"{source.parent.name} · {metric}",
        )
    axis.set_title("Training metrics")
    axis.set_xlabel("step")
    axis.set_ylabel("metric value")
    axis.grid(alpha=0.25)
    axis.legend()

    destination = (
        Path(output_path)
        if output_path is not None
        else requested[0][0].parent / "training_plot.png"
    )
    destination.parent.mkdir(parents=True, exist_ok=True)
    figure.savefig(destination, dpi=160)
    plt.close(figure)
    return destination


def _read_log(source: Path) -> tuple[list[dict[str, str]], list[str]]:
    if not source.is_file():
        raise FileNotFoundError(f"训练日志不存在：{source}")
    with source.open(newline="", encoding="utf-8") as file:
        reader = csv.DictReader(file)
        rows = list(reader)
        fields = reader.fieldnames or []
    if not rows:
        raise ValueError(f"训练日志为空：{source}")
    if "step" not in fields:
        raise ValueError(f"{source} 缺少 step 列")
    return rows, fields


def _float_or_nan(value: str | None) -> float:
    return float("nan") if value in (None, "") else float(value)


def arguments() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="可视化一个或多个 training_log.csv 中指定的指标")
    parser.add_argument("--input", nargs="+", help="training_log.csv 路径；与 --metrics 组合使用")
    parser.add_argument(
        "--metrics",
        nargs="+",
        help="要绘制的 CSV 列名；会应用到每个 --input 文件",
    )
    parser.add_argument(
        "--series",
        action="append",
        nargs=2,
        metavar=("FILE", "METRIC"),
        help="单条曲线的日志与指标；可重复，可与 --input/--metrics 混用",
    )
    parser.add_argument("--output", help="PNG 输出路径；默认写在日志同目录")
    return parser.parse_args()


def main() -> None:
    args = arguments()
    series = list(args.series or [])
    if args.input or args.metrics:
        if not args.input or not args.metrics:
            raise ValueError("--input 和 --metrics 必须同时提供")
        series.extend((path, metric) for path in args.input for metric in args.metrics)
    output = plot_training_logs(series, args.output)
    print(f"已生成训练曲线：{output}")


if __name__ == "__main__":
    main()
