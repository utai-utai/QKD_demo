#!/usr/bin/env python3
"""用 lm-evaluation-harness 比较原始模型与 Stage-2 压缩模型的公开任务分数。

安装（服务器 qkd 环境）：python -m pip install 'lm_eval[hf]'
任务名以安装版本的 ``lm_eval --tasks list`` 为准，例如 mmlu_pro、hellaswag、ceval。
"""

from __future__ import annotations

import argparse
import json
import os
from pathlib import Path

import torch

from qkd.modeling import load_causal_lm, load_tokenizer
from qkd.photonic.checkpoint import load_compressed_modules, read_compressed_checkpoint_config
from qkd.photonic.model import make_compressed_student, make_truncated_svd_student
from qkd.training.tools import provider_factory, training_device


def arguments() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="公开下游基准：原始教师或 Stage-2 压缩模型")
    parser.add_argument("--teacher", required=True)
    parser.add_argument("--tasks", nargs="+", required=True, help="lm-eval task 名，例如 mmlu_pro hellaswag")
    parser.add_argument("--checkpoint", help="Stage-2 checkpoint；省略时评估原始教师")
    parser.add_argument("--svd-baseline", action="store_true", help="无训练纯截断 SVD：W≈PB")
    parser.add_argument("--rank", type=int, default=512)
    parser.add_argument("--target-layers", nargs="+", type=int, default=[28, 29, 30, 31])
    parser.add_argument("--gate-scale", type=float, default=1.0)
    parser.add_argument("--batch-size", type=int, default=1)
    parser.add_argument("--max-length", type=int, default=1024, help="lm-eval 单次前向的最大 token 数；限制 rolling likelihood 的 logits 峰值显存")
    parser.add_argument("--num-fewshot", type=int, default=0)
    parser.add_argument("--limit", type=int, help="每个任务最多评估的样本数；仅用于冒烟，不可报告为正式分数")
    parser.add_argument("--hf-home", type=Path, default=Path("data/benchmark_cache"), help="离线 Hugging Face cache 根目录")
    parser.add_argument("--online", action="store_true", help="允许联网下载；默认严格离线")
    parser.add_argument("--output", required=True)
    return parser.parse_args()


def make_model(args: argparse.Namespace, device: torch.device):
    if args.checkpoint is not None and args.svd_baseline:
        raise ValueError("--checkpoint 与 --svd-baseline 只能选一个")
    if args.svd_baseline:
        teacher = load_causal_lm(args.teacher, trainable=False).to(device).eval()
        target_layers = tuple(args.target_layers)
        student = make_truncated_svd_student(teacher, args.rank, target_layers)
        return student.to(device).eval(), {"model_kind": "truncated_svd_only", "checkpoint": None, "target_layers": list(target_layers), "rank": args.rank, "definition": "gate/up: rank r; down: rank 2r; W≈PB; no g, C, provider, or training"}
    if args.checkpoint is None:
        teacher = load_causal_lm(args.teacher, trainable=False).to(device).eval()
        return teacher, {"model_kind": "teacher", "checkpoint": None}
    metadata = read_compressed_checkpoint_config(args.checkpoint)
    spec = metadata["spec"]
    if not isinstance(spec, dict):
        raise ValueError("checkpoint 缺少 spec")
    run_data = json.loads((Path(args.checkpoint) / "run.json").read_text(encoding="utf-8"))
    photonic = run_data.get("config", {}).get("photonic", {})
    if not isinstance(photonic, dict):
        photonic = {}
    compression = run_data.get("config", {}).get("compression", {})
    if not isinstance(compression, dict):
        compression = {}
    z_dim = int(spec["z_dim"])
    target_layers = tuple(int(index) for index in spec["target_layers"])
    student_name = str(metadata.get("student", args.teacher))
    student_base = load_causal_lm(student_name, trainable=False)
    student, replacements = make_compressed_student(
        student_base,
        provider_factory(str(metadata["provider"]), z_dim, photonic.get("ema_decay"), int(photonic.get("modes", z_dim)), int(photonic.get("layers", z_dim)), n_meshes=int(photonic.get("meshes", 1))),
        int(spec["rank"]), z_dim, float(spec["kappa"]), target_layers, args.gate_scale,
        encoded_input_mode=str(compression.get("encoded_input_mode", "input_dependent")),
        fixed_encoded_std=float(compression.get("fixed_encoded_std", 0.1)),
        pb_initialization=str(compression.get("pb_initialization", "svd")),
    )
    for replacement in replacements:
        replacement.shots = photonic.get("shots")
    load_compressed_modules(args.checkpoint, replacements, int(spec["rank"]), z_dim, float(spec["kappa"]), target_layers)
    return student.to(device).eval(), {"model_kind": "stage2_compressed", "checkpoint": str(Path(args.checkpoint)), "teacher": metadata.get("teacher", args.teacher), "student": student_name, "provider": metadata["provider"], "target_layers": list(target_layers), "rank": int(spec["rank"]), "gate_scale": args.gate_scale}


def main() -> None:
    args = arguments()
    # 必须在导入 transformers/lm_eval 前设定，以让两者使用同一可搬运离线 cache。
    hf_home = args.hf_home.resolve()
    os.environ["HF_HOME"] = str(hf_home)
    os.environ["HF_DATASETS_CACHE"] = str(hf_home / "datasets")
    if not args.online:
        os.environ["HF_HUB_OFFLINE"] = "1"
        os.environ["HF_DATASETS_OFFLINE"] = "1"
    try:
        from lm_eval import simple_evaluate
        from lm_eval.models.huggingface import HFLM
        import lm_eval
    except ImportError as error:
        raise SystemExit("缺少 lm-evaluation-harness；运行：python -m pip install 'lm_eval[hf]'") from error
    device = training_device()
    model, details = make_model(args, device)
    # 训练阶段会关闭 cache 以压低反向传播峰值显存；下游生成评测没有反向传播，
    # 应恢复 cache，避免每生成一个 token 都重新计算完整前缀。
    model.config.use_cache = True
    text_config = getattr(model.config, "text_config", None)
    if text_config is not None and hasattr(text_config, "use_cache"):
        text_config.use_cache = True
    tokenizer = load_tokenizer(str(details.get("student", args.teacher)))
    lm = HFLM(pretrained=model, tokenizer=tokenizer, batch_size=args.batch_size, max_length=args.max_length)
    print(f"[lm-eval] 请求 batch_size={args.batch_size}；HFLM 实际 batch_size={lm.batch_size}；max_length={lm.max_length}")
    results = simple_evaluate(
        model=lm, tasks=args.tasks, num_fewshot=args.num_fewshot,
        batch_size=args.batch_size, limit=args.limit,
    )
    payload = {
        **details,
        "teacher": args.teacher,
        "tasks": args.tasks,
        "num_fewshot": args.num_fewshot,
        "batch_size": args.batch_size,
        "max_length": args.max_length,
        "limit": args.limit,
        "hf_home": str(hf_home),
        "offline": not args.online,
        "lm_eval_version": getattr(lm_eval, "__version__", "unknown"),
        "results": results,
    }
    output = Path(args.output)
    output.parent.mkdir(parents=True, exist_ok=True)
    output.write_text(json.dumps(payload, ensure_ascii=False, indent=2, default=str) + "\n", encoding="utf-8")
    print(json.dumps({"output": str(output), "results": results.get("results", {})}, ensure_ascii=False, indent=2, default=str))


if __name__ == "__main__":
    main()
