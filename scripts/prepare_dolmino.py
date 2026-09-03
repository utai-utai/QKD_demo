#!/usr/bin/env python3
"""将已下载的 Dolmino zstd 分片流式转换为 Stage-2 所需 tokenized JSONL。"""
from __future__ import annotations

import argparse
import hashlib
import json
import shutil
import subprocess
from pathlib import Path
from typing import Iterator

from qkd.modeling import load_tokenizer


def arguments() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="预处理本地 Dolma 3 Dolmino Mix 分片")
    parser.add_argument("--raw-dir", type=Path, default=Path("data/raw/dolma3_dolmino_mix_10b"))
    parser.add_argument("--output-dir", type=Path, default=Path("data/processed/dolma3_dolmino_mix_10b"))
    parser.add_argument("--tokenizer", default="models/Qwen3.5-4B-Base")
    parser.add_argument("--max-length", type=int, default=2048)
    parser.add_argument("--validation-fraction", type=float, default=0.0529)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--overwrite", action="store_true")
    return parser.parse_args()


def zstd_lines(path: Path) -> Iterator[str]:
    if shutil.which("zstdcat") is None:
        raise RuntimeError("找不到 zstdcat；请在服务器安装 zstd 后再预处理")
    with subprocess.Popen(["zstdcat", str(path)], stdout=subprocess.PIPE, text=True, encoding="utf-8", errors="replace") as process:
        assert process.stdout is not None
        yield from process.stdout
    if process.returncode:
        raise RuntimeError(f"zstdcat 解压失败：{path}")


def is_validation(record_id: str, seed: int, fraction: float) -> bool:
    digest = hashlib.sha256(f"{seed}:{record_id}".encode()).digest()
    return int.from_bytes(digest[:8], "big") / 2**64 < fraction


def main() -> None:
    args = arguments()
    if not 0 < args.validation_fraction < 1:
        raise ValueError("--validation-fraction 必须在 (0, 1) 内")
    manifest_path = args.raw_dir / "manifest.json"
    manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
    paths = manifest.get("paths")
    if not isinstance(paths, list) or not paths:
        raise ValueError(f"{manifest_path} 缺少 paths")
    shards = [args.raw_dir / str(relative) for relative in paths]
    missing = [str(path) for path in shards if not path.is_file()]
    if missing:
        raise FileNotFoundError(f"缺少 {len(missing)} 个 Dolmino 分片，例如：{missing[0]}")
    if args.output_dir.exists() and any(args.output_dir.iterdir()) and not args.overwrite:
        raise FileExistsError(f"{args.output_dir} 已存在；确认覆盖请加 --overwrite")
    args.output_dir.mkdir(parents=True, exist_ok=True)
    tokenizer = load_tokenizer(args.tokenizer)
    counts = {"documents": 0, "train_sequences": 0, "validation_sequences": 0}
    train_path, validation_path = args.output_dir / "train.jsonl", args.output_dir / "validation.jsonl"
    with train_path.open("w", encoding="utf-8") as train, validation_path.open("w", encoding="utf-8") as validation:
        for shard_index, shard in enumerate(shards, 1):
            print(f"[{shard_index}/{len(shards)}] {shard}")
            for line in zstd_lines(shard):
                if not line.strip():
                    continue
                record = json.loads(line)
                text = record.get("text")
                if not isinstance(text, str) or not text.strip():
                    continue
                record_id = str(record.get("id", f"{shard}:{counts['documents']}"))
                ids = tokenizer(text, add_special_tokens=True, truncation=False)["input_ids"]
                target = validation if is_validation(record_id, args.seed, args.validation_fraction) else train
                count_key = "validation_sequences" if target is validation else "train_sequences"
                for start in range(0, len(ids), args.max_length):
                    chunk = ids[start:start + args.max_length]
                    if len(chunk) < 2:
                        continue
                    target.write(json.dumps({"input_ids": chunk, "attention_mask": [1] * len(chunk), "labels": chunk}) + "\n")
                    counts[count_key] += 1
                counts["documents"] += 1
    (args.output_dir / "metadata.json").write_text(json.dumps({
        "source_manifest": str(manifest_path), "tokenizer": args.tokenizer,
        "max_length": args.max_length, "validation_fraction": args.validation_fraction,
        "seed": args.seed, **counts,
    }, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
    print(json.dumps(counts, ensure_ascii=False))


if __name__ == "__main__":
    main()
