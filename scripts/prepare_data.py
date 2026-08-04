"""确定性切分 JSONL 对话，并写出带损失掩码的 token 化 JSONL。"""

from __future__ import annotations

import argparse
from pathlib import Path

from qkd.data import read_jsonl, split_records, tokenize_messages, write_jsonl
from qkd.modeling import DEFAULT_STUDENT_MODEL, load_tokenizer


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("--input", required=True, help="Source JSONL with a messages field per row.")
    parser.add_argument("--output-dir", required=True)
    parser.add_argument("--tokenizer", default=DEFAULT_STUDENT_MODEL)
    parser.add_argument("--max-length", type=int, default=512)
    parser.add_argument("--validation-fraction", type=float, default=0.1)
    parser.add_argument("--seed", type=int, default=42)
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    records = read_jsonl(args.input)
    train, validation = split_records(records, args.validation_fraction, args.seed)
    tokenizer = load_tokenizer(args.tokenizer)

    def prepare(records: list[dict]) -> list[dict]:
        return [{"messages": record["messages"], **tokenize_messages(tokenizer, record["messages"], args.max_length)}
                for record in records]

    output_dir = Path(args.output_dir)
    write_jsonl(output_dir / "train.jsonl", prepare(train))
    write_jsonl(output_dir / "validation.jsonl", prepare(validation))
    print(f"Wrote {len(train)} train and {len(validation)} validation records to {output_dir}")


if __name__ == "__main__":
    main()
