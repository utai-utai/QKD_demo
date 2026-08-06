"""确定性切分 JSONL 对话，并写出带损失掩码的 token 化 JSONL。"""

from __future__ import annotations

import argparse
from pathlib import Path

from qkd.data import read_jsonl, split_records, tokenize_messages, write_jsonl
from qkd.modeling import DEFAULT_STUDENT_MODEL, load_tokenizer


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--input",
        nargs="+",
        required=True,
        help="一个或多个包含 messages 字段的源 JSONL；会在切分前合并。",
    )
    parser.add_argument("--output-dir", required=True)
    parser.add_argument("--tokenizer", default=DEFAULT_STUDENT_MODEL)
    parser.add_argument("--max-length", type=int, default=512)
    parser.add_argument("--validation-fraction", type=float, default=0.1)
    parser.add_argument("--seed", type=int, default=42)
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    records = []
    for source in args.input:
        records.extend(read_jsonl(source))
    train, validation = split_records(records, args.validation_fraction, args.seed)
    tokenizer = load_tokenizer(args.tokenizer)

    def prepare(records: list[dict]) -> tuple[list[dict], int]:
        prepared, skipped = [], 0
        for record in records:
            try:
                tokenized = tokenize_messages(tokenizer, record["messages"], args.max_length)
            except ValueError as error:
                # 多轮对话可能在截断位置前还没有 assistant 回复；这种样本的标签全为
                # IGNORE_INDEX，不能贡献 Stage 1 的训练损失，故明确跳过并计数。
                if str(error) == "conversation has no assistant tokens after tokenization":
                    skipped += 1
                    continue
                raise
            prepared.append({"messages": record["messages"], **tokenized})
        return prepared, skipped

    output_dir = Path(args.output_dir)
    prepared_train, skipped_train = prepare(train)
    prepared_validation, skipped_validation = prepare(validation)
    write_jsonl(output_dir / "train.jsonl", prepared_train)
    write_jsonl(output_dir / "validation.jsonl", prepared_validation)
    print(
        f"Wrote {len(prepared_train)} train and {len(prepared_validation)} validation records to {output_dir} "
        f"(skipped {skipped_train} train and {skipped_validation} validation records without assistant tokens after truncation)"
    )


if __name__ == "__main__":
    main()
