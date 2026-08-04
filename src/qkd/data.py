"""JSONL 对话数据读取与 PyTorch 数据集工具。"""

from __future__ import annotations

import json
import random
from collections.abc import Mapping
from pathlib import Path
from typing import Any, Iterable

import torch
from torch.utils.data import Dataset

IGNORE_INDEX = -100


def read_jsonl(path: str | Path) -> list[dict[str, Any]]:
    """读取并校验包含非空 ``messages`` 列表的 JSONL 记录。"""
    records: list[dict[str, Any]] = []
    for line_number, line in enumerate(Path(path).read_text(encoding="utf-8").splitlines(), 1):
        if not line.strip():
            continue
        record = json.loads(line)
        messages = record.get("messages")
        if not isinstance(messages, list) or not messages:
            raise ValueError(f"{path}:{line_number}: expected a non-empty 'messages' list")
        if any(not isinstance(message.get("role"), str) or not isinstance(message.get("content"), str)
               for message in messages):
            raise ValueError(f"{path}:{line_number}: every message needs string role and content")
        records.append(record)
    if not records:
        raise ValueError(f"{path}: no JSONL records found")
    return records


def write_jsonl(path: str | Path, records: Iterable[dict[str, Any]]) -> None:
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8") as handle:
        for record in records:
            handle.write(json.dumps(record, ensure_ascii=False) + "\n")


def split_records(records: list[dict[str, Any]], validation_fraction: float, seed: int) -> tuple[list[dict[str, Any]], list[dict[str, Any]]]:
    """以固定随机种子生成确定性的训练/验证切分。"""
    if not 0 < validation_fraction < 1:
        raise ValueError("validation_fraction must be between 0 and 1")
    if len(records) < 2:
        raise ValueError("at least two records are required for a train/validation split")
    shuffled = list(records)
    random.Random(seed).shuffle(shuffled)
    validation_size = max(1, round(len(shuffled) * validation_fraction))
    validation_size = min(validation_size, len(shuffled) - 1)
    return shuffled[validation_size:], shuffled[:validation_size]


def _as_list(value: Any) -> list[int]:
    if isinstance(value, torch.Tensor):
        return value.tolist()
    return list(value)


def _input_ids(encoded: Any) -> list[int]:
    """从普通列表或类似 BatchEncoding 的映射中提取 token ID。"""
    if isinstance(encoded, Mapping):
        encoded = encoded["input_ids"]
    return _as_list(encoded)


def tokenize_messages(tokenizer: Any, messages: list[dict[str, str]], max_length: int) -> dict[str, list[int]]:
    """将对话编码，仅为 assistant 回复 token 设置标签。
    新版聊天模板可直接返回 assistant 掩码；回退路径比较各消息前缀，以兼容旧模板。
    """
    try:
        encoded = tokenizer.apply_chat_template(
            messages,
            tokenize=True,
            add_generation_prompt=False,
            return_dict=True,
            return_assistant_tokens_mask=True,
            truncation=True,
            max_length=max_length,
        )
        input_ids = _input_ids(encoded)
        assistant_mask = encoded.get("assistant_masks")
        assistant_mask = _as_list(assistant_mask) if assistant_mask is not None else []
        # 部分有效模板未使用 Transformers 的 `{% generation %}` 标签，会返回全 False 掩码而非抛出异常。
        if not any(assistant_mask):
            raise ValueError("chat template did not return an assistant mask")
        labels = [token if keep else IGNORE_INDEX for token, keep in zip(input_ids, assistant_mask)]
    except (TypeError, ValueError, KeyError):
        input_ids = _input_ids(tokenizer.apply_chat_template(
            messages, tokenize=True, add_generation_prompt=False, truncation=True, max_length=max_length
        ))
        labels = [IGNORE_INDEX] * len(input_ids)
        for index, message in enumerate(messages):
            if message["role"] != "assistant":
                continue
            before = _input_ids(tokenizer.apply_chat_template(
                messages[:index], tokenize=True, add_generation_prompt=False
            ))
            through = _input_ids(tokenizer.apply_chat_template(
                messages[: index + 1], tokenize=True, add_generation_prompt=False
            ))
            for position in range(len(before), min(len(through), len(input_ids))):
                labels[position] = input_ids[position]
    if not any(label != IGNORE_INDEX for label in labels):
        raise ValueError("conversation has no assistant tokens after tokenization")
    return {"input_ids": input_ids, "attention_mask": [1] * len(input_ids), "labels": labels}


class TokenizedChatDataset(Dataset[dict[str, torch.Tensor]]):
    """由 ``prepare_data.py`` 生成的 token 化 JSONL 支撑的数据集。"""
    def __init__(self, path: str | Path) -> None:
        self.records = read_jsonl(path)
        required = {"input_ids", "attention_mask", "labels"}
        if any(not required.issubset(record) for record in self.records):
            raise ValueError(f"{path} is not tokenized; run scripts/prepare_data.py first")

    def __len__(self) -> int:
        return len(self.records)

    def __getitem__(self, index: int) -> dict[str, torch.Tensor]:
        record = self.records[index]
        return {key: torch.tensor(record[key], dtype=torch.long)
                for key in ("input_ids", "attention_mask", "labels")}


def collate_tokenized(batch: list[dict[str, torch.Tensor]], pad_token_id: int) -> dict[str, torch.Tensor]:
    """填充输入与掩码，同时保持标签中的 ``-100`` 损失掩码。"""
    max_length = max(item["input_ids"].numel() for item in batch)
    result: dict[str, list[torch.Tensor]] = {"input_ids": [], "attention_mask": [], "labels": []}
    for item in batch:
        padding = max_length - item["input_ids"].numel()
        result["input_ids"].append(torch.nn.functional.pad(item["input_ids"], (0, padding), value=pad_token_id))
        result["attention_mask"].append(torch.nn.functional.pad(item["attention_mask"], (0, padding), value=0))
        result["labels"].append(torch.nn.functional.pad(item["labels"], (0, padding), value=IGNORE_INDEX))
    return {key: torch.stack(value) for key, value in result.items()}
