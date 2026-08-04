"""光子低秩压缩训练入口共用的模型加载工具。"""

from __future__ import annotations

from typing import Any

import torch

DEFAULT_COMPRESSION_TEACHER = "Qwen/Qwen3.5-0.8B-Base"
DEFAULT_STUDENT_MODEL = DEFAULT_COMPRESSION_TEACHER


def load_tokenizer(model_name: str) -> Any:
    from transformers import AutoTokenizer

    tokenizer = AutoTokenizer.from_pretrained(model_name, trust_remote_code=True)
    if tokenizer.pad_token_id is None:
        tokenizer.pad_token = tokenizer.eos_token
    tokenizer.padding_side = "right"
    return tokenizer


def load_causal_lm(model_name: str, *, trainable: bool) -> Any:
    """加载因果语言模型，调用方负责设备放置。"""
    from transformers import AutoModelForCausalLM

    dtype = torch.bfloat16 if torch.cuda.is_available() else torch.float32
    model = AutoModelForCausalLM.from_pretrained(
        model_name, dtype=dtype, trust_remote_code=True
    )
    model.config.use_cache = False
    for parameter in model.parameters():
        parameter.requires_grad = trainable
    return model
