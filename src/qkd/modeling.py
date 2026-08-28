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


def load_causal_lm(
    model_name: str,
    *,
    trainable: bool,
    model_class: str = "causal",
    preserve_quantization: bool = False,
    device: torch.device | None = None,
) -> Any:
    """加载文本或多模态语言模型。

    ``preserve_quantization`` 用于已量化教师（例如官方 Qwen FP8）：不传
    dtype，且由 ``device_map`` 在加载时直接放到目标设备，避免 `.to()`
    将量化权重转换回 bf16。
    """
    if model_class == "causal":
        from transformers import AutoModelForCausalLM as model_loader
    elif model_class == "multimodal":
        from transformers import AutoModelForMultimodalLM as model_loader
    else:
        raise ValueError("model_class 必须是 causal 或 multimodal")

    dtype = torch.bfloat16 if torch.cuda.is_available() else torch.float32
    kwargs: dict[str, Any] = {"trust_remote_code": True}
    if preserve_quantization:
        if device is None:
            raise ValueError("保留量化加载时必须提供 device")
        kwargs["device_map"] = {"": str(device)}
    else:
        kwargs["dtype"] = dtype
    model = model_loader.from_pretrained(model_name, **kwargs)
    model.config.use_cache = False
    for parameter in model.parameters():
        parameter.requires_grad = trainable
    return model
