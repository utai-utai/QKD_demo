"""光子低秩压缩训练入口共用的模型加载工具。"""

from __future__ import annotations

from typing import Any

import torch

DEFAULT_COMPRESSION_TEACHER = "Qwen/Qwen3.5-0.8B-Base"
DEFAULT_STUDENT_MODEL = DEFAULT_COMPRESSION_TEACHER


def _patch_transformers_fp8_tp_plan_bug() -> None:
    """兼容 Transformers 5.16.x 的 fine-grained FP8 单卡加载回归。

    当 Qwen3.5 没有 experts implementation 时，该版本仍会对空的
    ``layer_overrides`` 调用 ``.get``。原始方法在报错前已写入 Qwen 的
    tensor-parallel plan；单卡 ``device_map`` 不需要不存在的 experts 重写，
    因而安全地返回该配置即可。
    """
    from transformers.quantizers.quantizer_finegrained_fp8 import FineGrainedFP8HfQuantizer

    quantizer_type = FineGrainedFP8HfQuantizer
    if getattr(quantizer_type, "_qkd_safe_tp_plan", False):
        return
    original_update = quantizer_type.update_tp_plan

    def safe_update(self: Any, config: Any) -> Any:
        try:
            return original_update(self, config)
        except AttributeError as error:
            if "'NoneType' object has no attribute 'get'" not in str(error):
                raise
            return config

    quantizer_type.update_tp_plan = safe_update
    quantizer_type._qkd_safe_tp_plan = True


def _allow_qwen_fp8_hub_kernel() -> None:
    """允许本进程加载 Transformers 指定的 finegrained-FP8 Hub kernel。

    AutoDL 在无法联网验证 ``kernels-community`` 发布者元数据时，会拒绝
    该 kernel。这里仅在官方 Qwen FP8 教师加载路径开启该显式许可；kernel
    的具体仓库与版本仍由 Transformers 固定为 finegrained-fp8。
    """
    from transformers.integrations import hub_kernels

    hub_kernels.ALLOW_ALL_KERNELS = True


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
        # Transformers 5.16.x 的 FP8 quantizer 会假定这两个计划为映射，
        # 但 Qwen3.5 的复合配置默认给出 None；加载前归一化以兼容官方
        # Qwen3.5-27B-FP8，而不修改模型目录中的 config.json。
        from transformers import AutoConfig

        config = AutoConfig.from_pretrained(model_name, trust_remote_code=True)
        for attribute in ("base_model_tp_plan", "base_model_ep_plan"):
            if getattr(config, attribute, None) is None:
                setattr(config, attribute, {})
        kwargs["config"] = config
        kwargs["device_map"] = {"": str(device)}
        _patch_transformers_fp8_tp_plan_bug()
        _allow_qwen_fp8_hub_kernel()
    else:
        kwargs["dtype"] = dtype
    model = model_loader.from_pretrained(model_name, **kwargs)
    model.config.use_cache = False
    for parameter in model.parameters():
        parameter.requires_grad = trainable
    return model
