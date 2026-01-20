import torch
from typing import Optional, Tuple
from . import _C


def _per_token_quant_fp8(input: torch.tensor, op) -> Tuple[torch.Tensor, torch.Tensor]:
    output = torch.empty_like(input, dtype=torch.float8_e4m3fn)
    scales = torch.empty(size=(input.shape[0], 1), device=input.device, dtype=torch.float32)
    op(output, input, scales)
    return output, scales

def per_token_quant_fp8(input: torch.tensor) -> Tuple[torch.Tensor, torch.Tensor]:
    """Quantize the given input using per token quant method (BF16/FP16)"""
    if input.dtype == torch.bfloat16:
        return _per_token_quant_fp8(input, _C.per_token_quant_bf16_fp8)
    if input.dtype == torch.float16:
        return _per_token_quant_fp8(input, _C.per_token_quant_fp16_fp8)
    raise TypeError(f"per_token_quant_fp8 expects bf16/fp16, got {input.dtype}")

def per_token_quant_bf16_fp8(input: torch.tensor) -> Tuple[torch.Tensor, torch.Tensor]:
    """Quantize the given input using per token quant method"""
    return _per_token_quant_fp8(input, _C.per_token_quant_bf16_fp8)

def per_token_quant_fp16_fp8(input: torch.tensor) -> Tuple[torch.Tensor, torch.Tensor]:
    """Quantize the given input using per token quant method"""
    return _per_token_quant_fp8(input, _C.per_token_quant_fp16_fp8)

def per_token_quant_bf16_int8(input: torch.tensor) -> Tuple[torch.Tensor, torch.Tensor]:
    """Quantize the given input using per token quant method"""
    output = torch.empty_like(input, dtype=torch.int8)
    scales = torch.empty(size=(input.shape[0], 1), device=input.device, dtype=torch.float32)
    _C.per_token_quant_bf16_int8(output, input, scales)
    return output, scales

def per_token_quant_fp16_int8(input: torch.tensor) -> Tuple[torch.Tensor, torch.Tensor]:
    """Quantize the given input using per token quant method"""
    output = torch.empty_like(input, dtype=torch.int8)
    scales = torch.empty(size=(input.shape[0], 1), device=input.device, dtype=torch.float32)
    _C.per_token_quant_fp16_int8(output, input, scales)
    return output, scales

def _per_token_quant_int8(input: torch.tensor, op) -> Tuple[torch.Tensor, torch.Tensor]:
    output = torch.empty_like(input, dtype=torch.int8)
    scales = torch.empty(size=(input.shape[0], 1), device=input.device, dtype=torch.float32)
    op(output, input, scales)
    return output, scales

def per_token_quant_int8(input: torch.tensor) -> Tuple[torch.Tensor, torch.Tensor]:
    """Quantize the given input using per token quant method (BF16/FP16)"""
    if input.dtype == torch.bfloat16:
        return _per_token_quant_int8(input, _C.per_token_quant_bf16_int8)
    if input.dtype == torch.float16:
        return _per_token_quant_int8(input, _C.per_token_quant_fp16_int8)
    raise TypeError(f"per_token_quant_int8 expects bf16/fp16, got {input.dtype}")
