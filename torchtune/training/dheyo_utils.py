import torch
from torchao.quantization.quant_primitives import MappingType, dequantize_affine, quantize_affine, ZeroPointDomain
from torchao.quantization.utils import _get_per_token_block_size

from typing import List
from torchtune.utils.dheyo_quant_primitives import choose_qparams_affine_float, fake_quantize_float_affine_cachemask, TorchAODTypeFloat, _DTYPE_TO_QVALUE_BOUNDS


def per_token_dynamic_quant_float( ### TODO: This won't be used in QAT so take care of it later
    input: torch.Tensor,
    scale_dtype: torch.dtype = torch.float32,
    zero_point_dtype: torch.dtype = torch.float32,
) -> torch.Tensor:
    mapping_type = MappingType.ASYMMETRIC
    block_size = _get_per_token_block_size(input)
    quant_min = -6.0
    quant_max = 6.0
    quant_dtype = torch.bfloat16
    output_dtype = input.dtype

    scales, zero_points = choose_qparams_affine_float(
        input,
        mapping_type,
        block_size,
        quant_dtype,
        quant_min,
        quant_max,
        scale_dtype=scale_dtype,
        zero_point_dtype=zero_point_dtype,
    )
    import pdb; pdb.set_trace()
    q = quantize_affine(
        input,
        block_size,
        scales,
        zero_points,
        quant_dtype,
        quant_min,
        quant_max,
    )
    dq = dequantize_affine(
        q,
        block_size,
        scales,
        zero_points,
        quant_dtype,
        quant_min,
        quant_max,
        output_dtype=output_dtype,
    )
    return dq


def get_group_qparams_symmetric_float(
    w,
    n_bit=4,
    groupsize=32,
    precision=torch.float32,
    mapping_type=MappingType.SYMMETRIC,
    representation_dtype=TorchAODTypeFloat.FLOAT4_E2M1
):
    # needed for GPTQ with padding
    if groupsize > w.shape[-1]:
        groupsize = w.shape[-1]
    assert groupsize > 1
    assert w.shape[-1] % groupsize == 0
    assert w.dim() == 2
    assert n_bit <= 8, f"unsupported n_bit: {n_bit}"

    block_size = (1, groupsize)
    eps = torch.finfo(w.dtype).eps
    # ranges = {}
    # ranges[1] = (-1, 0)
    ## generating ranges for bit 2 to 8
    # for i in range(4, 5): ## only adding support for FLOAT4_E2M1
        # ranges[i] = (-6.0, 6.0) ## TODO: Expand the range with proper EM formula
    # quant_min, quant_max = ranges[n_bit]
    quant_min, quant_max = _DTYPE_TO_QVALUE_BOUNDS[representation_dtype]
    scale, zero_point = choose_qparams_affine_float(
        w,
        mapping_type,
        block_size,
        target_dtype=torch.bfloat16,
        quant_min=quant_min,
        quant_max=quant_max,
        eps=eps,
        scale_dtype=precision,
        zero_point_dtype=precision,
        representation_dtype=representation_dtype
    )
    return scale.reshape(w.shape[0], -1), zero_point.reshape(w.shape[0], -1)



class _GenericFakeQuantizeWrapper(torch.autograd.Function): ## TODO: verify if you need to extend _GenericFakeQuantizerWrapper or not
    """
    Implementation of generic fake quantize with backward STE.

    With the appropriate input tensor shape, this can be used to express
    grouped per channel fake quantize or per token fake quantize.
    """

    @staticmethod
    def forward(
        ctx: torch.autograd.function.FunctionCtx,
        input: torch.Tensor,
        block_size: List[int],
        scales: torch.Tensor,
        zero_points: torch.Tensor,
        quant_min: float,
        quant_max: float,
        zero_point_domain: ZeroPointDomain = ZeroPointDomain.FLOAT,
        representation_dype: TorchAODTypeFloat = TorchAODTypeFloat.FLOAT4_E2M1
    ) -> torch.Tensor:
        # avoid circular dependencies
        from torchao.quantization.qat.affine_fake_quantized_tensor import (
            AffineFakeQuantizedTensor,
        )

        if isinstance(input, AffineFakeQuantizedTensor):
            _input = input.original_tensor
        else:
            _input = input

        (fq, mask) = fake_quantize_float_affine_cachemask( ## check this
            _input,
            block_size,
            scales,
            zero_points,
            torch.bfloat16,
            quant_min,
            quant_max,
            zero_point_domain,
            representation_dype
        )

        ctx.save_for_backward(mask)
        return fq

    @staticmethod
    def backward(ctx, gy):
        (mask,) = ctx.saved_tensors
        return gy * mask, None, None, None, None, None, None



def _fake_quantize_per_channel_group(
    input: torch.Tensor,
    scales: torch.Tensor,
    zero_points: torch.Tensor,
    quant_min: int,
    quant_max: int,
    group_size: int,
    zero_point_domain: ZeroPointDomain = ZeroPointDomain.FLOAT,
    representation_dtype: TorchAODTypeFloat = TorchAODTypeFloat.FLOAT4_E2M1
) -> torch.Tensor:
    assert group_size > 1
    assert input.shape[-1] % group_size == 0
    assert input.dim() == 2
    block_size = (1, group_size)
    return _GenericFakeQuantizeWrapper.apply(
        input,
        block_size,
        scales,
        zero_points,
        quant_min,
        quant_max,
        zero_point_domain,
        representation_dtype
    )