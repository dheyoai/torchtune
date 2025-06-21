from enum import Enum, auto
import torch
from typing import Dict, Union, Tuple, Optional, List
from torchao.quantization.quant_primitives import (
    MappingType, 
    ZeroPointDomain, 
    # _quantize_affine_no_dtype_cast,
    _dequantize_affine_no_dtype_check,
    _quantize_affine_tinygemm_no_dtype_cast,
    _dequantize_affine_tinygemm_no_dtype_check,
    _quantize_affine_no_zero_point_no_dtype_cast,
    _dequantize_affine_no_zero_point_no_dtype_check
    )

from torchao.utils import (
    TORCH_VERSION_AT_LEAST_2_3,
    TORCH_VERSION_AT_LEAST_2_5,
    TORCH_VERSION_AT_LEAST_2_6,
    _is_float8_type,
    _register_custom_op,
)

from torchao.quantization.quant_primitives import FP8_TYPES, _get_reduction_params

quant_lib = torch.library.Library("torchao", "FRAGMENT")

register_custom_op = _register_custom_op(quant_lib)

def calculate_mx_range(exponent_bits, mantissa_bits):
    """
    Calculate the range for an MX format given exponent and mantissa bits.
    
    Args:
        exponent_bits (int): Number of exponent bits (e).
        mantissa_bits (int): Number of mantissa bits (m).
    
    Returns:
        tuple: (min_value, max_value) representing the range (-max, max).
    """
    bias = 2 ** (exponent_bits - 1) - 1
    max_exponent = (2 ** exponent_bits) - 1
    effective_exponent = max_exponent - bias
    max_mantissa = (2 ** mantissa_bits - 1) / (2 ** mantissa_bits)
    max_value = (1 + max_mantissa) * (2 ** effective_exponent)
    return -max_value, max_value


class TorchAODTypeFloat(Enum):
    """
    Placeholder for float dtypes that do not exist in PyTorch core yet.
    """
    FLOAT4_E2M1 = 4
    FLOAT6_E2M3 = 6
    FLOAT6_E3M2 = 6



_DTYPE_TO_QVALUE_BOUNDS: Dict[Union[torch.dtype, TorchAODTypeFloat], Tuple[float, float]] = {
    TorchAODTypeFloat.FLOAT4_E2M1: calculate_mx_range(2, 1), ## TODO: use the EM formula and expand this later
    TorchAODTypeFloat.FLOAT6_E2M3: calculate_mx_range(2, 3), ## TODO: use the EM formula and expand this later
    TorchAODTypeFloat.FLOAT6_E3M2: calculate_mx_range(3, 2), ## TODO: use the EM formula and expand this later

    torch.bfloat16: (-3.40e38, 3.40e38)


}
_DTYPE_TO_BIT_WIDTH: Dict[Union[torch.dtype, TorchAODTypeFloat], Tuple[int, int]] = {
    TorchAODTypeFloat.FLOAT4_E2M1: 4,
    TorchAODTypeFloat.FLOAT6_E2M3: 6,
    TorchAODTypeFloat.FLOAT6_E3M2: 6,

}


_DTYPE_TO_EXPONENT_BITS: Dict[Union[torch.dtype, TorchAODTypeFloat], Tuple[float, float]] = {
    TorchAODTypeFloat.FLOAT4_E2M1: 2,
    TorchAODTypeFloat.FLOAT6_E2M3: 2, 
    TorchAODTypeFloat.FLOAT6_E3M2: 3, 

}

_DTYPE_TO_BIAS: Dict[Union[torch.dtype, TorchAODTypeFloat], Tuple[float, float]] = {
    TorchAODTypeFloat.FLOAT4_E2M1: 2**(_DTYPE_TO_EXPONENT_BITS[TorchAODTypeFloat.FLOAT4_E2M1] - 1) - 1,
    TorchAODTypeFloat.FLOAT6_E2M3: 2**(_DTYPE_TO_EXPONENT_BITS[TorchAODTypeFloat.FLOAT6_E2M3] - 1) - 1, ## verify
    TorchAODTypeFloat.FLOAT6_E3M2: 2**(_DTYPE_TO_EXPONENT_BITS[TorchAODTypeFloat.FLOAT6_E3M2] - 1) - 1, ## verify

}


_DTYPE_TO_EMAX: Dict[Union[torch.dtype, TorchAODTypeFloat], Tuple[float, float]] = {
    TorchAODTypeFloat.FLOAT4_E2M1: 2**_DTYPE_TO_EXPONENT_BITS[TorchAODTypeFloat.FLOAT4_E2M1] - 1 - _DTYPE_TO_BIAS[TorchAODTypeFloat.FLOAT4_E2M1],
    TorchAODTypeFloat.FLOAT6_E2M3: 2**_DTYPE_TO_EXPONENT_BITS[TorchAODTypeFloat.FLOAT6_E2M3] - 1 - _DTYPE_TO_BIAS[TorchAODTypeFloat.FLOAT6_E2M3], ## verify
    TorchAODTypeFloat.FLOAT6_E3M2: 2**_DTYPE_TO_EXPONENT_BITS[TorchAODTypeFloat.FLOAT6_E3M2] - 1 - _DTYPE_TO_BIAS[TorchAODTypeFloat.FLOAT6_E3M2], ## verify

}

_DYPE_TO_MANTISSA_BITS: Dict[Union[torch.dtype, TorchAODTypeFloat], Tuple[float, float]] = {
    TorchAODTypeFloat.FLOAT4_E2M1: 1,
    TorchAODTypeFloat.FLOAT6_E2M3: 3, ## verify
    TorchAODTypeFloat.FLOAT6_E3M2: 2 ## verify

}

_SUB_BYTE_UINT_BOUNDS: Dict[Union[torch.dtype, TorchAODTypeFloat], Tuple[float, float]] = {}
_SUB_BYTE_INT_BOUNDS: Dict[Union[torch.dtype, TorchAODTypeFloat], Tuple[float, float]] = {
    TorchAODTypeFloat.FLOAT4_E2M1: calculate_mx_range(2, 1), ## TODO: use the EM formula and expand this later
    TorchAODTypeFloat.FLOAT6_E2M3: calculate_mx_range(2, 3), ## TODO: use the EM formula and expand this later
    TorchAODTypeFloat.FLOAT6_E3M2: calculate_mx_range(3, 2), ## TODO: use the EM formula and expand this later

}

# FP8_TYPES = {
#     torch.float8_e4m3fn,
#     torch.float8_e5m2,
#     torch.float8_e4m3fnuz,
#     torch.float8_e5m2fnuz,
#     # torch.float8_e8m0fnu
# }

# TODO: decide on if we want to allow custom quant_min/quant_max here
def _get_and_check_qmin_qmax(dtype, quant_min, quant_max):
    """Get quant_min and quant_max args based on dtype and also
    verify that they are within the range of possible quant_min/quant_max
    for dtype
    """
    if dtype in FP8_TYPES:
        quant_min_lower_bound, quant_max_upper_bound = (
            torch.finfo(dtype).min,
            torch.finfo(dtype).max,
        )
    elif dtype not in _DTYPE_TO_QVALUE_BOUNDS:
        raise ValueError(f"Unsupported dtype: {dtype}")
    else:
        quant_min_lower_bound, quant_max_upper_bound = _DTYPE_TO_QVALUE_BOUNDS[dtype]
    if quant_min is None:
        quant_min = quant_min_lower_bound
    if quant_max is None:
        quant_max = quant_max_upper_bound

    assert quant_min >= quant_min_lower_bound, (
        "quant_min out of bound for dtype, "
        f"quant_min_lower_bound: {quant_min_lower_bound} quant_min: {quant_min}"
    )

    assert quant_max <= quant_max_upper_bound, (
        "quant_max out of bound for dtype, "
        f"quant_max_upper_bound: {quant_max_upper_bound} quant_max: {quant_max}"
    )
    return quant_min, quant_max


# @register_custom_op
def _choose_qparams_affine_float(
    input: Optional[torch.Tensor],
    mapping_type: str,
    block_size: List[int],
    target_dtype: torch.dtype,
    quant_min: Optional[Union[int, float, bool]] = None,
    quant_max: Optional[Union[int, float, bool]] = None, ## condition on quant_max for EMAX choice in scale calculation
    eps: Optional[float] = None,
    scale_dtype: Optional[torch.dtype] = None,
    zero_point_dtype: Optional[torch.dtype] = None,
    preserve_zero: bool = True,
    zero_point_domain: Optional[str] = "FLOAT",
    min_val: Optional[torch.Tensor] = None,
    max_val: Optional[torch.Tensor] = None,
    representation_dtype: TorchAODTypeFloat = TorchAODTypeFloat.FLOAT4_E2M1
) -> Tuple[torch.Tensor, torch.Tensor]:
    """op definition that has compatible signatures with custom op library

    The op does the following:
    1. figure out the dimension for reduction based on block_size
    2. find min_val/max_val based on the dimension for reduction
    3. calculate quantization parameters based on min_val/max_val based on args like `preserve_zero`
       and `zero_point_domain`
    """
    quant_min, quant_max = _get_and_check_qmin_qmax(target_dtype, quant_min, quant_max)
    assert mapping_type in [
        MappingType.SYMMETRIC.name,
        MappingType.SYMMETRIC_NO_CLIPPING_ERR.name,
        MappingType.ASYMMETRIC.name,
    ], f"Unsupported mapping type: {mapping_type}"
    if target_dtype in FP8_TYPES:
        assert mapping_type == MappingType.SYMMETRIC.name, (
            f"Only symmetric quantization is supported for FP8 types, got {mapping_type}"
        )

    if input is not None:
        if scale_dtype is None:
            scale_dtype = input.dtype
        if eps is None:
            eps = torch.finfo(input.dtype).eps

        assert len(block_size) == input.dim(), (
            f"Got input dim:{input.dim()}, block_size: {block_size}"
        )
        shape_for_reduction, reduction_dims = _get_reduction_params(
            block_size, input.size()
        )
        input = input.view(shape_for_reduction)

        min_val = torch.amin(input, dim=reduction_dims, keepdim=False)
        max_val = torch.amax(input, dim=reduction_dims, keepdim=False)
    else:
        assert min_val is not None and max_val is not None, (
            "Need to provide `min_val` and `max_val` when `input` is None, got: {min_val, max_val}"
        )
        assert min_val.dtype == max_val.dtype, (
            "Expecting `min_val` and `max_val` to have the same dtype, got: {min_val.dtype, max_val.dtype}"
        )

        if scale_dtype is None:
            scale_dtype = min_val.dtype
        if eps is None:
            eps = torch.finfo(min_val.dtype).eps

    if preserve_zero:
        min_val_neg = torch.min(min_val, torch.zeros_like(min_val))
        max_val_pos = torch.max(max_val, torch.zeros_like(max_val))
    else:
        min_val_neg = min_val
        max_val_pos = max_val

    if (
        mapping_type == MappingType.SYMMETRIC.name
        or mapping_type == MappingType.SYMMETRIC_NO_CLIPPING_ERR.name
    ):
        # scales
        if mapping_type == MappingType.SYMMETRIC.name: ### only handle this part for now
            ## change this calculation here
            # import pdb; pdb.set_trace()
            max_val_pos = torch.max(-min_val_neg, max_val_pos)
            # scale = max_val_pos / (float(quant_max - quant_min) / 2)
            scale_power = torch.floor(torch.log2(max_val_pos)) - _DTYPE_TO_EMAX[representation_dtype]
            scale = 2 ** torch.clamp(scale_power, -127, 128)
            # import pdb; pdb.set_trace()
        else:
            assert mapping_type == MappingType.SYMMETRIC_NO_CLIPPING_ERR.name
            # calculate smin and smax individually and choose the larger one. For example, if quant_min = -8 and
            # quant_max = 7.
            # - If smin is bigger: There would be coverage on negative values down to -8, and less rounding
            # error than the existing SYMMETRIC case.
            # - If smax is bigger: it covers the positive values up to 7. The round
            # error may be bigger than the existing SYMMETRIC case. Either way, there's no out-of-range fp values after
            # quantization.
            smin = min_val_neg / float(quant_min)
            smax = max_val_pos / float(quant_max)
            mask = smin > smax
            scale = torch.where(mask, smin, smax)
        # zeros
        if not preserve_zero:
            raise ValueError(
                "preserve_zero == False is not supported for symmetric quantization"
            )
        if zero_point_domain == ZeroPointDomain.FLOAT.name:
            # TODO INT should not be a valid ZeroPointDomain for symmetric quantization since
            # symmetric quant doesn't have a zero_point
            raise ValueError(
                "zero_point_domain should be ZeroPointDomain.INT or ZeroPointDomain.NONE for symmetric quantization"
            )
        if zero_point_domain == ZeroPointDomain.NONE.name:
            zero_point = None
        else:
            zero_point = torch.full_like(scale, int((quant_max + quant_min + 1) / 2))
        scale = torch.clamp(scale, min=eps)
    else:
        assert mapping_type == MappingType.ASYMMETRIC.name
        scale = (max_val_pos - min_val_neg) / float(quant_max - quant_min)
        scale = torch.clamp(scale, min=eps)
        if zero_point_domain == ZeroPointDomain.NONE.name:
            zero_point = None
        elif zero_point_domain == ZeroPointDomain.INT.name:
            zero_point = quant_min - torch.round(min_val_neg / scale)
            zero_point = torch.clamp(zero_point, quant_min, quant_max)
            if zero_point_dtype is None:
                zero_point_dtype = torch.int32
        else:
            assert zero_point_domain == ZeroPointDomain.FLOAT.name, (
                "zero_point must be in FLOAT/INT/None domain for asymmetric quantization"
            )
            mid_point = (quant_max + quant_min + 1) / 2
            # this is not preserving zero_point, this is converting to TensorCoreTiledFormat
            # TODO move the conversion of zero_point out of quant_primitives
            # and into TensorCoreTiledLayout.from_plain
            zero_point = min_val_neg + scale * mid_point

    if zero_point is not None:
        zero_point = zero_point.to(dtype=zero_point_dtype)
    return scale.to(dtype=scale_dtype), zero_point



@torch.no_grad()
def choose_qparams_affine_float(
    input: torch.Tensor,
    mapping_type: MappingType,
    block_size: Tuple[int, ...],
    target_dtype: torch.dtype,
    quant_min: Optional[Union[int, float]] = None,
    quant_max: Optional[Union[int, float]] = None,
    eps: Optional[float] = None,
    scale_dtype: Optional[torch.dtype] = None,
    zero_point_dtype: Optional[torch.dtype] = None,
    preserve_zero: bool = True,
    zero_point_domain: ZeroPointDomain = ZeroPointDomain.INT,
    representation_dtype: TorchAODTypeFloat = TorchAODTypeFloat.FLOAT4_E2M1
) -> Tuple[torch.Tensor, torch.Tensor]:
    """
    Args:
        input (torch.Tensor): fp32, bf16, fp16 input Tensor
        mapping_type (MappingType): determines how the qparams are calculated, symmetric or asymmetric
        block_size: (Tuple[int, ...]): granularity of quantization, this means the size of the tensor elements that's sharing the same qparam
          e.g. when size is the same as the input tensor dimension, we are using per tensor quantization
        target_dtype (torch.dtype): dtype for target quantized Tensor
        quant_min (Optional[int]): minimum quantized value for target quantized Tensor
        quant_max (Optioanl[int]): maximum quantized value for target quantized Tensor
        eps (Optional[float]): minimum scale, if not provided, default to eps of input.dtype
        scale_dtype (torch.dtype): dtype for scale Tensor
        zero_point_dtype (torch.dtype): dtype for zero_point Tensor
        preserve_zero (bool): a flag to indicate whether we need zero to be exactly
          representable or not, this is typically required for ops that needs zero padding, like convolution
          it's less important for ops that doesn't have zero padding in the op itself, like linear.

          For example, given a floating point Tensor [1.2, 0.1, 3.0, 4.0, 0.4, 0], if `preserve_zero` is True,
          we'll make sure there is a integer value corresponding to the floating point 0, e.g. [-3, -8, 3, 7, -7, -8], 0 will be mapped to `-8` without loss. But if `preserve_zero` is not True, there won't be such
          gurantee.

          If we don't need zero to be exactly representable, we won't do rounding and clamping for zero_point

        zero_point_domain (ZeroPointDomain): the domain that zero_point is in, should be either integer or float
            if zero_point is in integer domain, zero point is added to the quantized integer value during
            quantization
            if zero_point is in floating point domain, zero point is subtracted from the floating point (unquantized)
            value during quantization
            default is ZeroPointDomain.INT

    Output:
        Tuple of scales and zero_points Tensor with requested dtype
    """
    if zero_point_domain is None:
        raise ValueError("Please use ZeroPointDomain.NONE instead of None")

    return _choose_qparams_affine_float( ## need to customize this!
        input,
        mapping_type.name,
        block_size,
        target_dtype,
        quant_min,
        quant_max,
        eps,
        scale_dtype,
        zero_point_dtype,
        preserve_zero,
        zero_point_domain.name,
        representation_dtype=representation_dtype
    )


def is_even(x):
    # Assuming is_even checks if the mantissa in FP4 E2M1 is even
    # In FP4 E2M1, "even" mantissa corresponds to values like -6.0, -4.0, -2.0, -1.0, -0.5, 0, 0.5, 1.0, 2.0, 4.0, 6.0
    even_values = torch.tensor([-6.0, -4.0, -2.0, -1.0, -0.5, 0.0, 0.5, 1.0, 2.0, 4.0, 6.0], device=x.device)
    return torch.isin(x, even_values)


# def float_to_e2m1(y):
#     if y == 0:
#         return 0.0
#     R = [-6.0, -4.0, -3.0, -2.0, -1.5, -1.0, -0.5, 0, 0.5, 1.0, 1.5, 2.0, 3.0, 4.0, 6.0]
#     min_dist = min(abs(r - y) for r in R)
#     candidates = [r for r in R if abs(r - y) == min_dist]
#     even_candidates = [r for r in candidates if is_even(r)]
#     chosen_r = even_candidates[0] if even_candidates else candidates[0]
#     return chosen_r



def float_to_e2m1(tensor):
    # Define FP4 E2M1 values
    R = torch.tensor([-6.0, -4.0, -3.0, -2.0, -1.5, -1.0, -0.5, 0.0, 0.5, 1.0, 1.5, 2.0, 3.0, 4.0, 6.0], 
                     device=tensor.device, dtype=tensor.dtype)
    

    original_shape = tensor.shape
    tensor_flat = tensor.flatten() 
    
    # Broadcast: compute |tensor[i] - R[j]| for all i, j
    # tensor_flat: (N,), R: (15,) -> dists: (N, 15)
    dists = torch.abs(tensor_flat.unsqueeze(-1) - R)
    
    # Find indices of minimum distances
    min_dist_indices = torch.argmin(dists, dim=-1)  # Shape: (N,)
    
    # Initialize output with closest values
    result = R[min_dist_indices]  # Shape: (N,)
    
    # Handle ties: check if multiple R values have the same minimum distance
    min_dists = dists.gather(1, min_dist_indices.unsqueeze(-1)).squeeze(-1)  # Shape: (N,)
    tie_mask = (dists == min_dists.unsqueeze(-1)).sum(dim=-1) > 1  # Shape: (N,)
    
    if tie_mask.any():
        # For elements with ties, find all candidates
        tie_indices = torch.where(tie_mask)[0]
        for idx in tie_indices:
            candidates = R[dists[idx] == min_dists[idx]]
            even_candidates = candidates[is_even(candidates)]
            result[idx] = even_candidates[0] if even_candidates.numel() > 0 else candidates[0]
    
    # Reshape back to original shape
    return result.reshape(original_shape)





# Example usage
# weight_matrix = torch.randn(256, 1536, device='cuda' if torch.cuda.is_available() else 'cpu')
# quantized_matrix = vectorized_float_to_e2m1(weight_matrix)



# def _do_fake_quantize_float_affine(
#     input: torch.Tensor,
#     block_size: Tuple[int, ...],
#     scale: torch.Tensor,
#     zero_point: Optional[torch.Tensor],
#     quant_dtype: torch.dtype,
#     quant_min: Optional[Union[int, float]] = None,
#     quant_max: Optional[Union[int, float]] = None,
#     zero_point_domain: ZeroPointDomain = ZeroPointDomain.INT,
# ) -> Tuple[torch.Tensor, torch.Tensor]:
#     """
#     Helper function for `fake_quantize_affine` that returns both the
#     intermediate quantized values and the final dequantized values.
#     """
#     input_dtype = input.dtype
#     quant_min, quant_max = _get_and_check_qmin_qmax(quant_dtype, quant_min, quant_max)
#     q = _quantize_affine_no_dtype_cast(
#         input,
#         block_size,
#         scale,
#         zero_point,
#         quant_min,
#         quant_max,
#         quant_dtype,
#         zero_point_domain.name,
#     )

#     print(f"WEIGHT MATRIX SHAPE: {input.shape}")

#     ### TODO: after quantizing map the values to mxfp4_e2m1 ranges (close match) here
#     dq = _dequantize_affine_no_dtype_check(
#         q,
#         block_size,
#         scale,
#         zero_point,
#         quant_min,
#         quant_max,
#         zero_point_domain.name,
#         output_dtype=input_dtype,
#     )
#     return (q, dq)

def _quantize_affine_no_dtype_cast(
    input: torch.Tensor,
    block_size: List[int],
    scale: torch.Tensor,
    zero_point: Optional[torch.Tensor],
    quant_min: Union[int, float],
    quant_max: Union[int, float],
) -> torch.Tensor:
    """
    The op does the following:
    1. figure out the dimension for reduction based on block_size, also reshape the input to align with
       the shape after reduction
    2. quantize the input based on the quantization parameters scale and zero_point and zero_point_domain = INT
    3. reshape the quantized result to origianl shape
    """
    # TODO: validations
    # TODO: validate scale/zero_point dimensions are compatible with block_size
    assert input.dtype in [
        torch.float32,
        torch.float16,
        torch.bfloat16,
    ], f"Unsupported input dtype: {input.dtype}"
    assert len(block_size) == input.dim(), (
        f"Got input dim:{input.dim()}, block_size: {block_size}"
    )
    shape_for_reduction, reduction_dims = _get_reduction_params(
        block_size, input.size()
    )
    original_shape = input.shape
    input = input.view(shape_for_reduction)
    shape_after_reduction = shape_for_reduction
    for i in reduction_dims:
        shape_after_reduction[i] = 1
    scale = scale.view(shape_after_reduction)

    if zero_point is not None and zero_point.numel() > 0:
        zero_point = zero_point.view(shape_after_reduction)
    else:
        # in some cases zero_point being a non-value shows as a tensor
        # with numel=0 which we handle by unifying the two
        zero_point = None

    quant = torch.clamp(
        input * (1.0 / scale) + zero_point, quant_min, quant_max
    )
    quant = quant.view(original_shape)

    return quant


def _do_fake_quantize_float_affine(
    input: torch.Tensor,
    block_size: Tuple[int, ...],
    scale: torch.Tensor,
    zero_point: Optional[torch.Tensor],
    quant_dtype: torch.dtype,
    quant_min: Optional[Union[int, float]] = None,
    quant_max: Optional[Union[int, float]] = None,
    zero_point_domain: ZeroPointDomain = ZeroPointDomain.FLOAT,
    representation_dtype: TorchAODTypeFloat = TorchAODTypeFloat.FLOAT4_E2M1
) -> Tuple[torch.Tensor, torch.Tensor]:
    """
    Helper function for `fake_quantize_affine` that returns both the
    intermediate quantized values and the final dequantized values.
    """
    input_dtype = input.dtype
    quant_min, quant_max = _get_and_check_qmin_qmax(quant_dtype, quant_min, quant_max)
    if zero_point_domain == ZeroPointDomain.INT:
        _quantize_affine = _quantize_affine_no_dtype_cast
        _dequantize_affine = _dequantize_affine_no_dtype_check
    elif zero_point_domain == ZeroPointDomain.FLOAT:
        _quantize_affine = _quantize_affine_tinygemm_no_dtype_cast
        _dequantize_affine = _dequantize_affine_tinygemm_no_dtype_check
    elif ZeroPointDomain == ZeroPointDomain.NONE:
        _quantize_affine = _quantize_affine_no_zero_point_no_dtype_cast
        _dequantize_affine = _dequantize_affine_no_zero_point_no_dtype_check
    else:
        raise ValueError(f"Unrecognized zero point domain: {zero_point_domain}")
    

    print(f"========================== OG INPUT ==========================\n{input}")
    print(input)
    torch.save(input, "/shareddata/dheyo/shivanvitha/torchtune/dummy_og1.pt")
    q = _quantize_affine(
        input,
        block_size,
        scale,
        zero_point,
        quant_min,
        quant_max,
    )

    ### TODO: after quantizing map the values to mxfp4_e2m1 ranges (close match) here
    # print(f"WEIGHT MATRIX SHAPE: {input.shape}") ## Its a 2D normal weight matrix!!!
    # mapped_q = float_to_e2m1(q)
    abs_clamped = torch.abs(q)
    binade_exp = torch.floor(torch.log2(torch.where(abs_clamped == 0, 1.0, abs_clamped)))
    binade_exp = torch.clamp(binade_exp, - (_DTYPE_TO_BIAS[representation_dtype] + 1), _DTYPE_TO_EMAX[representation_dtype])  # Valid exponents
    binade = torch.pow(2.0, binade_exp)
    quant_step = binade * (2 ** (-_DYPE_TO_MANTISSA_BITS[representation_dtype]))
    mapped_q = torch.round(q / quant_step) * quant_step

    print(f"========================== Mapped Q ==========================\n{mapped_q}")
    print(f"MAPPED Q RANGE for {representation_dtype}: {(torch.min(mapped_q), torch.max(mapped_q))}")

    dq = _dequantize_affine(
        mapped_q,
        block_size,
        scale,
        zero_point,
        quant_min,
        quant_max,
        output_dtype=input_dtype,
    )
    print(f"========================== DeQuant ==========================\n{dq}")
    print(dq)
    torch.save(dq, "/shareddata/dheyo/shivanvitha/torchtune/dummy_after1.pt")
    print(f"FP4's zero point domain: {zero_point_domain} - {zero_point}")
    import pdb; pdb.set_trace()
    return (q, dq)


def fake_quantize_float_affine_cachemask(
    input: torch.Tensor,
    block_size: Tuple[int, ...],
    scale: torch.Tensor,
    zero_point: Optional[torch.Tensor],
    quant_dtype: torch.dtype,
    quant_min: Optional[Union[int, float]] = None,
    quant_max: Optional[Union[int, float]] = None,
    zero_point_domain: ZeroPointDomain = ZeroPointDomain.FLOAT,
    representation_dtype: TorchAODTypeFloat = TorchAODTypeFloat.FLOAT4_E2M1
) -> Tuple[torch.Tensor, torch.Tensor]:
    """
    General fake quantize op for quantization-aware training (QAT).
    This is equivalent to calling `quantize_affine` + `dequantize_affine`
    but without the dtype casts.

    Note: Compared to :func:`~torchao.quantization.quant_primitives.fake_quantize_affine`,
    this consumes more memory and returns an additional outlier mask for
    intermediate quantized values.

    Args:
      Same as :func:`~torchao.quantization.quant_primitives.fake_quantize_affine`.

    Returns:
      A 2-tuple of (
          final fake quantized values,
          outlier mask for intermediate quantized values
      )

    """
    if zero_point_domain is None:
        raise ValueError("Please use ZeroPointDomain.NONE instead of None")
    elif zero_point_domain is None and zero_point is not None:
        raise ValueError("zero_point should be None when zero_point_domain is NONE")
    (q, dq) = _do_fake_quantize_float_affine(
        input,
        block_size,
        scale,
        zero_point,
        quant_dtype,
        quant_min,
        quant_max,
        zero_point_domain,
        representation_dtype
    )
    mask = torch.logical_and((q >= quant_min), (q <= quant_max))
    return (dq, mask)