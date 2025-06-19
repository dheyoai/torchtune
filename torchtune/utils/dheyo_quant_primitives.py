from enum import Enum, auto
import torch
from typing import Dict, Union, Tuple, Optional, List
from torchao.quantization.quant_primitives import (
    MappingType, 
    ZeroPointDomain, 
    _quantize_affine_no_dtype_cast, 
    _dequantize_affine_no_dtype_check,
    _quantize_affine_no_dtype_cast,
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


class TorchAODTypeFloat(Enum):
    """
    Placeholder for float dtypes that do not exist in PyTorch core yet.
    """
    FLOAT4_E2M1 = 4


_DTYPE_TO_QVALUE_BOUNDS: Dict[Union[torch.dtype, TorchAODTypeFloat], Tuple[float, float]] = {
    TorchAODTypeFloat.FLOAT4_E2M1: (-6.0, 6.0), ## TODO: use the EM formula and expand this later
    torch.bfloat16: (-3.40e38, 3.40e38)


}
_DTYPE_TO_BIT_WIDTH: Dict[Union[torch.dtype, TorchAODTypeFloat], Tuple[int, int]] = {
    TorchAODTypeFloat.FLOAT4_E2M1: 4,
}

_DTYPE_TO_EMAX: Dict[Union[torch.dtype, TorchAODTypeFloat], Tuple[float, float]] = {
    TorchAODTypeFloat.FLOAT4_E2M1: 2**(2 - 1),
}

_SUB_BYTE_UINT_BOUNDS: Dict[Union[torch.dtype, TorchAODTypeFloat], Tuple[float, float]] = {}
_SUB_BYTE_INT_BOUNDS: Dict[Union[torch.dtype, TorchAODTypeFloat], Tuple[float, float]] = {
    TorchAODTypeFloat.FLOAT4_E2M1: (-6.0, 6.0) ## TODO: use the EM formula and expand this later
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


@register_custom_op
def _choose_qparams_affine_float(
    input: Optional[torch.Tensor],
    mapping_type: str,
    block_size: List[int],
    target_dtype: torch.dtype,
    quant_min: Optional[Union[int, float, bool]] = None,
    quant_max: Optional[Union[int, float, bool]] = None,
    eps: Optional[float] = None,
    scale_dtype: Optional[torch.dtype] = None,
    zero_point_dtype: Optional[torch.dtype] = None,
    preserve_zero: bool = True,
    zero_point_domain: Optional[str] = "FLOAT",
    min_val: Optional[torch.Tensor] = None,
    max_val: Optional[torch.Tensor] = None,
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
            # import pdb; pdb.set_trace()
            max_val_pos = torch.max(-min_val_neg, max_val_pos)
            # scale = max_val_pos / (float(quant_max - quant_min) / 2)
            scale_power = torch.floor(torch.log2(max_val_pos)) - _DTYPE_TO_EMAX[TorchAODTypeFloat.FLOAT4_E2M1]
            scale = 2 ** scale_power
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
    )


def is_even(r):
    even_values = [-4.0, -2.0, -1.0, 0, 1.0, 2.0, 4.0]
    return r in even_values


def float_to_e2m1(y):
    if y == 0:
        return 0.0
    R = [-6.0, -4.0, -3.0, -2.0, -1.5, -1.0, -0.5, 0, 0.5, 1.0, 1.5, 2.0, 3.0, 4.0, 6.0]
    min_dist = min(abs(r - y) for r in R)
    candidates = [r for r in R if abs(r - y) == min_dist]
    even_candidates = [r for r in candidates if is_even(r)]
    chosen_r = even_candidates[0] if even_candidates else candidates[0]
    return chosen_r



def vectorized_float_to_e2m1(tensor):
    # Define the sets as PyTorch tensors
    R = torch.tensor([-6.0, -4.0, -3.0, -2.0, -1.5, -1.0, -0.5, 0.0, 0.5, 1.0, 1.5, 2.0, 3.0, 4.0, 6.0], device=tensor.device)
    even_values = torch.tensor([-4.0, -2.0, -1.0, 0.0, 1.0, 2.0, 4.0], device=tensor.device)
    
    # Initialize result tensor
    result = torch.zeros_like(tensor)
    
    # Handle non-zero elements
    mask_nonzero = tensor != 0
    if mask_nonzero.any():
        # Compute absolute differences: shape (256, 1536, len(R))
        tensor_nonzero = tensor[mask_nonzero]
        diffs = torch.abs(tensor_nonzero[:, None] - R)
        
        # Find minimum distance and indices: shape (num_nonzero,)
        min_dist, min_indices = torch.min(diffs, dim=1)
        
        # Create mask for candidates (where distance equals min_dist)
        candidates_mask = diffs == min_dist[:, None]
        
        # Get all candidate indices per element
        candidate_indices = candidates_mask.nonzero(as_tuple=True)[1]
        element_indices = candidates_mask.nonzero(as_tuple=True)[0]
        
        # Initialize chosen indices with the first candidate
        chosen_indices = torch.zeros_like(min_indices)
        first_candidate = torch.zeros_like(min_indices)
        
        # Map element indices to their first candidate
        for i in range(len(element_indices)):
            if first_candidate[element_indices[i]] == 0:
                first_candidate[element_indices[i]] = 1
                chosen_indices[element_indices[i]] = candidate_indices[i]
        
        # Check for even candidates
        even_mask = torch.isin(R, even_values)
        even_candidates = candidates_mask & even_mask[None, :]
        
        # Update chosen indices if an even candidate exists
        even_candidate_indices = even_candidates.nonzero(as_tuple=True)[1]
        even_element_indices = even_candidates.nonzero(as_tuple=True)[0]
        
        for i in range(len(even_element_indices)):
            chosen_indices[even_element_indices[i]] = even_candidate_indices[i]
        
        # Map chosen indices to R values
        result[mask_nonzero] = R[chosen_indices]
    
    return result

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



def _do_fake_quantize_float_affine(
    input: torch.Tensor,
    block_size: Tuple[int, ...],
    scale: torch.Tensor,
    zero_point: Optional[torch.Tensor],
    quant_dtype: torch.dtype,
    quant_min: Optional[Union[int, float]] = None,
    quant_max: Optional[Union[int, float]] = None,
    zero_point_domain: ZeroPointDomain = ZeroPointDomain.INT,
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
    mapped_q = vectorized_float_to_e2m1(q)
    print(f"========================== Mapped Q ==========================\n{mapped_q}")
    # print(f"{scale} and {zero_point}")

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
    torch.save(dq, "/shareddata/dheyo/shivanvitha/torchtune/dummy_after1.pt")
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
    )
    mask = torch.logical_and((q >= quant_min), (q <= quant_max))
    return (dq, mask)