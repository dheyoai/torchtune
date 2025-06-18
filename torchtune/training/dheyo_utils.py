import torch
from torchao.quantization.quant_primitives import MappingType

    
from torchtune.utils.dheyo_quant_primitives import choose_qparams_affine

def get_group_qparams_symmetric(
    w,
    n_bit=4,
    groupsize=32,
    precision=torch.float32,
    mapping_type=MappingType.SYMMETRIC,
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
    ranges = {}
    ranges[1] = (-1, 0)
    # generating ranges for bit 2 to 8
    for i in range(4, 5): ## only adding support for FLOAT4_E2M1
        ranges[i] = (-6.0, 6.0) ## TODO: Expand the range with proper EM formula
    quant_min, quant_max = ranges[n_bit]
    scale, zero_point = choose_qparams_affine(
        w,
        mapping_type,
        block_size,
        target_dtype=torch.bfloat16,
        quant_min=quant_min,
        quant_max=quant_max,
        eps=eps,
        scale_dtype=precision,
        zero_point_dtype=precision,
    )
    return scale.reshape(w.shape[0], -1), zero_point.reshape(w.shape[0], -1)