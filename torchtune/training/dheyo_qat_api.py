from torchao.quantization.qat.api import FakeQuantizeConfig
from torchtune.utils.dheyo_quant_primitives import TorchAODTypeFloat, _SUB_BYTE_FLOAT_BOUNDS, _SUB_BYTE_UINT_BOUNDS
import torch 
from typing import Union, Optional
from torchao.quantization.quant_primitives import MappingType, ZeroPointDomain
from torchao.quantization.granularity import (
    Granularity,
    PerAxis,
    PerGroup,
    PerToken,
)
from dataclasses import dataclass



@dataclass
class FakeQuantizeConfigWrapper(FakeQuantizeConfig):
    """
    Config for how to fake quantize weights or activations.

    args:
        dtype: dtype to simulate during fake quantization, e.g. torch.int8.
            For PyTorch versions older than 2.6, you may use `TorchAODType` to represent
            torch.int1 to torch.int7 instead, e.g. TorchAODType.INT4.
        granularity: granularity of scales and zero points, e.g. PerGroup(32).
            We also support the following strings:
               1) 'per_token': equivalent to PerToken()
               2) 'per_channel': equivalent to PerAxis(0)
               3) 'per_group': equivalent to PerGroup(group_size), must be combined
                   with separate `group_size` kwarg, Alternatively, just set the
                   `group_size` kwarg and leave this field empty.
        mapping_type: whether to use symmetric (default) or asymmetric quantization
            Alternatively, set `is_symmetric` (bool) and leave this field empty.
        scale_precision: scale dtype (default torch.fp32)
        zero_point_precision: zero point dtype (default torch.int32)
        zero_point_domain: whether zero point is in integer (default) or float domain
        is_dynamic: whether to use dynamic (default) or static scale and zero points
        range_learning: whether to learn scale and zero points during training (coming soon)

    kwargs (optional):
        group_size: size of each group in per group fake quantization,
            can be set instead of `granularity`
        is_symmetric: whether to use symmetric or asymmetric quantization,
            can be set instead of `mapping_type`

    Example usage::

        # Per token asymmetric quantization
        FakeQuantizeConfig(torch.int8, "per_token", is_symmetric=False)
        FakeQuantizeConfig(torch.int8, PerToken(), MappingType.ASYMMETRIC)

        # Per channel symmetric quantization
        FakeQuantizeConfig(torch.int4, "per_channel")
        FakeQuantizeConfig(torch.int4, "per_channel", is_symmetric=True)
        FakeQuantizeConfig(torch.int4, PerAxis(0), MappingType.SYMMETRIC)

        # Per group symmetric quantization
        FakeQuantizeConfig(torch.int4, group_size=32)
        FakeQuantizeConfig(torch.int4, group_size=32, is_symmetric=True)
        FakeQuantizeConfig(torch.int4, "per_group", group_size=32, is_symmetric=True)
        FakeQuantizeConfig(torch.int4, PerGroup(32), MappingType.SYMMETRIC)
    """

    dtype: Union[torch.dtype, TorchAODTypeFloat]
    granularity: Granularity
    mapping_type: MappingType
    scale_precision: torch.dtype
    zero_point_precision: torch.dtype
    zero_point_domain: ZeroPointDomain
    is_dynamic: bool = True
    range_learning: bool = False

    def __init__(
        self,
        dtype: Union[torch.dtype, TorchAODTypeFloat],
        granularity: Union[Granularity, str, None] = None,
        mapping_type: Optional[MappingType] = None,
        scale_precision: torch.dtype = torch.float32,
        zero_point_precision: torch.dtype = torch.int32,
        zero_point_domain: ZeroPointDomain = ZeroPointDomain.INT,
        is_dynamic: bool = True,
        range_learning: bool = False,
        *,
        group_size: Optional[int] = None,
        is_symmetric: Optional[bool] = None,
    ):
        if zero_point_domain is None:
            raise ValueError("Please use ZeroPointDomain.NONE instead of None")
        self.dtype = dtype
        self.granularity = self._get_granularity(granularity, group_size)
        self.mapping_type = self._get_mapping_type(mapping_type, is_symmetric)
        self.scale_precision = scale_precision
        self.zero_point_precision = zero_point_precision
        self.zero_point_domain = zero_point_domain
        self.is_dynamic = is_dynamic
        self.range_learning = range_learning

        # Validate dtype
        all_dtypes = [torch.int8, torch.uint8]
        all_dtypes.extend(list(_SUB_BYTE_FLOAT_BOUNDS.keys()))
        all_dtypes.extend(list(_SUB_BYTE_UINT_BOUNDS.keys()))
        if dtype not in all_dtypes:
            raise ValueError(
                "Unsupported dtype '%s', choose from %s" % (dtype, all_dtypes)
            )