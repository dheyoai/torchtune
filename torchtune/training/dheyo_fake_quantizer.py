from torchao.quantization.qat.fake_quantizer import FakeQuantizer
import torch
from torchao.quantization.granularity import (
    PerAxis,
    PerGroup,
    PerToken,
)
from torchao.quantization.utils import (
    _get_per_token_block_size,
    get_group_qparams_symmetric,
    get_groupwise_affine_qparams,
)

from torchao.quantization.qat.fake_quantizer.utils import (
    _fake_quantize_per_channel_group,
    _fake_quantize_per_token,
)

from torchtune.utils.dheyo_quant_primitives import (
    _DTYPE_TO_BIT_WIDTH,
    _DTYPE_TO_QVALUE_BOUNDS,
    # MappingType,
    # choose_qparams_affine,
)

from torchao.quantization.quant_primitives import (
    MappingType,
    choose_qparams_affine,
)


class FakeQuantizerWrapper(FakeQuantizer):
    def _per_channel_or_group_forward(self, x: torch.Tensor):
        """
        Perform per channel or per group fake quantization on the tensor.
        We express per channel using per group where the group size is the size
        of the last dimension of the tensor.
        """
        granularity = self.config.granularity
        scale_precision = self.config.scale_precision
        zero_point_precision = self.config.zero_point_precision
        zero_point_domain = self.config.zero_point_domain
        is_symmetric = self.config.is_symmetric

        # get group size
        if isinstance(granularity, PerAxis):
            assert granularity.axis == 0
            group_size = x.size()[-1]
        elif isinstance(granularity, PerGroup):
            group_size = granularity.group_size
        else:
            raise ValueError("Unexpected granularity '%s'" % granularity)

        # get scales and zero points
        # TODO: refactor this to use `choose_qparams_affine`
        if self._should_compute_qparams():
            bit_width = _DTYPE_TO_BIT_WIDTH[self.config.dtype]
            if is_symmetric:
                (self.scale, self.zero_point) = get_group_qparams_symmetric(
                    x,
                    bit_width,
                    group_size,
                    scale_precision,
                )
            else:
                (self.scale, self.zero_point) = get_groupwise_affine_qparams(
                    x,
                    bit_width,
                    group_size,
                    scale_precision,
                )
            self.zero_point = self.zero_point.to(zero_point_precision)

        qmin, qmax = _DTYPE_TO_QVALUE_BOUNDS[self.config.dtype]
        return _fake_quantize_per_channel_group(
            x,
            self.scale,
            self.zero_point,
            qmin,
            qmax,
            group_size,
            zero_point_domain,
        )