from torchao.quantization.qat.api import FakeQuantizeConfig
from torchao.quantization.qat.fake_quantizer import FakeQuantizer
from torchao.quantization.qat.linear import FakeQuantizedLinear, _LegacyQATQuantizer, Int8DynActInt4WeightQATLinear
from typing import Any, Optional
from torchtune.utils.dheyo_quant_primitives import TorchAODTypeFloat
from torchtune.training.dheyo_fake_quantizer import FakeQuantizerWrapper

class FakeQuantizedLinearWrapper(FakeQuantizedLinear):
    """
    General linear layer with fake quantized weights and activations.

    Specific target dtypes, granularity, schemes etc. are specified
    through separate configs for weights and activations.

    Example usage::

        activation_config = FakeQuantizeConfig(
            dtype=torch.int8,
            granularity="per_token",
            is_symmetric=False,
        )
        weight_config = FakeQuantizeConfig(
            dtype=torch.int4,
            group_size=8,
            is_symmetric=True,
        )
        fq_linear = FakeQuantizedLinear(
            16, 32, False, activation_config, weight_config,
        )
        fq_linear(torch.randn(16))
    """

    def __init__(
        self,
        in_features: int,
        out_features: int,
        bias: bool = False,
        activation_config: Optional[FakeQuantizeConfig] = None,
        weight_config: Optional[FakeQuantizeConfig] = None,
        *args,
        **kwargs,
    ) -> None:
        super().__init__(
            in_features,
            out_features,
            bias,
            *args,
            **kwargs,
        )
        # initialize activation fake quantizer
        if activation_config is not None:
            self.activation_fake_quantizer = FakeQuantizer(activation_config)
        else:
            self.activation_fake_quantizer = None

        # initialize weight fake quantizer
        if weight_config is not None:
            group_size = weight_config.group_size
            if group_size is not None and in_features % group_size != 0:
                raise ValueError(
                    "in_features (%s) %% group_size (%s) must be == 0"
                    % (in_features, group_size)
                )
            
            if isinstance(weight_config.dtype, TorchAODTypeFloat):
                self.weight_fake_quantizer = FakeQuantizerWrapper(weight_config)
            else:
                self.weight_fake_quantizer = FakeQuantizer(weight_config)
        else:
            self.weight_fake_quantizer = None