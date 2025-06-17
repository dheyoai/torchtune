from typing import Any, Optional
import pdb
import torch
import torch.nn.functional as F

from torchao.dtypes.utils import is_device
from torchao.quantization.GPTQ import (
    Int8DynActInt4WeightLinear,
    WeightOnlyInt4Linear,
    # _check_linear_int4_k,
    _replace_linear_8da4w,
    _replace_linear_int4,
    # groupwise_affine_quantize_tensor,
)
from torchao.quantization.quant_primitives import (
    TorchAODType,
    ZeroPointDomain,
)
from torchao.quantization.unified import TwoStepQuantizer
from torchao.quantization.utils import get_group_qparams_symmetric
from torchao.utils import TORCH_VERSION_AT_LEAST_2_6

from torchao.quantization.qat.api import FakeQuantizeConfig
from torchao.quantization.qat.fake_quantizer import FakeQuantizer
from torchao.quantization.qat.utils import (
    _get_qmin_qmax,
)

from torchao.quantization.qat.linear import FakeQuantizedLinear, _LegacyQATQuantizer, Int8DynActInt4WeightQATLinear
from torchao.quantization.quant_primitives import (
    MappingType,
    dequantize_affine,
)

from torchao.quantization.utils import (
    _MultiInput,
    get_group_qparams_symmetric,
    get_groupwise_affine_qparams,
    group_quantize_tensor_symmetric,
    groupwise_affine_dequantize_tensor_from_qparams,
    groupwise_affine_quantize_tensor,
    groupwise_affine_quantize_tensor_from_qparams,
    pack_tinygemm_scales_and_zeros,
    per_token_dynamic_quant,
)

from typing import Any, Callable, Dict, List, Optional, Type, Tuple
import torch.nn as nn
from torchao.float8.float8_linear import Float8Linear
import json


# =========================================================
# |   Linear int8 dynamic activations + int6 weight QAT   |
# =========================================================
bit_map = {
    'Q2_K': {'bits': 2, 'group_size': 256},
    'Q3_K': {'bits': 2, 'group_size': 256},
    'Q3_K_M': {'bits': 2, 'group_size': 256},
    'Q3_K_S': {'bits': 2, 'group_size': 256},
    'Q3_K_L': {'bits': 2, 'group_size': 256},
    'Q4_0': {'bits': 2, 'group_size': 32},
    'Q4_1': {'bits': 2, 'group_size': 32},
    'Q4_K': {'bits': 2, 'group_size': 256},
    'Q4_K_M': {'bits': 2, 'group_size': 256},
    'Q4_K_S': {'bits': 2, 'group_size': 256},
    'Q5_0': {'bits': 2, 'group_size': 32},
    'Q5_1': {'bits': 2, 'group_size': 32},
    'Q5_K': {'bits': 2, 'group_size': 256},
    'Q5_K_M': {'bits': 2, 'group_size': 256},
    'Q5_K_S': {'bits': 2, 'group_size': 256},
    'Q6_K': {'bits': 2, 'group_size': 256},
    'Q8_0': {'bits': 2, 'group_size': 32},
    'Q8_K': {'bits': 2, 'group_size': 256},
    'Q8_1': {'bits': 2, 'group_size': 32},
    'F16': {'bits': 16, 'group_size': 256},
    'F32': {'bits': 32, 'group_size': 256}
}


def _check_linear_int4_k(k, group_size=1, inner_k_tiles=None):
    """
    Check if the dimensions are compatible with int4 quantization.

    Args:
        k: The dimension size to check
        group_size: The group size for quantization
        inner_k_tiles: The inner k tiles size

    Returns:
        bool: Whether the dimensions are compatible
    """
    k_divisible_by_group_size = k % group_size == 0
    if inner_k_tiles is not None:
        k_divisible_by_16_times_inner_k_tiles = k % (inner_k_tiles * 16) == 0
        return k_divisible_by_group_size and k_divisible_by_16_times_inner_k_tiles
    return k_divisible_by_group_size

def linear_forward_8davarw(
    x,
    weight_int8,
    bias,
    scales,
    zeros,
    out_features,
    groupsize,
    output_precision,
    bits
):
    # uses fp32 to match torchao.quantization.quant_api._int8_asymm_per_token_quant
    # and activation_scale_dtype in QAT configs
    # TODO: in future add ability to specify activation_scale_dtype to PTQ configs
    # and enable similar change here
    x = per_token_dynamic_quant(
        x, scale_dtype=torch.float32, zero_point_dtype=torch.float32
    )

    # TODO: verify and remove following reshape code
    # origin_x_size = x.size()
    # x = x.reshape(-1, origin_x_size[-1])

    # TODO: better API
    # weight_int8 = torch.ops.quantized_decomposed.unpack_int4_to_int8(weight_int4packed)
    n_bit = bits
    quant_min = -(2 ** (n_bit - 1))
    quant_max = 2 ** (n_bit - 1) - 1
    block_size = (1, groupsize)

    w_dq = dequantize_affine(  ### check this once
        weight_int8,
        block_size,
        scales,
        zeros,
        torch.int8,
        quant_min,
        quant_max,
        output_dtype=output_precision,
    )

    # x = x.to(torch.float16)
    # w_dq = w_dq.to(torch.float16)
    c = torch.nn.functional.linear(x, w_dq, bias)

    # new_shape = origin_x_size[:-1] + (out_features,)
    # c = c.reshape(new_shape)

    return c



### int 6 weights linear
class Int8DynActIntVarWeightLinear(torch.nn.Module):
    __constants__ = ["in_features", "out_features"]

    in_features: int
    out_features: int
    weight: torch.Tensor
    bias: torch.Tensor

    """
    This module implements a dynamic quantized linear layer with int4 weight.
    Weights are per channel groupwise quantized. Parameters of importance
    groupsize: the number of elements in each quantized group
    precision: precision of input and output. e.g. torch.float32 means input
    activation is float32 and output is float32.
    scales_precision: precision of per group scale.
    """

    def __init__(
        self,
        in_features: int,
        out_features: int,
        bias=True,
        device=None,
        # TODO: remove this field, not used
        dtype=None,
        groupsize: int = 256,
        precision: torch.dtype = torch.float32,
        scales_precision: torch.dtype = torch.float32,
        bits: int = 4
    ) -> None:
        super().__init__()
        # always pad if needed since it becomes a noop at runtime if not needed
        # self.origin_in_features = in_features
        assert in_features % groupsize == 0, (
            f"require in_features:{in_features} % groupsize:{groupsize} == 0"
        )
        # in_features = _calc_padded_size_linear_int4(
        #    in_features, groupsize
        # )
        self.bits = bits
        self.in_features = in_features
        self.out_features = out_features
        # TODO: align groupsize naming
        self.groupsize = groupsize
        # Precision of the activation which also indicates
        # output precision of the dynamically quantized linear layer
        # that his module represents.
        self.precision = precision

        if dtype is not None:
            raise ValueError("Please specify 'precision' instead of 'dtype'")

        # currently storing unpacked int8 weights
        self.register_buffer(
            "weight",
            torch.zeros((out_features, in_features), dtype=torch.int8),
        )
        self.register_buffer(
            "scales",
            torch.zeros(
                (out_features, in_features // groupsize),
                dtype=scales_precision,
            ),
        )
        self.register_buffer(
            "zeros",
            torch.zeros(
                (out_features, in_features // groupsize),
                dtype=scales_precision,
            ),
        )

        if bias:
            self.register_buffer("bias", torch.zeros(out_features, dtype=precision))
        else:
            self.bias = None

    def forward(self, input: torch.Tensor) -> torch.Tensor:
        input = input.to(self.precision)
        # padding is removed for perf
        # input = F.pad(input, pad=(0, self.in_features - self.origin_in_features))
        return linear_forward_8davarw(
            input,
            self.weight,
            self.bias,
            self.scales,
            self.zeros,
            self.out_features,
            self.groupsize,
            self.precision,
            self.bits
        )




def _get_8davarw_activation_config(qparams_precision: torch.dtype) -> FakeQuantizeConfig:
    """
    Return the activation `FakeQuantizeConfig` for `Int8DynActInt4WeightQATQuantizer`.
    """
    return FakeQuantizeConfig(
        dtype=torch.int8,
        granularity="per_token",
        is_symmetric=False,
        is_dynamic=True,
        scale_precision=qparams_precision,
        zero_point_precision=qparams_precision,
    )


def _get_8davarw_weight_config(
    group_size: int,
    qparams_precision: torch.dtype,
    bits: int
) -> FakeQuantizeConfig:
    """
    Return the weight `FakeQuantizeConfig` for `Int8DynActInt4WeightQATQuantizer`.
    """
    if bits == 2:
        return FakeQuantizeConfig(
            dtype=TorchAODType.INT2,
            group_size=group_size,
            is_symmetric=True,
            is_dynamic=True,
            scale_precision=qparams_precision,
            zero_point_precision=qparams_precision,
        )
    elif bits == 3:
        return FakeQuantizeConfig(
            dtype=TorchAODType.INT3,
            group_size=group_size,
            is_symmetric=True,
            is_dynamic=True,
            scale_precision=qparams_precision,
            zero_point_precision=qparams_precision,
        )
    elif bits == 4:
        return FakeQuantizeConfig(
            dtype=TorchAODType.INT4,
            group_size=group_size,
            is_symmetric=True,
            is_dynamic=True,
            scale_precision=qparams_precision,
            zero_point_precision=qparams_precision,
        )
    elif bits == 5:
        return FakeQuantizeConfig(
            dtype=TorchAODType.INT5,
            group_size=group_size,
            is_symmetric=True,
            is_dynamic=True,
            scale_precision=qparams_precision,
            zero_point_precision=qparams_precision,
        )
    elif bits == 6:
        return FakeQuantizeConfig(
            dtype=TorchAODType.INT6,
            group_size=group_size,
            is_symmetric=True,
            is_dynamic=True,
            scale_precision=qparams_precision,
            zero_point_precision=qparams_precision,
        )
    elif bits == 8:
        return FakeQuantizeConfig(
            dtype=torch.int8,
            group_size=group_size,
            is_symmetric=True,
            is_dynamic=True,
            scale_precision=qparams_precision,
            zero_point_precision=qparams_precision,
        )
    elif bits == 16:
        return FakeQuantizeConfig(
            dtype=torch.float16,
            group_size=group_size,
            is_symmetric=True,
            is_dynamic=True,
            scale_precision=qparams_precision,
            zero_point_precision=qparams_precision,
        )
    elif bits == 32:
        return FakeQuantizeConfig(
            dtype=torch.float32,
            group_size=group_size,
            is_symmetric=True,
            is_dynamic=True,
            scale_precision=qparams_precision,
            zero_point_precision=qparams_precision,
        )
    else:
        raise AssertionError(f"Invalid quantization scheme - {bits} precision")




def _replace_with_custom_fn_if_matches_filter(
    model,
    replacement_fn,
    filter_fn,
    cur_fqn="",
    device=None,
    extra_args: Optional[Tuple[Any, ...]] = (),
) -> None:
    """
    Recursively replaces each child module in `model` with the result of `replacement_fn(child)`
    if `filter_fn(child)` returns `True`.

    Args:
        model (torch.nn.Module): The model containing modules to be replaced.
        replacement_fn (Callable[[torch.nn.Module], torch.nn.Module]): The function to replace matching modules.
        filter_fn (Callable[[torch.nn.Module], bool]): The filter function to determine which modules to replace.
        cur_fqn (str, optional): The current fully qualified name of the module being processed. Defaults to "".
        device (device, optional): Device to move the model to before applying `filter_fn`. Defaults to None.
        extra_args (Tuple[Any, ...], optional): optional extra args to pass to `replacement_fn`.

    Returns:
        None
    """

    if isinstance(model, Float8Linear):
        with torch.device("meta"):
            new_module = nn.Linear(model.in_features, model.out_features)
        new_module.weight = model.weight
        new_module.bias = model.bias
        model = new_module
    if filter_fn(model, cur_fqn[:-1]):
        if device is not None:
            model.to(device=device)  # move to device before quantization
        model = replacement_fn(model, cur_fqn[:-1], *extra_args)
        return model
    else:
        named_children_list = list(model.named_children())
        for name, child in named_children_list:
            new_child = _replace_with_custom_fn_if_matches_filter(
                child,
                replacement_fn,
                filter_fn,
                f"{cur_fqn}{name}.", ### cur_fqn and name will give you the layer name that you have to match with the JSON file => 
                device,
                extra_args,
            )
            if new_child is not child and new_child is not None:
                setattr(model, name, new_child)
        if device is not None:
            model.to(device=device)  # move parent module to device
        return model



def convert_layer_name(name: str) -> str:
    mapping = {
        'layers.{}.attn.q_proj': 'blk.{}.attn_q',
        'layers.{}.attn.q_proj': 'blk.{}.attn_q',
        'layers.{}.attn.k_proj': 'blk.{}.attn_k',
        'layers.{}.attn.k_proj': 'blk.{}.attn_k',
        'layers.{}.attn.v_proj': 'blk.{}.attn_v',
        'layers.{}.attn.v_proj': 'blk.{}.attn_v',
        'layers.{}.attn.output_proj': 'blk.{}.attn_output',
        'layers.{}.mlp.w1': 'blk.{}.ffn_gate',
        'layers.{}.mlp.w3': 'blk.{}.ffn_up',
        'layers.{}.mlp.w2': 'blk.{}.ffn_down',
        'output': 'output',
        'tok_embeddings': 'token_embd',
        "norm.scale": "output_norm",
    }

    if "layers." in name:
        for old_pattern, new_pattern in mapping.items():
            layer_num = name.split('layers.')[1].split('.')[0]
            if old_pattern.format(layer_num) in name:
                try:
                    mapped_pattern = new_pattern.format(layer_num)
                    # if "blk." not in mapped_pattern:
                    #     continue
                    return mapped_pattern
                except IndexError:
                    continue
    else:
        # try:
        # print(name)
        return mapping[name]
        # except:
            # import pdb; pdb.set_trace()
    
    return name



################ is called in prepare() for 8avarw #################
def _replace_linear_8davarw(
    module: torch.nn.Module,
    groupsize: int,
    padding_allowed: bool,
    precision: torch.dtype,
    scales_precision: torch.dtype,
    linear_class: Type[torch.nn.Module],
    copy_weights: bool = False,
):
    # import pdb; pdb.set_trace()
    # import the util function here to avoid circular dependency

    def filter_fn(child: torch.nn.Module, cur_fqn: str) -> bool:
        return isinstance(child, nn.Linear) and (
            _check_linear_int4_k(child.in_features, groupsize) or padding_allowed ## check if something has to be done here
        )

    def replacement_fn(child: torch.nn.Module, layer_type) -> torch.nn.Module:
        ### get layer_name -> quant_scheme mapping here ------------->
        ## sample layer_type = 'layers.0.attn.q_proj'
        with open('/shareddata/dheyo/shivanvitha/torchtune/quant_config_example.json', 'r') as quant_file:
            quant_map = json.load(quant_file)

        # import pdb; pdb.set_trace()
        try:
            quant_key = convert_layer_name(layer_type)
            quant_value = quant_map[quant_key + ".weight"]
            bits, group_size = bit_map[quant_value]["bits"], bit_map[quant_value]["group_size"]
            # pdb.set_trace()
            print(f"{layer_type} -> {quant_key} -> {quant_value} -> {group_size}")

        except:
            bits = 8 ## default

        new_linear = linear_class(
            child.in_features,
            child.out_features,
            bias=child.bias is not None,
            device=child.weight.device,
            groupsize=group_size,
            precision=precision,
            scales_precision=scales_precision,
            bits=bits ### replace with the mapping
        )
        # In distributed training, the model may be instantiated
        # on the meta device, in which case there is no need to
        # copy the weights, and doing so will result in an error
        if copy_weights and child.weight.device != torch.device("meta"):
            new_linear.weight = child.weight
            new_linear.bias = child.bias
        return new_linear


    _replace_with_custom_fn_if_matches_filter(module, replacement_fn, filter_fn)



class Int8DynActIntVarWeightQATQuantizer(_LegacyQATQuantizer):
    """
    Quantizer for performing QAT on a model, where linear layers have int8
    dynamic per token fake quantized activations and int4 fake quantized
    grouped per channel weights.
    """

    def __init__(
        self,
        groupsize: int = 256,
        padding_allowed: bool = False,
        precision: torch.dtype = torch.float32,
        scales_precision: torch.dtype = torch.float32,
    ) -> None:
        super().__init__()
        self.groupsize: int = groupsize
        self.padding_allowed: bool = padding_allowed
        self.precision: torch.dtype = precision
        self.scales_precision: torch.dtype = scales_precision
        self.layer_counter = 0



    def prepare(
        self, model: torch.nn.Module, *args: Any, **kwargs: Any
    ) -> torch.nn.Module:
        # import pdb; pdb.set_trace()

        _replace_linear_8davarw(
            model,
            self.groupsize,
            self.padding_allowed,
            self.precision,
            self.scales_precision,
            Int8DynActIntVarWeightQATLinear,
            copy_weights=True,
        )
        # import pdb; pdb.set_trace()
        f = open("/shareddata/dheyo/shivanvitha/torchtune/dummy1.md", 'w')
        f.write(str(model))
        f.close()
        return model


    def convert(
        self, model: torch.nn.Module, *args: Any, **kwargs: Any
    ) -> torch.nn.Module:
        self._convert_qat_linear_8davarw(model)
        return model

    def _convert_qat_linear_8davarw(self, module: torch.nn.Module): ### modify this method accordingly
        """
        Replace all `Int8DynActInt4WeightQATLinear` with `Int8DynActInt4WeightLinear`.
        """


        # for name, param in module.named_modules():
        #     if isinstance(param, Int8DynActIntVarWeightQATLinear):
        #         print(f"NAME: {name}")

        with open('/shareddata/dheyo/shivanvitha/torchtune/quant_config_example.json', 'r') as quant_file:
            quant_map = json.load(quant_file)

        for name, child in module.named_children():

            # import pdb; pdb.set_trace()
            #### get layer_name -> quant_scheme mapping here and set it to "bits" and pass to the classes/methods below -------->
                # print(f"{name} -> {quant_key}")
                # bits = 8

            if isinstance(child, Int8DynActIntVarWeightQATLinear):
                print(f"{name}")



                if "proj" in name:
                    name_proxy = f"layers.{self.layer_counter}.attn.{name}"
                elif "w1" in name or "w2" in name or "w3" in name:
                    name_proxy = f"layers.{self.layer_counter}.mlp.{name}"
                    if "w3" in name:
                        self.layer_counter += 1

                quant_key = convert_layer_name(name_proxy)
                quant_value = quant_map[quant_key + ".weight"]
                bits, group_size = bit_map[quant_value]["bits"], bit_map[quant_value]["group_size"]
                print(f"{name} -> {quant_key} -> {bits} bits -> {group_size}")


                # bits = 4

                config = child.weight_fake_quantizer.config
                quantized_linear = Int8DynActIntVarWeightLinear(
                    child.in_features,
                    child.out_features,
                    child.bias is not None,
                    groupsize=group_size,
                    precision=child.weight.dtype,
                    scales_precision=config.scale_precision,
                    bits=bits ## set it to bits
                )
                setattr(module, name, quantized_linear)

                # Load weights and qparams into quantized linear
                n_bit = bits
                (qmin, qmax) = _get_qmin_qmax(n_bit)
                (s, zp) = get_group_qparams_symmetric(
                    child.weight,
                    n_bit,
                    config.group_size,
                    precision=config.scale_precision,
                )
                zp = zp.to(config.zero_point_precision)
                from torchao._executorch_ops import (
                    _quantized_decomposed_quantize_per_channel_group_wrapper,
                )

                q_weight = _quantized_decomposed_quantize_per_channel_group_wrapper(
                    child.weight,
                    s,
                    zp,
                    qmin,
                    qmax,
                    torch.int8,
                    config.group_size,
                )
                quantized_linear.weight = q_weight
                quantized_linear.scales = s
                quantized_linear.zeros = zp
                if child.bias is not None:
                    quantized_linear.bias = child.bias
                
                # print(f"Number of named_modules = {len(list(module.named_modules()))}")

            else:
                # print(f"CHILD in ELSE: {child}")
                # print(f"Number of named_modules = {len(list(module.named_modules()))}")

                self._convert_qat_linear_8davarw(child)

    # def get_activation_fake_quantize_config(self) -> Optional[FakeQuantizeConfig]:
    #     return _get_8davarw_activation_config(self.scales_precision)

    # def get_weight_fake_quantize_config(self) -> Optional[FakeQuantizeConfig]:
    #     print("This method is called")
    #     return _get_8davarw_weight_config(self.groupsize, self.scales_precision)



###### int 6 weights
class Int8DynActIntVarWeightQATLinear(FakeQuantizedLinear):
    """
    This module implements a linear layer with int8 dynamic per token fake
    quantized activations with int4 fake quantized grouped per channel weights.

    args:
        groupsize: the number of elements in each quantized group for weights
        precision: precision of weights
        scales_precision: precision of per group scales and zero points

    Note: we hardcode activation scales to use torch.fp32, but allow users to specify the weight scales (defaults to torch.fp32).
    To get an exact numerical match with Int8DynamicActivationInt4WeightConfig, users must use the same dtype for both the weights
    and the scales. Here scales_precision refers specifically to the weight scales only, not the activation scales.
    """

    def __init__(
        self,
        in_features: int,
        out_features: int,
        bias: bool = False,
        device: torch.device = None,
        groupsize: int = 256,
        precision: torch.dtype = torch.float32,
        scales_precision: torch.dtype = torch.float32,
        bits: int = 4
    ) -> None:
        # Use torch.float32 to match torchao.quantization.quant_api._int8_asymm_per_token_quant,
        # which is used in PTQ routines
        activation_config = _get_8davarw_activation_config(torch.float32)
        weight_config = _get_8davarw_weight_config(groupsize, scales_precision, bits=bits)
        super().__init__(
            in_features,
            out_features,
            bias,
            activation_config,
            weight_config,
            device=device,
            dtype=precision,
        )

    def enable_fake_quant(self, enabled: bool = True):
        self.activation_fake_quantizer.enabled = enabled
        self.weight_fake_quantizer.enabled = enabled

    def disable_fake_quant(self):
        self.enable_fake_quant(False)