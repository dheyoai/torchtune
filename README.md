# Torchtune setup for DheyoAI

```
git clone git@github.com:dheyoai/torchtune.git
```

```
pip install .
```

## For AMD GPUs
```
pip install --pre --upgrade torch torchvision torchao --index-url https://download.pytorch.org/whl/nightly/rocm6.4
```

Note: For further instructions checkout [Original README](./README_original.md)

## Getting started with QAT
```
tune download <model> --output-dir <your_output_dir>
```

### To run distributed fine-tuning on 2 GPUs
```
tune run --nproc_per_node 2 qat_distributed --config recipes/configs/<path_to_yaml_config_file>
```

The checkpoints and trainer states will be stored in the path mentioned under `output_dir` in the above yaml config file.

All quantization modes supported by torchtune are in [torchtune/training/quantization.py](torchtune/training/quantization.py)

