# `benchmark`
Utilities to perform deep learning models benchmarking.

Model inference efficiency is a big concern in deploying deep learning models. Efficiency is quantified as the Pareto-optimality of the target metric (eg accuracy) and model number of parameters, computational complexity like FLOPS and latency. `benchmark` is a tool to compute parameters, FLOPS and latency. The sample usage below shows how to determine the number of parameters and FLOPS. Also indicated are the different latency improvements as a function of accelerator and model format. The fastest is when both ONNX and TensorRT are utilized.

## FLOPS, Parameters and Latency of ResNet18

Experiment performed on GPU: Quadro RTX 6000 24GB, CPU: AMD Ryzen Threadripper 3970X 32-Core Processor. Assuming 1k classes, `224x224x3` image and batch size of `1`.
```
FLOPS: 1,819,065,856
Parameters: 11,689,512
```

| **Accelerator** | **Latency (usec)** | Speed up (x) |
| :--- | ---: | --: |
| CPU | 10,706 | 1 |
| CPU + ONNX | 3,990 | 2.7 |
| GPU | 1,982 | 5.4 |
| GPU + ONNX | 1,218 | 8.8 |
| GPU + ONNX + TensorRT | 917 | 11.7 |




## Install requirements
```
pip3 install -r requirements.txt
```

Additional packages.

- CUDA:
Remove the old.
```
conda uninstall cudatoolkit
```
Update to the new cudnn
```
conda install cudnn
```

- [TensorRT](https://docs.nvidia.com/deeplearning/tensorrt/install-guide/index.html#installing-pip)
```
python3 -m pip install --upgrade setuptools pip
python3 -m pip install nvidia-pyindex
python3 -m pip install --upgrade nvidia-tensorrt
```

- (Optional) Torch-tensort
```
pip3 install torch-tensorrt -f https://github.com/NVIDIA/Torch-TensorRT/releases
```
Warning: need super user access
```
sudo apt install python3-libnvinfer-dev python3-libnvinfer 
```

## Sample benchmarking of `resnet18`

- GPU + ONNX + TensorRT
```
python3 benchmark.py --model resnet18 --onnx-model model.onnx --tensorrt
```

- GPU + ONNX
```
python3 benchmark.py --model resnet18 --onnx-model model.onnx
```

- GPU 
```
python3 benchmark.py --model resnet18 
```

- CPU 
```
python3 benchmark.py --model resnet18  --device cpu
```

- CPU + ONNX
```
python3 benchmark.py --model resnet18 --device cpu --onnx-model model.onnx
```

## Compute model accuracy on ImageNet1k
Assuming imagenet dataset folder is `/data/imagenet`. Else modify the location using `--imagenet` option.

```
python3 benchmark.py --model resnet18 --compute-accuracy
```

## List all supported models
All `torchvision.models` and `timm` models will be listed:

```
python3 benchmark.py --list-models
```

## Find a specific model

```
python3 benchmark.py --find-model xcit_tiny_24_p16_224
```

## Other models 
- Latency in usec

| **Accelerator** | **Resnet50** | **MobileV2** | **MobileV3** | **ShuffleV2** | **Squeeze** | **SwinV2** | **Deit** | **Eff0** |
| :--- | ---: | --: | ---: | --: | ---: | --: | --: | --: |
| CPU | 29,840 | 11,870 | 6,498 | 6,607 | 8,717 | 52,120 | 14,952 | 14,089 | 
| CPU + ONNX | 10,666 | 2,564 | 4,484 | 2,479 | 3,136 | 50,094  | 10,484 | 10,392 |
| GPU | 1,982 | 4,781 | 3,689 |  4,135 | 1,741 | 6,963 | 3,526 | 6,047|
| GPU + ONNX | 2,715 | 1,107 | 1,128 | 1,392 | 851 | 3,731 | 1,650 | 2,175 |
| GPU + ONNX + TensorRT | 1,881 | 670 | 570 | 404 | 443 | 3,327 | 1,170 | 1,222 |

ResNet50 - `resnet50`, MobileV2 - `mobilenet_v2`, MobileV3 - `mobilenet_v3_small`, Shuffle - `shufflenet_v2_x0_5`, Squeeze - `squeezenet1_0`, SwinV2 - `swinv2_cr_tiny_ns_224`, Deit - `deit_tiny_patch16_224`

- Parameters and FLOPS

| **Model** | **Parameters (M)** | **GFLOPS** | **Top1 (%)** | **Top5 (%)** |
| :--- | ---: | --: | --: |  --: |
| `resnet18` | 11.7 | 1.8 | 69.76 | 89.08 | 
| `resnet50` | 25.6 | 4.1 | 76.15 | 92.87 | 
| `mobilenet_v2` | 3.5 | 0.3 | 71.87 | 90.29  |
| `mobilenet_v3_small` | 2.5 | 0.06 | 67.67 | 87.41 |
| `shufflenet_v2_x0_5` | 1.4 | 0.04 | 60.55 | 81.74 |
| `squeezenet1_0` | 1.2 | 0.8 | 58.10  | 80.42 |
| `swinv2_cr_tiny_ns_224` | 28.3 | 4.7 | 81.54 | 95.77 |
| `deit_tiny_patch16_224` | 5.7 | 1.3  |  72.02 | 91.10 |
| `efficientnet_b0` | 5.3 | 0.4 | 77.67 |  93.58 |


