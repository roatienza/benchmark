# `benchmark`
Utilities to perform deep learning models benchmarking.

Model inference efficiency is a big concern in deploying deep learning models. Efficiency is quantified as the Pareto-optimality of the target metric (eg accuracy) and model number of parameters, computational complexity like FLOPS and latency. `benchmark` is a tool to compute parameters, FLOPS and latency. The sample usage below shows how to determine the number of parameters and FLOPS. Also indicated are the different latency improvement as a function of accelerators and model format. The fastest is when ONNX and TensorRT are utilized.

## FLOPS, Parameters and Latency of ResNet18

Experiment performed on RTX 6000. Assuming 1k classes, `224x224x3` image and batch size of `1`.
```
FLOPs: 1,819,065,856
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

## Sample benchmarking of `resnet18` using `timm`

- GPU + ONNX + TensorRT
```
python3 benchmark.py --resnet --onnx-model model.onnx --tensorrt
```

- GPU + ONNX
```
python3 benchmark.py --resnet --onnx-model model.onnx
```

- GPU 
```
python3 benchmark.py --resnet --onnx-model
```

- CPU 
```
python3 benchmark.py --resnet --onnx-model --device cpu
```

- CPU + ONNX
```
python3 benchmark.py --resnet --onnx-model --device cpu --onnx-model model.onnx
```



