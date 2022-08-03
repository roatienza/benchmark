# benchmark
Utilities to perform deep learning models benchmarking

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



