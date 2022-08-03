'''
conda uninstall cudatoolkit
conda install cudnn

Install TensorRT:
https://docs.nvidia.com/deeplearning/tensorrt/install-guide/index.html#installing-pip

python3 -m pip install --upgrade setuptools pip
python3 -m pip install nvidia-pyindex
python3 -m pip install --upgrade nvidia-tensorrt

# for torch-tensort
pip install torch-tensorrt -f https://github.com/NVIDIA/Torch-TensorRT/releases
sudo apt install python3-libnvinfer-dev python3-libnvinfer 

'''


import torch.nn as nn
import torch 
from itertools import repeat
import collections.abc

# From PyTorch internals
# https://github.com/rwightman/pytorch-image-models/blob/master/timm/models/layers/helpers.py
def _ntuple(n):
    def parse(x):
        if isinstance(x, collections.abc.Iterable):
            return x
        return tuple(repeat(x, n))
    return parse


to_2tuple = _ntuple(2)

# MACs vs FLOPs : https://github.com/sovrasov/flops-counter.pytorch/issues/16
# MACs is the number of multiplications and additions (Multiply Accumulate)
# FLOPs is the number of floating point operations
# GFLOPs = 2 * GMACs as general each MAC contains one multiplication and one addition.
# Fused Multiply Add (FMA) is a special operation that is used to perform 
# a multiply and add operation in one step.

class SimpleCNN(nn.Module):
    def __init__(self, in_channel=3, out_channel=24, 
                 kernel_size=3, group=False, separable=False, residual=False):
        super(SimpleCNN, self).__init__()
        self.separable = separable
        self.residual = residual
        groups = in_channel if group or separable else 1
        assert out_channel % groups == 0
        if separable:
            self.conv1 = nn.Conv2d(in_channel, in_channel, padding=kernel_size//2, \
                                   kernel_size=kernel_size, groups=groups)
            self.conv2 = nn.Conv2d(in_channel, out_channel, \
                                   kernel_size=1,)
        else:
            self.conv1 = nn.Conv2d(in_channel, out_channel, padding=kernel_size//2, \
                                   kernel_size=kernel_size, groups=groups)

    def forward(self, x):
        y = self.conv1(x)
        if self.separable:
            y = self.conv2(y)
        # concatenate x and y
        
        if self.residual:
            y = torch.cat((x, y), 1)
        return y


# fvcore does not compute FLOPS due to softmax (maybe all activations)
# softmax torch code in c++ 
# https://github.com/pytorch/pytorch/blob/master/aten/src/ATen/native/SoftMax.cpp
class Softmax(nn.Module):
    def __init__(self, dim=-1):
        super(Softmax, self).__init__()
        self.dim = dim
    def forward(self, x):
        return torch.softmax(x, dim=self.dim)


# https://github.com/rwightman/pytorch-image-models/blob/master/timm/models/vision_transformer.py
class Attention(nn.Module):
    def __init__(self, dim, num_heads=8, qkv_bias=False, attn_drop=0., proj_drop=0.,
                 reduction_factor=1):
        super().__init__()
        assert dim % num_heads == 0, 'dim should be divisible by num_heads'
        self.num_heads = num_heads
        head_dim = dim // num_heads
        self.scale = head_dim ** -0.5
        self.reduction_factor = reduction_factor
        if reduction_factor > 1:
            self.fc = nn.Linear(dim * reduction_factor, dim)
     
        self.qkv = nn.Linear(dim, dim * 3, bias=qkv_bias)
        self.attn_drop = nn.Dropout(attn_drop)
        self.proj = nn.Linear(dim, dim)
        self.proj_drop = nn.Dropout(proj_drop)

    def forward(self, x):
        B, N, C = x.shape
        if self.reduction_factor > 1:
            x = x.reshape(B, N // self.reduction_factor, C * self.reduction_factor)
            x = self.fc(x)
            B, N, C = x.shape
        
        qkv = self.qkv(x).reshape(B, N, 3, self.num_heads, C // self.num_heads).permute(2, 0, 3, 1, 4)
        # q,k,v: [B, num_heads, N, C // num_heads]
        q, k, v = qkv.unbind(0)   # make torchscript happy (cannot use tensor as tuple)
            
        # attn: [B, num_heads, N, N]
        attn = (q @ k.transpose(-2, -1)) * self.scale
        attn = attn.softmax(dim=-1)
        attn = self.attn_drop(attn)

        # x: [B, N, C]
        x = (attn @ v).transpose(1, 2).reshape(B, N, C)
        x = self.proj(x)
        x = self.proj_drop(x)
        return x

# modified from timm module
# https://github.com/rwightman/pytorch-image-models/blob/master/timm/models/layers/mlp.py
class Mlp(nn.Module):
    """ MLP as used in Vision Transformer, MLP-Mixer and related networks
    """
    def __init__(self, in_features, hidden_features=None, out_features=None, \
                 act_layer=nn.GELU, bias=True, drop=0.):
        super().__init__()
        out_features = out_features or in_features
        hidden_features = hidden_features or in_features
        bias = to_2tuple(bias)
        drop_probs = to_2tuple(drop)

        self.fc1 = nn.Linear(in_features, hidden_features, bias=bias[0])
        self.act = act_layer()
        self.drop1 = nn.Dropout(drop_probs[0])
        self.fc2 = nn.Linear(hidden_features, out_features, bias=bias[1])
        self.drop2 = nn.Dropout(drop_probs[1])

    def forward(self, x):
        x = self.fc1(x)
        x = self.act(x)
        x = self.drop1(x)
        x = self.fc2(x)
        x = self.drop2(x)
        return x

# a ViT transformer block
class TransformerBlock(nn.Module):
    def __init__(self, in_features, hidden_features, num_heads, \
                 layer_norm=False, residual=False, reduction_factor=1):
        super().__init__()
        self.attn = Attention(in_features, num_heads, reduction_factor=reduction_factor)
        self.mlp = Mlp(in_features, hidden_features=hidden_features)
        self.norm1 = nn.LayerNorm(in_features) if layer_norm else None
        self.norm2 = nn.LayerNorm(in_features) if layer_norm else None
        self.residual = residual

    def forward(self, x):
        if self.residual:
            if self.norm1 is not None:
                x = x + self.attn(self.norm1(x))
                x = x + self.mlp(self.norm2(x))
            else:
                x = x + self.attn(x)
                x = x + self.mlp(x)
            
            return x
            
        if self.norm1 is not None:
            x = self.norm1(x)
        x = self.attn(x)
        if self.norm2 is not None:
            x = self.norm2(x)
        x = self.mlp(x)
        return x

