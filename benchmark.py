'''

Install the following packages.

pip3 install fvcore
pip3 install onnxruntime-gpu

CUDA:
conda uninstall cudatoolkit
conda install cudnn

TensorRT:
https://docs.nvidia.com/deeplearning/tensorrt/install-guide/index.html#installing-pip

python3 -m pip install --upgrade setuptools pip
python3 -m pip install nvidia-pyindex
python3 -m pip install --upgrade nvidia-tensorrt

(Optional) Torch-tensort
pip install torch-tensorrt -f https://github.com/NVIDIA/Torch-TensorRT/releases
sudo apt install python3-libnvinfer-dev python3-libnvinfer 

'''

import torch 
import numpy as np
import time
from fvcore.nn import FlopCountAnalysis, flop_count_table, parameter_count
from argparse import ArgumentParser
from models import SimpleCNN, TransformerBlock


def to_numpy(tensor):
    return tensor.detach().cpu().numpy() if tensor.requires_grad else tensor.cpu().numpy()

def compare_results(torch_out, ort_outs):
    # compare ONNX Runtime and PyTorch results
    return np.testing.assert_allclose(to_numpy(torch_out), ort_outs[0], rtol=1e-03, atol=1e-05)


# for some reason the timing using cpu is smaller than the timing using cuda
# TODO: find out why
def timeit_cpu(dummy_input, torch_model=None, onnx_model=None,\
               device='cpu', repetitions=100, verbose=False):
    torch_model.eval()
    timings = np.zeros((repetitions, 1))
    if onnx_model is not None:
        # load only if needed
        import onnxruntime
        import onnx
        model = onnx.load(onnx_model)
        onnx.checker.check_model(model)
        if verbose: 
            print(onnx.helper.printable_graph(onnx_model.graph))

        if device == 'cpu':
            providers = ['CPUExecutionProvider']
        elif device == 'cuda':
            providers = ['CUDAExecutionProvider']
        else: # use all including tensorrt
            providers = onnxruntime.get_available_providers()

        session = onnxruntime.InferenceSession(onnx_model, providers=providers, \
                                               session_options=onnxruntime.SessionOptions())
        if verbose:
            print(f"Providers: {providers}")
        
        # make sure both torch_model and onnx_model have the same prediction
        # will trigger assertion if not the same
        with torch.no_grad():
            torch_out = torch_model(dummy_input)
            dummy_input = {session.get_inputs()[0].name: to_numpy(dummy_input)}
            ort_outs = session.run(None, dummy_input)
            compare_results(torch_out, ort_outs)

    # warm up the device by loading the graph and performing inference
    with torch.no_grad():
        for _ in range(repetitions):
            if onnx_model is not None:
                y = session.run(None, dummy_input)
                y = y[0]
            else:
                y = torch_model(dummy_input)
                y = y.detach().cpu().numpy()
            #print(type(y))
            
    # log elapsed times
    with torch.no_grad():
        for rep in range(repetitions):
            start_time = time.time()
            if onnx_model is not None:
                y = session.run(None, dummy_input)
                y = y[0]
            else:
                y = torch_model(dummy_input)
                y = y.detach().cpu().numpy()
            timings[rep] = (time.time() - start_time)  * 1e6

    mean_syn = np.sum(timings) / repetitions
    std_syn = np.std(timings)
    print("(CPU Timer) Ave infer time: {0:.1f} usec".format(mean_syn))
    print("(CPU Timer) Std infer time: {0:.1f} usec".format(std_syn))


def get_args():
    parser = ArgumentParser(description='EfficientDL')    
    # for testing resnet18 model
    parser.add_argument('--resnet', action='store_true', default=False, help='use resnet')
    
    # for testing SimpleCNN
    parser.add_argument('--group', action='store_true', \
                        default=False, help='use group convolution')
    parser.add_argument('--separable', action='store_true', \
                        default=False, help='use depthwise separable convolution')
    parser.add_argument('--residual', action='store_true', \
                        default=False, help='use residual connection')
 
    # for testing a transformer block
    parser.add_argument('--attention', action='store_true', default=False, help='use attention')
    parser.add_argument('--embed-dim', type=int, default=192, help='embedding size')
    parser.add_argument('--patch-size', type=int, default=16, help='patch size')
    parser.add_argument('--num-heads', type=int, default=3, help='number of heads')
    parser.add_argument('--reduction-factor', type=int, default=1, help='reduction factor')
    parser.add_argument('--layer-norm', action='store_true', default=False, help='use layer norm')
 
    # testing hyperparameters
    parser.add_argument('--batch-size', type=int, default=1, help='batch size')
    parser.add_argument('--image-size', type=int, default=224, help='image size')
    # print more debug info
    parser.add_argument('--verbose', action='store_true', default=False, help='verbose')

    # choice of acceleration
    parser.add_argument('--onnx-model', default=None, help='use onnx if not None')
    # use tensorrt in onnxruntime, --onnx-model is required
    parser.add_argument('--tensorrt', action='store_true', \
                        default=False, help='use tensorrt, --onnx-model is required')
    # either gpu (cuda) or cpu device
    choices = ["cuda", "cpu",]
    parser.add_argument('--device', type=str, default=choices[0], \
                        choices=choices, help='device for inference')

    args = parser.parse_args()
    return args

if __name__ == '__main__':
    args = get_args()
    device = torch.device(args.device)
    if args.attention:
        # ViT Tiny ImageNet1k configuration in Timm
        seqlen = (args.image_size // args.patch_size) ** 2
        model = TransformerBlock(in_features=args.embed_dim, hidden_features=args.embed_dim*4, \
                                 num_heads=args.num_heads, reduction_factor=args.reduction_factor, \
                                 layer_norm=args.layer_norm, residual=args.residual).to(device)
        
        dummy_input = torch.randn(args.batch_size, seqlen, args.embed_dim, \
                                  dtype=torch.float).to(device)
    elif args.resnet:
        import torchvision
        model = torchvision.models.resnet18(pretrained=True).to(device)
        dummy_input = torch.randn(args.batch_size, 3, args.image_size, args.image_size, \
                                  dtype=torch.float,).to(device)
    else:
        # SimpleCNN ImageNet
        model = SimpleCNN(group=args.group, separable=args.separable, \
                          residual=args.residual).to(device)
        dummy_input = torch.randn(args.batch_size, 3, args.image_size, args.image_size, \
                                  dtype=torch.float,).to(device)

    model.eval()
    if args.onnx_model is not None:
        # set model parameters to non-trainable, it seems that torch.onnx.export() 
        # does not recognize torch.no_grad() action
        for param in model.parameters():
            param.requires_grad = False

        with torch.no_grad():
            torch.onnx.export(model,              
                              dummy_input,
                              args.onnx_model,
                              # store the trained parameter weights inside the model file
                              export_params=True,
                              # whether to execute constant folding for optimization   
                              do_constant_folding=True,  
                              input_names=['inputs'],
                              output_names=['outputs'],
                              verbose=args.verbose
                              )
        
        device = 'tensorrt' if args.tensorrt else args.device
        timeit_cpu(dummy_input, torch_model=model, onnx_model=args.onnx_model, device=device)
        exit(0)
    
    elif args.tensorrt:
        # TODO: not working yet
        # Reference: 
        # https://medium.com/pytorch/accelerating-pytorch-inference-with-torch-tensorrt-on-gpus-896e06ff1637
        # set model parameters to non-trainable
        for param in model.parameters():
            param.requires_grad = False
        # convert to tensorrt

        import torch_tensorrt
        trt_module = torch_tensorrt.compile(model,
            # input shape
            inputs = [torch_tensorrt.Input((args.batch_size, 3, args.image_size, args.image_size))],    
            enabled_precisions = {torch_tensorrt.dtype.half} # Run with FP16
        )
        # save the TensorRT embedded Torchscript
        torch.jit.save(trt_module, "trt_torchscript_module.ts")
        result = trt_module(dummy_input) # Run inference
        
        exit(0)
        

    flops = FlopCountAnalysis(model, dummy_input)
    param = parameter_count(model)
    print("FLOPs: {:,}".format(flops.total()))
    print("Parameters: {:,}".format(param[""]))

    if args.verbose:
        print(flop_count_table(flops))
    
    timeit_cpu(dummy_input, torch_model=model, device=device)
    print("In shape:", dummy_input.shape)
    y = model(dummy_input)
    print("Out shape:", y.shape)