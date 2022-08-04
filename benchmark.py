'''

Install the following packages.

pip3 install fvcore
pip3 install onnxruntime-gpu

# if no gpu, use cpu
pip3 install onnxruntime

CUDA:
# remove the old
conda uninstall cudatoolkit
# update to the new cudnn
conda install cudnn

TensorRT:
https://docs.nvidia.com/deeplearning/tensorrt/install-guide/index.html#installing-pip

python3 -m pip install --upgrade setuptools pip
python3 -m pip install nvidia-pyindex
python3 -m pip install --upgrade nvidia-tensorrt

(Optional) Torch-tensort
pip install torch-tensorrt -f https://github.com/NVIDIA/Torch-TensorRT/releases
# need super user access
sudo apt install python3-libnvinfer-dev python3-libnvinfer 

TODO:
Try on nvidia docker containers

'''

import torch 
import numpy as np
import time
import os
import torchvision
import timm     
from fvcore.nn import FlopCountAnalysis, flop_count_table, parameter_count
from argparse import ArgumentParser
from models import SimpleCNN, TransformerBlock
from dataloaders import ClassifierLoader
from metrics import AverageMeter, accuracy
from ui import progress_bar

def to_numpy(tensor):
    return tensor.detach().cpu().numpy() if tensor.requires_grad else tensor.cpu().numpy()

def compare_results(torch_out, ort_outs):
    # compare ONNX Runtime and PyTorch results
    return np.testing.assert_allclose(to_numpy(torch_out), ort_outs[0], rtol=1e-02, atol=1e-03)


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


def evaluate_model(model, loader, device='cpu'):
    # evaluate the model
    model.eval()
    top1 = AverageMeter()
    top5 = AverageMeter()
    with torch.no_grad():
        for i, data in enumerate(loader):
            x, target = data
            x = x.to(device)
            target = target.to(device)

            y = model(x)
            acc1, acc5 = accuracy(y, target, (1, 5))
            top1.update(acc1[0], x.size(0))
            top5.update(acc5[0], x.size(0))

            progress_bar(i,
                         len(loader),
                         'Accuracy: Top1: %0.2f%%, Top5: %0.2f%%'
                         % (top1.avg, top5.avg))



def get_args():
    parser = ArgumentParser(description='EfficientDL')    
    # for testing torchvision and timm models
    parser.add_argument('--model',  default=None, help='name of timm or torchvision model')
    
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

    # number of repetitions
    parser.add_argument('--repetitions', type=int, default=100, help='number of repetitions')

    # list all models
    parser.add_argument('--list-models', action='store_true', default=False, help='list all models')
    # find a model
    parser.add_argument('--find-model', type=str, default=None, help='find a model')
    # compute accuracy
    parser.add_argument('--compute-accuracy', action='store_true', \
                        default=False, help='compute model accuracy on ImageNet1k')
    # imagenet folder
    parser.add_argument('--imagenet', type=str, default="/data/imagenet", help='imagenet folder')
    args = parser.parse_args()
    return args

def get_torchvision_models():
    models = list(torchvision.models.__dict__.keys())
    torchvision_models = []
    for model in models:
        if model.islower() and "__" not in model and model[0] != "_":
            torchvision_models.append(model)

    return torchvision_models

def get_timm_models():
    timm_models = timm.list_models(pretrained=True)
    return timm_models

if __name__ == '__main__':
    args = get_args()
    device = torch.device(args.device)
    torchvision_models = get_torchvision_models()
    timm_models = get_timm_models()
    if args.list_models:
        for model in torchvision_models:
            print("torchvision.models." +  model)
        print(80 * '-')
        for model in timm_models:
            print("timm:", model)
        exit(0)
    elif args.find_model:
        if args.find_model in torchvision_models:
            print("Found: torchvision.models." + args.find_model)
            exit(0)
        elif args.find_model in timm_models:
            print("Found in timm models:", args.find_model)
            exit(0)
        else:
            print("model not found")
            exit(1)

    if args.attention:
        # ViT Tiny ImageNet1k configuration in Timm
        seqlen = (args.image_size // args.patch_size) ** 2
        model = TransformerBlock(in_features=args.embed_dim, hidden_features=args.embed_dim*4, \
                                 num_heads=args.num_heads, reduction_factor=args.reduction_factor, \
                                 layer_norm=args.layer_norm, residual=args.residual).to(device)
        
        dummy_input = torch.randn(args.batch_size, seqlen, args.embed_dim, \
                                  dtype=torch.float).to(device)
    elif args.model is not None:
        if args.model in torchvision_models:
            print(f"Using torchvision model: {args.model}")
            model_name = "torchvision.models." + args.model
            model = eval(model_name)(pretrained=True).to(device)
        elif args.model in timm_models:
            print(f"Using timm model: {args.model}")
            model = timm.create_model(args.model, pretrained=True).to(device)
        else:
            print("Model not found in torchvision or timm", args.model)
            exit(0)
        #model = torchvision.models.resnet18(pretrained=True).to(device)
        dummy_input = torch.randn(args.batch_size, 3, args.image_size, args.image_size, \
                                  dtype=torch.float,).to(device)
    else:
        # SimpleCNN ImageNet
        model = SimpleCNN(group=args.group, separable=args.separable, \
                          residual=args.residual).to(device)
        dummy_input = torch.randn(args.batch_size, 3, args.image_size, args.image_size, \
                                  dtype=torch.float,).to(device)

    model.eval()
    flops = FlopCountAnalysis(model, dummy_input)
    param = parameter_count(model)
    print("FLOPS: {:,}".format(flops.total()))
    print("Parameters: {:,}".format(param[""]))

    if args.compute_accuracy:
        loader = ClassifierLoader(root=args.imagenet).test 
        evaluate_model(model, loader, device=device)
        exit(0)

    if args.verbose:
        print(flop_count_table(flops))

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
        timeit_cpu(dummy_input, torch_model=model, onnx_model=args.onnx_model, \
                   device=device, repetitions=args.repetitions)
        os.remove(args.onnx_model)
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
        

    timeit_cpu(dummy_input, torch_model=model, device=device, repetitions=args.repetitions)
    if args.verbose:
        y = model(dummy_input)
        print("In shape:", dummy_input.shape)
        print("Out shape:", y.shape)