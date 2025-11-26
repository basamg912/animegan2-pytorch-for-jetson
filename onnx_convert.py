import os
import argparse

import torch
import torch.onnx
from model import Generator

torch.backends.cudnn.enabled = False
torch.backends.cudnn.benchmark = False
torch.backends.cudnn.deterministic = True

def convert_model(args):
    device = args.device
    net = Generator()
    net.load_state_dict(torch.load(args.model, map_location='cuda'))
    net.to(device)
    net.eval()
    
    x = torch.randn(1,3,args.input_shape, args.input_shape,requires_grad=True)
    x = x.to(device)
    out = net(x)
    
    torch.onnx.export(
        net,
        x,
        args.output,
        export_params=True,
        #opset_version=20,
        input_names=['input'],
        output_names=['output'],
        dynamic_axes={
            'input' : {
                0: 'batch_size',
                2: 'height',
                3: 'width'
                },
            'output' : {
                0: 'batch_size',
                2: 'height',
                3: 'width'
                }
        }
    )
    

if __name__ =='__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument(
        '--model',
        type=str,
        default='./weights/paprika.pt'
    )
    parser.add_argument(
        '--output',
        type=str,
        default='./weights/model.onnx'
    )
    parser.add_argument(
        '--input_shape',
        type=int,
        default=256
    )
    parser.add_argument(
        '--device',
        type=str,
        default='cuda:0'
    )
    
    args = parser.parse_args() 
    
    convert_model(args)