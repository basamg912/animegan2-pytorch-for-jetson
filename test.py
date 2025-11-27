import os
import argparse
import tensorrt as trt
from PIL import Image
import numpy as np

import torch
from torchvision.transforms.functional import to_tensor, to_pil_image
from cuda.bindings import runtime as cudart
import ctypes

from model import Generator
import tensorrt as trt

torch.backends.cudnn.enabled = False
torch.backends.cudnn.benchmark = False
torch.backends.cudnn.deterministic = True

TRT_Logger = trt.Logger(trt.Logger.VERBOSE)
runtime = trt.Runtime(TRT_Logger)
trt.init_libnvinfer_plugins(TRT_Logger, "")

def load_image(image_path, x32=False):
    img = Image.open(image_path).convert("RGB")

    if x32:
        def to_32s(x):
            return 256 if x < 256 else x - x % 32
        w, h = img.size
        img = img.resize((to_32s(w), to_32s(h)))

    return img

def infer(engine, input, output):
    image_width = input.width
    image_height = input.height
    with engine.create_execution_context() as context:
        input_buffers = {}
        input_memories = {}
        
        tensor_names = [engine.get_tensor_name(i) for i in range(engine.num_io_tensors)]
        for tensor in tensor_names:
            size = trt.volume(context.get_tensor_shape(tensor))
            dtype = trt.nptype(engine.get_tensor_dtype(tensor))
            
            if engine.get_tensor_mode(tensor) == trt.TensorIOMode.INPUT:
                context.set_input_shape(tensor, (1,3, image_height, image_width))
                input_buffers[tensor] = np.ascontiguousarray(input_image)
                err, input_memories[tensor] = cudart.cudaMalloc(input_image.nbytes)
                assert err==cudart.cudaError_t.cudaSuccess
                context.set_tensor_address(tensor, input_memories[tensor])
            else:
                err, output_buffer_ptr = cudart.cudaMalloc(size * dtype().itemsize())
                assert err == cudart.cudaError_t.cudaSuccess
                pointer_type = ctypes.POINTER(np.ctypelib.as_ctypes_type(dtype))
                output_buffer = np.ctypeslib.as_array(ctypes.cast(output_buffer_ptr, pointer_type), (size,))
                err, output_memory = cudart.cudaMalloc(output_buffer.nbytes)
                assert err == cudart.cudaError_t.cudaSuccess
                context.set_tensor_address(tensor, output_memory)
        err, stream = cudart.cudaStreamCreate()
        assert err==cuda.cudaError_t.cudaSuccess
        
        
    

def test(args):
    TRT_Logger = trt.Logger(trt.Logger.Warning) # runtime instance
    runtime = trt.Runtime(TRT_Logger)
    with open(args.checkpoint, 'rb') as f:
        engine = runtime.deserialize_cuda_engine(f.read())
    
    os.makedirs(args.output_dir, exist_ok=True)

    for image_name in sorted(os.listdir(args.input_dir)):
        if os.path.splitext(image_name)[-1].lower() not in [".jpg", ".png", ".bmp", ".tiff"]:
            continue
            
        image = load_image(os.path.join(args.input_dir, image_name), args.x32)

        with torch.no_grad():
            image = to_tensor(image).unsqueeze(0) * 2 - 1
            out = net(image.to(device), args.upsample_align).cpu()
            out = out.squeeze(0).clip(-1, 1) * 0.5 + 0.5
            out = to_pil_image(out)

        out.save(os.path.join(args.output_dir, image_name))
        #print(f"image saved: {image_name}")


if __name__ == '__main__':

    parser = argparse.ArgumentParser()
    parser.add_argument(
        '--checkpoint',
        type=str,
        default='./weights/paprika.pt',
    )
    parser.add_argument(
        '--input_dir', 
        type=str, 
        default='./samples/inputs',
    )
    parser.add_argument(
        '--output_dir', 
        type=str, 
        default='./samples/results',
    )
    parser.add_argument(
        '--device',
        type=str,
        default='cuda:0',
    )
    parser.add_argument(
        '--upsample_align',
        type=bool,
        default=False,
        help="Align corners in decoder upsampling layers"
    )
    parser.add_argument(
        '--x32',
        action="store_true",
        help="Resize images to multiple of 32"
    )
    args = parser.parse_args()
    
    test(args)
