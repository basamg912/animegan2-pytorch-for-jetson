import tensorrt as trt
from model import Generator
from PIL import Image
import pycuda.driver as cuda
import pycuda.autoinit
from torchvision.transforms.functional import to_tensor, to_pil_image

import ctypes
import numpy as np
TRT_Logger = trt.Logger(trt.Logger.VERBOSE)
runtime = trt.Runtime(TRT_Logger)
trt.init_libnvinfer_plugins(TRT_Logger, "")
try:
    with open("./weights/model.engine", "rb") as f:
        engine = runtime.deserialize_cuda_engine(f.read())
        
except exception as e:
    print(e)
    
#context = engine.create_execution_context()

def preprocess_image(input_image):
    mean = np.array([0.485, 0.456, 0.406]).astype('float32')
    std = np.array([0.229, 0.224, 0.225]).astype('float32')
    data = (np.array(input_image).astype('float32')/float(255.0) - mean) / std
    return np.moveaxis(data,2,0)

def postprocess_image(output_array):
    output_array = np.moveaxis(output_array, 0, 2)
    mean = np.array([0.485, 0.456, 0.406]).astype('float32')
    std = np.array([0.229, 0.224, 0.225]).astype('float32')
    output = output_array * std + mean
    output = output*255.0
    output = np.clip(output, 0, 255)
    return output.astype(np.uint8) # uint8
    
target_size = (512,512)
input_image = Image.open("./samples/inputs/1.jpg").convert("RGB")
input_image = input_image.resize(target_size)
input_image.save('./samples/outputs/input.png','png')
image_width = input_image.width
image_height = input_image.height

input_tensor = preprocess_image(input_image)
output_file = "./samples/outputs/output.png"

with engine.create_execution_context() as context:
    context.set_binding_shape(engine.get_binding_index("input"), (1,3, image_height, image_width))
    bindings = []
    for binding in engine:
        binding_idx = engine.get_binding_index(binding)
        size = trt.volume(context.get_binding_shape(binding_idx))
        dtype = trt.nptype(engine.get_binding_dtype(binding))
        
        if engine.binding_is_input(binding):
            input_buffer = np.ascontiguousarray(input_tensor)
            # input_tensor.nbytes 를 손수 계산 → image_size = image_width * image_height * 3 * 4
            input_memory = cuda.mem_alloc(input_tensor.nbytes)
            bindings.append(int(input_memory))
        else:
            output_buffer = cuda.pagelocked_empty(size, dtype)
            output_memory = cuda.mem_alloc(output_buffer.nbytes)
            bindings.append(int(output_memory))
    stream = cuda.Stream()
    # copy data to GPU
    cuda.memcpy_htod_async(input_memory, input_buffer, stream)
    
    context.execute_async_v2(bindings=bindings, stream_handle=stream.handle)
    cuda.memcpy_dtoh_async(output_buffer, output_memory, stream)
    stream.synchronize()
    
    output = np.reshape(output_buffer, (3, image_height, image_width))
    img = postprocess_image(output)
    img = Image.fromarray(img)
    img.convert("RGB").save(output_file,"png")
    print(img.size)