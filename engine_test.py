import tensorrt as trt
from model import Generator
TRT_Logger = trt.Logger(trt.Logger.VERBOSE)
runtime = trt.Runtime(TRT_Logger)

try:
    with open("./weights/model.engine", "rb") as f:
        engine = runtime.deserialize_cuda_engine(f.read())
        
except exception as e:
    print(e)
    
context = engine.create_execution_context()
