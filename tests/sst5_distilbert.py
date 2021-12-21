from transformers import AutoTokenizer

from os import environ
from psutil import cpu_count
import time
import sys

# Constants from the performance optimization available in onnxruntime
# It needs to be done before importing onnxruntime
environ["OMP_NUM_THREADS"] = str(cpu_count(logical=True))
environ["OMP_WAIT_POLICY"] = 'ACTIVE'
from onnxruntime import GraphOptimizationLevel, InferenceSession, SessionOptions, get_all_providers
import numpy as np

model_path = '/Users/tharakarehan/Desktop/transformers-master/notebooks/onnx/distilbert-zero.onnx'
provider = "CPUExecutionProvider"
tokenizer = AutoTokenizer.from_pretrained('/Users/tharakarehan/Desktop/distilbert-base-uncased-mnli')
# assert provider in get_all_providers(), f"provider {provider} not found, {get_all_providers()}"
# Few properties that might have an impact on performances (provided by MS)
options = SessionOptions()
options.intra_op_num_threads = 1
options.graph_optimization_level = GraphOptimizationLevel.ORT_ENABLE_ALL

def postprocess(x):
        '''
        x: prediction
        returns: final result
        '''
        postprocessed = x[0][0]
        output = np.exp(postprocessed) / np.sum(np.exp(postprocessed))
        return output
# Load the model as a graph and prepare the CPU backend 
session = InferenceSession(model_path, options, providers=[provider])
session.disable_fallback()
cpu_model_qnt = session

encoded_input = tokenizer.encode_plus('Beef burger was horrible', 'food', return_tensors='np', truncation_strategy='only_first')
inputs_onnx = {k: v for k, v in encoded_input.items()}
print(inputs_onnx)
y = cpu_model_qnt.run(None, inputs_onnx)
y = postprocess(y)
print(y)
