'''
creted by @tharakaR
2021/04/25
'''
from os import environ
from psutil import cpu_count
import time
import sys

# Constants from the performance optimization available in onnxruntime
# It needs to be done before importing onnxruntime
environ["OMP_NUM_THREADS"] = str(cpu_count(logical=True))
environ["OMP_WAIT_POLICY"] = 'ACTIVE'
from onnxruntime import GraphOptimizationLevel, InferenceSession, SessionOptions, get_all_providers
from transformers import DistilBertTokenizer
import numpy as np
import time
from sentenceCreator import split_into_sentences


class DistilBertSST5QONNXbase(object):

    def __init__(self,model_path,provider):
        '''
        Constructor
        '''
        self.model_path = model_path
        self.provider = provider
        self.tokenizer = DistilBertTokenizer.from_pretrained("models/tokenizers/distilbert_sst5")
        # assert provider in get_all_providers(), f"provider {provider} not found, {get_all_providers()}"
        # Few properties that might have an impact on performances (provided by MS)
        self.options = SessionOptions()
        self.options.intra_op_num_threads = 1
        self.options.graph_optimization_level = GraphOptimizationLevel.ORT_ENABLE_ALL

        # Load the model as a graph and prepare the CPU backend 
        self.session = InferenceSession(self.model_path, self.options, providers=[self.provider])
        self.session.disable_fallback()
        self.cpu_model_qnt = self.session

    def preprocess(self, x):
        '''
        x: text input
        returns: tokenized text
        '''
        encoded_input = self.tokenizer(x, return_tensors='np')
        inputs_onnx = {k: v for k, v in encoded_input.items()}
        return inputs_onnx

    def predict(self, x):
        '''
        x: text input
        returns: prediction values; negative value, positive value, time(ms)
        '''
        tktext = self.preprocess(x)
        t1 = time.time()
        y = self.cpu_model_qnt.run(None, tktext)
        T = time.time() - t1
        y = self.postprocess(y)
        return y[0], y[1], y[2], T
        
    def postprocess(self, x):
        '''
        x: prediction
        returns: final result
        '''
        postprocessed = x[0][0]
        output = np.exp(postprocessed) / np.sum(np.exp(postprocessed))
        return output

    def batchpreprocess(self, x):
        '''
        x: text input
        returns: batch of tokenized sentences
        '''
        xs = split_into_sentences(x)
        print('xs',xs)
        encoded_inputs = self.tokenizer.__call__(xs, return_tensors='np', padding=True)
        inputs_onnx = {k: v for k, v in encoded_inputs.items()}
        print('inputs_onnx',inputs_onnx,file=sys.stderr)
        return inputs_onnx,xs

    def batchpredict(self, x):
        '''
        x: batch of tokenized inputs
        returns: batch predictions
        '''
        tktexts, xs = self.batchpreprocess(x)
        t1 = time.time()
        y = self.cpu_model_qnt.run(None, tktexts)
        T = time.time() - t1
        print('y',y,file=sys.stderr)
        y1, y2 = self.batchpostprocess(y)
        print('y1',y1, file=sys.stderr)
        print('y2',y2,file=sys.stderr)
        return y1[0], y1[1], y1[2], T, self.colorcombiner( xs, y2 )

    def batchpostprocess(self, x):
        '''
        x: batch predictions
        returns: final results 
        '''

        print('X',x,file=sys.stderr)
        postprocessed = x[0]
        print('postprocessed =', postprocessed, file=sys.stderr)
        output = np.exp(postprocessed) / np.sum(np.exp(postprocessed), axis=1).reshape(-1, 1)
        print('Output =',output, file=sys.stderr)
        return np.mean(output, axis=0), output
        
    def colorcombiner(self,senslist,values):
        outlist = []
        for idx, sen in enumerate(senslist):
            if values[idx][0] >=0.5 :
                color = 'aquamarine'
            elif values[idx][1] >= 0.5:
                color = 'rgb(236, 127, 127)'
            else:
                color = 'yellow'
            outlist.append((color, sen))
        return outlist
        
    