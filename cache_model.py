from FlagEmbedding import BGEM3FlagModel,FlagLLMReranker,FlagReranker
import torch
import os

os.environ.setdefault('HF_HUB_CACHE',"./HF_HUB_CACHE")

# Assuming your model loading code here
model_name = 'BAAI/bge-m3'
model = BGEM3FlagModel('BAAI/bge-m3', use_fp16=True)
reranker = FlagReranker('BAAI/bge-reranker-v2-m3', use_fp16=True)

# Check CUDA availability and move the model to GPU if available
if torch.cuda.is_available():
    model = model.model.to('cuda')
    reranker = reranker.model.to('cuda')
    print("Model moved to GPU.")
else:
    print("GPU not available, model using CPU.")