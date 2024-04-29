from fastapi import FastAPI
from FlagEmbedding import BGEM3FlagModel,FlagLLMReranker,FlagReranker
from pydantic import BaseModel
import asyncio
import torch
import numpy as np
import os

os.environ.setdefault('HF_HUB_CACHE',"./HF_HUB_CACHE")

app = FastAPI()

model = None
reranker = None
# Timer task for releasing the model
model_release_task = None
reranker_release_task = None
# Lock for thread-safe operations
model_lock = asyncio.Lock()
reranker_lock = asyncio.Lock()
# Set timeout for model release (in seconds). Adjust as necessary.
MODEL_RELEASE_TIMEOUT = 300  # 300 = 5 minutes for example
RERANKER_RELEASE_TIMEOUT = 300  # 300 = 5 minutes for example

class Items(BaseModel):
    queries: list[str]
class RerankItems(BaseModel):
    pairs: list[list[str]]
async def release_model():
    """
    A coroutine that waits for a specified timeout and then releases the model resource.
    """
    global model
    await asyncio.sleep(MODEL_RELEASE_TIMEOUT)
    async with model_lock:
        if model is not None:
            # model:BGEM3FlagModel=model
            print("Releasing model resources...")
            # Include any required cleanup for your model here
            # model.model.cpu()
            del model.model
            model = None
            torch.cuda.empty_cache()
            print("Model resources have been released.")
async def release_reranker():
    """
    A coroutine that waits for a specified timeout and then releases the reranker resource.
    """
    global reranker
    await asyncio.sleep(RERANKER_RELEASE_TIMEOUT)
    async with reranker_lock:
        if reranker is not None:
            # reranker:BGEM3FlagModel=reranker
            print("Releasing reranker resources...")
            # Include any required cleanup for your reranker here
            # reranker.model.cpu()
            del reranker.model
            reranker = None
            torch.cuda.empty_cache()
            print("Reranker resources have been released.")
def safe_convert_to_list(vectors):
    converted = []
    for vector in vectors:
        if isinstance(vector, torch.Tensor):
            vector = vector.cpu().detach().numpy().tolist()
        elif isinstance(vector, np.ndarray):
            vector = vector.tolist()
        if hasattr(vector, "__iter__") and not isinstance(vector, str):  # Check if it's iterable and not a string
            converted.append(list(vector))
        else:
            converted.append([vector])  # Wrap non-iterables in a list
    return converted
def safe_convert_to_list_fp16(vectors):
    converted = []
    for vector in vectors:
        # Case: PyTorch Tensor
        if isinstance(vector, torch.Tensor):
            vector = vector.cpu().detach().numpy()
        # No further else condition needed as we handle NumPy conversion next
        
        # Conversion to list; this holds for both NumPy arrays and lists derived from PyTorch tensors
        if isinstance(vector, np.ndarray):
            # Ensure maintaining the data type, especially for floating point numbers
            if vector.dtype == np.float16 or vector.dtype == np.float32 or vector.dtype == np.float64:
                vector = vector.tolist()
            else:
                # This is a safeguard; this branch maintains the original logic,
                # handling both floating points (though unnecessary) and other dtypes
                vector = vector.tolist()
        else:
            # This else block might be redundant but ensures non-np.ndarray iterables are handled
            if hasattr(vector, "__iter__") and not isinstance(vector, str):
                vector = list(vector)
            else:
                vector = [vector]  # Wrap non-iterables in a list
        
        converted.append(vector)
    return converted

@app.post("/embeddings/")
async def create_embedding(items: Items):
    global model, model_release_task
    async with model_lock:
        if model is None:
            model = BGEM3FlagModel('BAAI/bge-m3', use_fp16=True)
            print("Model loaded.")
        else:
            # If the model exists and there's a pending release task, cancel it.
            if model_release_task is not None and not model_release_task.done():
                model_release_task.cancel()
                print("Model release cancelled.")

        # Invoke the model to get embeddings
        embeddings = model.encode(items.queries, return_dense=True, return_sparse=True, return_colbert_vecs=False)
        
        dense_vectors:list=embeddings['dense_vecs']
        lexical_weights:list=embeddings['lexical_weights']
        sparse_vectors:list=[]
        for weights in lexical_weights:
            sparse_vector={}
            for token_id in weights:
                sparse_vector[token_id]=float(weights[token_id])
            sparse_vectors.append(sparse_vector)
    
    # Set or reset the release task
    model_release_task = asyncio.create_task(release_model())

    dense_vectors = safe_convert_to_list(dense_vectors)

    # return {"embeddings": embeddings}
    return {"dense_vectors": dense_vectors, "sparse_vectors": sparse_vectors}

@app.post("/rerank/")
async def rerank(items: RerankItems):
    global reranker, reranker_release_task
    async with reranker_lock:
        if reranker is None:
            reranker = FlagReranker('BAAI/bge-reranker-v2-m3', use_fp16=True)
            print("Reranker loaded.")
        else:
            # If the reranker exists and there's a pending release task, cancel it.
            if reranker_release_task is not None and not reranker_release_task.done():
                reranker_release_task.cancel()
                print("Reranker release cancelled.")

        # Invoke the reranker to get reranked pairs
        pairs=items.pairs
        scores = reranker.compute_score(pairs)
        combined = [(pair, scores[index]) for index, pair in enumerate(pairs)]
        sorted_combined = sorted(combined, key=lambda x: x[1], reverse=True)
        return {"sorted_pairs": sorted_combined}
