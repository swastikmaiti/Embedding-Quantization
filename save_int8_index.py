from datasets import load_from_disk
import numpy as np
from usearch.index import Index
from sentence_transformers.quantization import quantize_embeddings


import os
path_to_vectorised_dataset = os.path.join(os.getcwd(),'vectorized_dataset')

dataset = load_from_disk(path_to_vectorised_dataset)
embeddings = np.array(dataset["embedding"], dtype=np.float32)

int8_embeddings = quantize_embeddings(embeddings, "int8")
index = Index(ndim=384, metric="ip", dtype="i8")             ### embedding dimension
index.add(np.arange(len(int8_embeddings)), int8_embeddings)
index.save("conala_int8_usearch.index")