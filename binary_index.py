from datasets import load_from_disk
import numpy as np
from faiss import IndexBinaryFlat, write_index_binary
from sentence_transformers.quantization import quantize_embeddings

import os
path_to_vectorised_dataset = os.path.join(os.getcwd(),'vectorized_dataset')

dataset = load_from_disk(path_to_vectorised_dataset)
embeddings = np.array(dataset["embedding"], dtype=np.float32)

ubinary_embeddings = quantize_embeddings(embeddings, "ubinary")
index = IndexBinaryFlat(384)    ## embedding dimension
index.add(ubinary_embeddings)
write_index_binary(index, "conala.index")