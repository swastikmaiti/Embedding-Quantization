from datasets import load_from_disk
import numpy as np
from usearch.index import Index
from sentence_transformers.quantization import quantize_embeddings

dataset = load_from_disk("/workspaces/senetence-transformer-in-action/vectorized_dataset")
embeddings = np.array(dataset["embedding"], dtype=np.float32)

int8_embeddings = quantize_embeddings(embeddings, "int8")
index = Index(ndim=1024, metric="ip", dtype="i8")
index.add(np.arange(len(int8_embeddings)), int8_embeddings)
index.save("medicine_details_int8_usearch.index")