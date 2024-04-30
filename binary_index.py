from datasets import load_from_disk
import numpy as np
from faiss import IndexBinaryFlat, write_index_binary
from sentence_transformers.quantization import quantize_embeddings

dataset = load_from_disk("/workspaces/senetence-transformer-in-action/vectorized_dataset")
embeddings = np.array(dataset["embedding"], dtype=np.float32)

ubinary_embeddings = quantize_embeddings(embeddings, "ubinary")
index = IndexBinaryFlat(1024)
index.add(ubinary_embeddings)
write_index_binary(index, "medicine_details.index")