# Embedding Quantization

In this this work we implement Embedding Quantization. This is a great technique for significantly faster and cheaper retrieval.
We go through the step by step procedure of qunatizing embedding along with conceptual explanation and implementation.

# Introduction

To implement a RAG System, we will mrequire some kind of retrieval system. Thus there has to be some database to retrieve from.
In LLM world this is a Vector Database. This database is different from normal SQL database in the sense that a Vector Databse store
embedding of some dimension and for retrieval purpose we need to perform heavy computation on the databse to generate similarity score.

Vector Database are costly becasuse it is both `Memory hungry` and `Compuation hungry`. We need a sophisticated technique to bring down
compuation load and memory load while preserving the accuray. Here comes `Embedding Quantization`.

We will take our example implemetation to make comarison with normal embedding.

# Tools and Parameters Specifications

- **faiss:** Store binary quantization embedding
- **USearch:** Store scalar (int8) quantization
- **all-MiniLM-L6-v2:** Embedding Model
- **Embedding Dimension:** 384
- **Databse Size:** `5,93,891`

# F32 Implementation Memory Requirement

- No of Index = 593891
- Embedding dim = 384
- dtype = f32, each diension is of 32 bits

  Memory usage = `913 MB` (Rounded)

# Binary Quantization

To generate Binary Quantization from a `float32` embedding we simply threshold quantize each dimension at 0. It simple means `f(x)=0 if x<=0
else 1`. We store the Binary Quantized Embedding in Vector DB. To perform retrieval we convert user query to binary quantized embedding. 
Then we use Hamming Distance between qury embedding and Vector DB embeddings to perform similarity check on the Vector Database. Hamming distance
is the number of bits by with two embeddings differ. Lowe the Hamming Distance more is the document relevant and hence higher similarity.

### What is the Memory usage with Binary Quantized Embedding?
- No of Index = 593891
- Embedding dim = 384
- dtype = bit, each diension is of 1 bit
With our implemented specifications it comes to be `29 MB` (Rounded). It is `32X` lower memory requirement than `float32`

# Rescoring Technique
We have a way to retrieve similar documents with humming distance as similarity measure for Binary Quantization. Though it speed ups the
retrieval process but preserve roughly `92.5%` retrieval performance.
We use a technique called `rescoring` intoduced in the [paper](https://arxiv.org/abs/2106.00882) to preserve alomost `96%` retrieval performance.
In rescoring techinque we first retrive `top_k`*`rescore multiplier` document i.e. `rescore multiplier x` times more documents than require.
Then we perform dot prduct between their embeddings (binary) and query (f32) embedding to calculate similarity scores and return `top_k` according to 
the new similarity score.

