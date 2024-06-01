# Embedding Quantization

In this this work we implement Embedding Quantization. This is a great technique for significantly faster and cheaper retrieval.
We go through the step by step procedure of qunatizing embedding along with conceptual explanation and implementation.

All the explanations and Codes are reproduced based on the [Article](https://huggingface.co/blog/embedding-quantization) from
Hugging Face.

#

<img src="https://github.com/swastikmaiti/Embedding-Quantization/blob/b611b302ebe0ecc8303d965a3b117a086e5b5205/embedding-quantization.png">

# Introduction

To implement a RAG System, we will mrequire some kind of retrieval system. Thus there has to be some database to retrieve from.
In LLM world this is a Vector Database. This database is different from normal SQL database in the sense that a Vector Databse store
embedding of some dimension and for retrieval purpose we need to perform heavy computation on the databse to generate similarity score.

Vector Database are costly becasuse it is both `Memory hungry` and `Compuation hungry`. That means the embedding has to be present
on primary memory for compuation and computation is done on all the embeddings for similarity scoring.
We need a sophisticated technique to bring down compuation load and memory load while preserving the accuray. Here comes `Embedding Quantization`.

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
is the number of bits by with two embeddings differ. Lowe the Hamming Distance more is the document relevant and hence higher similarity. On average
`Binary Quantization` gives `24.76x` speed up and exactly `32x` memory saving.

### What is the Memory usage with Binary Quantized Embedding?
- No of Index = 593891
- Embedding dim = 384
- dtype = bit, each diension is of 1 bit
  
  Memory usage = `29 MB` (Rounded). It is `32X` lower memory requirement than `float32`

# Rescoring Technique
We have a way to retrieve similar documents with humming distance as similarity measure for Binary Quantization. Though it speed ups the
retrieval process but preserve roughly `92.5%` `retrieval performance`.
We use a technique called `rescoring` intoduced in the [paper](https://arxiv.org/abs/2106.00882) to preserve alomost `96%` `retrieval performance`.
In rescoring techinque we first retrive `top_k`*`rescore multiplier` document i.e. `rescore multiplier x` times more documents than require.
Then we perform dot prduct between their embeddings (binary) and query (f32) embedding to calculate similarity scores and return `top_k` according to 
the new similarity score.

# Scalar Quantization
This is another type of quantization to improve `retrieval performance`. Here instead of binary quantization we convert the `f32` embedding into 
`int8` embeddings or `unit8` embeddings. Scalar quantization reduce memory requirement by `4x` as compared to `8x` in Binary Quantization but it has
a `retrieval performance` of more than `99%` with a `rescore multiplier` of 10. On average `Scalar Quantization` gives `3.77x` speed up and exactly `4x` memory saving.

# Best of Both the World
To benifit from the `memory` and `computation` requiremnt of Binary Quantization and `retrieval performance` of Scalar Quantization, in practice we
use both the technique together.

We use `Binary Quantization` for the actual Vetor DB for in memory compuation. Separately we store the `Scalar Quantization` in disk space. We store in
disk space bcause we `do not` perfrom any compuation on this database. Firt we perform retrieval with `Binary Quantization` which is computation heavy, 
then we perform rescoring with `Scalar Quantization` and return top_k documents.

# Hugging Face
The App is Hosted on Hugging Face Gradio Space. [Embedding Quantization](https://huggingface.co/spaces/SwastikM/Embedding-Quantization)

#Acknowledgement
Thanks to Hugging Face Team for the in depth explanation on Embedding Quantization. [Article](https://huggingface.co/blog/embedding-quantization)

#
### If you find the repo helpful, please drop a ⭐

