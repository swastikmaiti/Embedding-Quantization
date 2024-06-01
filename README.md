# Embedding Quantization

In this this work we implement Embedding Quantization. This is a great technique for significantly faster and cheaper retrieval.
We go through the step by step procedure of qunatizing embedding along with conceptual explanation and implementation.

# Introduction

To implement a RAG System, we will mrequire some kind of retrieval system. Thus there has to be some database to retrieve from.
In LLM world this is a Vector Database. This database is different from normal SQL database in the sense that a Vector Databse store
embedding of some dimension and for retrieval purpose we need to perform heavy computation on the databse to generate similarity score.

Vector Database are costly becasuse it is both `Memory hungry` and `Compuation hungry`. We need a sophisticated technique to bring down
compuation load and memory load while preserving the accuray. Here comes `Embedding Quantization`.

# Binary Quantization

To generate Binary Quantization from a `float32` embedding we simply threshold quantize each dimension at 0. It simple means `f(x)=0 if x<=0
else 1`
