---
title: quantized-semnatic-search
emoji: 💻
colorFrom: yellow
colorTo: purple
sdk: gradio
sdk_version: 4.28.3
app_file: app.py
pinned: false
license: openrail
---

# senetence-transformer-in-action

Language model gets boosted performance from their ability to capture semantics. To capture the semantic similarity between documents we
make use of sentence transformer. Senetence Transformers prodoce embedding of a gievn text. These emedding are what we call vectors in mathematics.
Just like a vector, a embedding may have several hundreds to thousands dimension. These embedding are in continuous number scale. They represent the 
semantic meaning of input text. Each ddimension has a different meaning (which dimeansion-what meaning is unknown, the unexplainable ai). So greater
the dimension better is the semantic representation.

# Curse of Dimensionality

High dimensional space increase computation and store complexity. Space and Time complexity are the reason preventing these wonderful models from
practical appliaction. Reducing dimensionality is a rudimentary and naive approach which drasctically reduces performance of a model.

# Embedding Quantization

Here we will talk about quantization of embedding which is completely different from quantizing a model itself.
In Quantization approach we generate the embedding from model in F32 format then we apply quantization to convert
the embedding into binary or int8.

- ***To Note:*** By quantization we mean we convert each dimeansion of output embedding from F32 to a binary or int8
- ***Detailed Explanation:*** Thanks to [HuggingFace Team](https://huggingface.co/blog/embedding-quantization) for the Wonderful Explanation.
- ***Quantization and Rescoring Technique:*** [Efficient Passage Retrieval with Hashing for Open-domain Question Answering](https://arxiv.org/abs/2106.00882)

  
# Implemetation Flow
1. Load Dataset: [11000 Medicine details](https://www.kaggle.com/datasets/singhnavjot2062001/11000-medicine-details).
2. Create F32 Embedding of Text.
3. Quantize Embedding to ubinary to perform similarity search with faiss
4. Quantize Embedding to int8 to which is required to perfom rescoring step.
5. Take User Query
6. Create Query embedding and create a new copy of quantized query embedding.
7. Perfrom similary search on faiss with quantized query embedding. Get indexes.
8. Fetch Int8 embedding of corresponding indexes.
9. Perform Rescoring
10. Yield Output.


# Project Architecture

- ***Sentence Transformer:*** mixedbread-ai/mxbai-embed-large-v1
- ***Vector Databse Implementation Libray:*** [faiss](https://github.com/facebookresearch/faiss.git), [USearch](https://unum-cloud.github.io/usearch/)
- ***App Library:*** Gradio

# Model Card Author

Swastik Maiti


