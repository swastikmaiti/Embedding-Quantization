
import gradio as gr
from datasets import load_from_disk
import pandas as pd
from sentence_transformers import SentenceTransformer
from sentence_transformers.quantization import quantize_embeddings
import faiss
from usearch.index import Index
import numpy as np
import os

base_path = os.getcwd()
full_path = os.path.join(base_path, 'conala')
conala_dataset = load_from_disk(full_path)

int8_view = Index.restore(os.path.join(base_path, 'conala_int8_usearch.index'), view=True)
binary_index: faiss.IndexBinaryFlat = faiss.read_index_binary(os.path.join(base_path, 'conala.index'))

model = SentenceTransformer("sentence-transformers/all-MiniLM-L6-v2")

def search(query, top_k: int = 20):
    # 1. Embed the query as float32
    query_embedding = model.encode(query)

    # 2. Quantize the query to ubinary. To perform actual search with faiss
    query_embedding_ubinary = quantize_embeddings(query_embedding.reshape(1, -1), "ubinary")


    # 3. Search the binary index 
    index =  binary_index
    _scores, binary_ids = index.search(query_embedding_ubinary, top_k)
    binary_ids = binary_ids[0]


    # 4. Load the corresponding int8 embeddings. To perform rescoring to calculate score of fetched documents.
    int8_embeddings = int8_view[binary_ids].astype(int)

    # 5. Rescore the top_k * rescore_multiplier using the float32 query embedding and the int8 document embeddings
    scores = query_embedding @ int8_embeddings.T

    # 6. Sort the scores and return the top_k
    indices = scores.argsort()[::-1][:top_k]
    top_k_indices = binary_ids[indices]
    top_k_scores = scores[indices]

    top_k_codes = conala_dataset[top_k_indices]

    return top_k_codes


def response_generator(user_prompt):
    top_k_outputs = search(user_prompt)
    probs = top_k_outputs['prob']
    snippets = top_k_outputs['snippet']
    idx = np.argsort(probs)[::-1]
    results = np.array(snippets)[idx]
    filtered_results = []
    for item in results:
        if len(filtered_results)<3:
            if item not in filtered_results:
                filtered_results.append(item)

    output_template = "User Query: {user_query}\nBelow are some examples of previous conversations.\nQuery: {query1} Solution: {solution1}\nQuery: {query2} Solution: {solution2}\nYou may use the above examples for reference only. Create your own solution and provide only the solution"
    output_template = "The top three most relevant code snippets from the database are:\n\n1. {snippet1}\n\n2. {snippet2}\n\n3. {snippet3}"
    output = f'{output_template.format(snippet1=filtered_results[0],snippet2=filtered_results[1],snippet3=filtered_results[2])}'

    return {output_box:output}  


with gr.Blocks() as demo:
    
    gr.Markdown(
    """
    # Embedding Quantization

    ## Quantized Semantic Search

    - ***Embedding:*** all-MiniLM-L6-v2
    - ***Vetor DB:*** faiss, USearch
    - ***Vector_DB Size:*** `5,93,891`

    """)

    state_var = gr.State([])


    input_box = gr.Textbox(autoscroll=True,visible=True,label='User',info="Enter a query.",value="How to extract the n-th elements from a list of tuples in python?")
    output_box = gr.Textbox(autoscroll=True,max_lines=30,value="Output",label='Assistant')
    gr.Interface(fn=response_generator, inputs=[input_box], outputs=[output_box],
                 delete_cache=(20,10),
                 allow_flagging='never')
    
demo.queue()
demo.launch()
