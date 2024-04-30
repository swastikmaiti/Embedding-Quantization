


# %% [markdown]
# # Import Libraries

# %%
import gradio as gr
from datasets import load_from_disk
import pandas as pd
from sentence_transformers import SentenceTransformer
from sentence_transformers.quantization import quantize_embeddings
import faiss
from usearch.index import Index

# %% [markdown]
# # Load Dataset

# %%
medical_dataset = load_from_disk("/workspaces/senetence-transformer-in-action/medical_terms")

# %% [markdown]
# ```Text
# Load the int8 and binary indices. Int8 is loaded as a view to save memory, as we never actually perform search with it.
# Int8 embedding is required to perform rescoring of fetched document. This is done bt performing inner product with F32 embedding of Query
# ```
# [Efficient Passage Retrieval with Hashing for Open-domain Question Answering](https://arxiv.org/abs/2106.00882)

# %%
# Load the int8 and binary indices. Int8 is loaded as a view to save memory, as we never actually perform search with it.
int8_view = Index.restore("/workspaces/senetence-transformer-in-action/medicine_details_int8_usearch.index", view=True)
binary_index: faiss.IndexBinaryFlat = faiss.read_index_binary("/workspaces/senetence-transformer-in-action/medicine_details.index")

# %% [markdown]
# # Import Model to generate embedding

# %%
model = SentenceTransformer(
    "mixedbread-ai/mxbai-embed-large-v1",
    prompts={
        "retrieval": "Represent this sentence for searching relevant passages: ",
    },
    default_prompt_name="retrieval",
)


# %%
def search(query, top_k: int = 5):
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
    top_k_Medicine_Name, top_k_Composition,top_k_Uses,top_k_Side_effects = zip(
        *[(medical_dataset[idx]["Medicine Name"], medical_dataset[idx]["Composition"],
           medical_dataset[idx]["Uses"], medical_dataset[idx]["Side_effects"]) for idx in top_k_indices.tolist()]
    )
    df = pd.DataFrame(
        {"Medicine_Name": top_k_Medicine_Name, "Composition": top_k_Composition,
                                                                "Uses": top_k_Uses, "Side_effects": top_k_Side_effects}
    )

    return df


# %%

with gr.Blocks(title="Quantized Retrieval") as demo:

    query = gr.Textbox(
        label="Query for Medicine Name",
        placeholder="Enter a Medicine Name",
    )
    query_com = gr.Textbox(
        label="Query for Composition",
        placeholder="Enter a composition",
    )
    query_use = gr.Textbox(
        label="Query for Uses",
        placeholder="Enter a Use Case",
    )
    query_side_effect = gr.Textbox(
        label="Query for Side_effects",
        placeholder="Enter a Side effect",
    )
    #with gr.Row():
    top_k = gr.Dropdown(choices=[1,2,4,8,10,15], label="No of Relevant Info",value=4)
    search_button = gr.Button(value="Search")

    with gr.Row():
        #with gr.Column(scale=4):
        output = gr.Dataframe(headers=['Medicine Name', 'Composition', 'Uses', 'Side_effects'])

    query.submit(search, inputs=[query, top_k], outputs=[output])
    search_button.click(search, inputs=[query, top_k], outputs=[output])

demo.queue()
demo.launch()


# %%
