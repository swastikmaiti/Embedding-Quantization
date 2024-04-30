# %%


# %% [markdown]
# # Import Libraries

# %%
import numpy as np
from sentence_transformers import SentenceTransformer

# %% [markdown]
# ## Load Dataset

# %%
from datasets import load_from_disk

# %%
dataset = load_from_disk('/workspaces/senetence-transformer-in-action/medical_terms')

# %% [markdown]
# # Create vector embedding

# %%
model = SentenceTransformer("mixedbread-ai/mxbai-embed-large-v1")

# %%
'Medicine Name', 'Composition', 'Uses', 'Side_effects'
def get_embeddings(examples):
    vectors = {}
    model_input = ['Medicine Name: '+mn+' Composition:'+c+' Uses:'+u+' Side_effects:'+se for mn,c,u,se in zip(examples['Medicine Name'],examples['Composition'],examples['Uses'],examples['Side_effects'])]
    out =  model.encode(model_input)
    vectors['embedding'] = out
    return vectors

# %%
vectorized_dataset = dataset.map(get_embeddings,batched=True)

# %%
vectorized_dataset.save_to_disk('vectorized_dataset')


