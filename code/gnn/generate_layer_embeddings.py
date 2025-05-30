#############################
# GENERATE LAYER EMBEDDINGS #
############################

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import DataLoader, Dataset
from transformers import AutoModel, AutoTokenizer
import torch.optim as optim
import pandas as pd
import numpy as np
from sklearn.decomposition import PCA
from sklearn.metrics.pairwise import cosine_similarity
import seaborn as sns
import matplotlib.pyplot as plt
from nltk.tokenize import word_tokenize
import nltk
from nltk.corpus import stopwords
from nltk.tokenize import RegexpTokenizer
from gnnutils import *
from models import *

nltk.download('punkt_tab')

# Read the HS2012 list
hs_text = pd.read_csv("../../data/UNComtrade/HS2012.csv", encoding="latin1", index_col=False)
# Get Product Code (first 2)
hs_text["product_code"] = hs_text.id.astype(str).str[:2]
# Keep only cols
hs_text = hs_text[["product_code", "text"]]
# Drop the first row (TOTAL), and keep only first two cols
hs_text.drop(0, inplace=True)
product_ids = [f"{i:02d}" for i in range(1, 100) if i not in [77, 98, 99]]
hs_text = hs_text[hs_text.product_code.isin(product_ids)] # 77 & 98 don't exist. 99 is not present in BACI

# TEXT PREPROCESSING
# Stopwords
remove_words = set(stopwords.words('english'))
remove_words.add("other")

# Join all textual descriptions
data = hs_text.groupby("product_code").apply(lambda x: ", ".join(x.text), include_groups=False).reset_index(name="text")
# Remove uninformative text: numbers, dashes -, and empty strings
data.loc[:, "text"] = data.text.replace(r"(\d|\-)", "", regex=True, inplace=False)
data.loc[:, "text"]= data.text.replace(r"\s+", " ", regex=True, inplace=False)
data.loc[:, "text"] = data.text.str.lower()
tokenizer = RegexpTokenizer(r'\w+')
data.loc[:, "text"] = data.text.apply(lambda x: tokenizer.tokenize(x))
data.loc[:, "text"] = data.text.apply(lambda x: " ".join(set([word for word in x if word not in remove_words])))

# Get the list of descriptions.
descriptions = data['text'].tolist()

# Make integer labels for downstream finetuning
labels = torch.tensor(range(len(product_ids))) 

# Create dataset and loader
dataset = SentenceDataset(descriptions, labels)
dataloader = DataLoader(dataset, batch_size=32, shuffle=True)

# Initialize model and move to device.
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
model = SentenceTransformerWithClassifier(projection_dim=8, num_classes=96, fine_tune_transformer=False)
model.to(device)

criterion = nn.CrossEntropyLoss().to(device)

# Setup an optimizer. If only fine-tuning the projection head, this will be relatively few parameters.
optimizer = optim.Adam(model.parameters(), lr=1e-3)

# Training loop
n_epochs = 20

model.train()
for epoch in range(n_epochs):
    running_loss = 0.0
    for batch_sentences, batch_labels in dataloader:
        optimizer.zero_grad()
        # Forward pass: compute predicted embeddings.
        outputs = model(batch_sentences)
        loss = criterion(outputs, batch_labels.to(device))
        loss.backward()
        optimizer.step()
        running_loss += loss.item() * len(batch_sentences)
    epoch_loss = running_loss / len(dataset)
    print(f"Epoch {epoch+1}/{n_epochs} Loss: {epoch_loss:.4f}")

print("Fine-tuning complete!")

# Run the forward pass to generate the embeddings (without classifier)
embeddings = model(descriptions, return_embedding=True)
print("Projected Embeddings Shape:", embeddings.shape) 

## Add PCI scalar to the embeddings
df = pd.concat([pd.read_stata("../../data/atlas/hs12_country_country_product_year_4_2012_2016.dta"),
                pd.read_stata("../../data/atlas/hs12_country_country_product_year_4_2017_2021.dta"),
                pd.read_stata("../../data/atlas/hs12_country_country_product_year_4_2022.dta")])
products = pd.read_csv("../../data/atlas/product_hs12.csv", dtype={"code": str})
# Include product code
pci = df.merge(products[["product_id", "code"]], how="left", on="product_id")
pci.rename(columns={"code": "product_code"}, inplace=True)

pci["product_code"] = pci.product_code.str[:2] ## Reduce product code to 2 digits

# We only need one PCI value per product-year
pci = pci.drop_duplicates(subset=["product_id", "year"], keep="first")

# Average PCI per commodity calass
pci_all_years = pci.groupby(["year", "product_code"])["pci"].mean().reset_index(name="pci").sort_values(["year", "product_code"])


# Scale the embeddings yearwise (since PCI is year-dependent)
layer_embeddings = {}

for year in range(2012, 2023):

    print(f"Scaling year {year}...")
    
    pci_year = pci_all_years.loc[(pci_all_years.year == year) & (pci_all_years.product_code.isin(product_ids))].pci.values
        
    # Scale the embeddings by the PCI values
    scaled_embeddings = embeddings * torch.tensor(pci_year, dtype=embeddings.dtype).unsqueeze(1)
    
    # Store the scaled embeddings in a dictionary
    layer_embeddings[year] = scaled_embeddings

# Save embeddings
with open("layer_embeddings.pickle", "wb") as f:
    pickle.dump(layer_embeddings, f)
