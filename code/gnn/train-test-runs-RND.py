### TRAINING AND TESTING

#from torch_geometric.datasets import Planetoid
from torch_geometric.transforms import NormalizeFeatures
#from torch_geometric.data import HeteroData, download_url, extract_zip
from torch_geometric.data import Data
from torch_geometric.nn import GCNConv, GATConv, global_mean_pool, GATv2Conv
import torch
import numpy as np
import pandas as pd
import torch_geometric
from torch.nn import functional as F
from torch_geometric import loader
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.metrics import confusion_matrix, classification_report, precision_score, recall_score, f1_score, accuracy_score
from functools import total_ordering
import os
import pickle
from gnnutils import *
from models import *

# Settings
model_type = "RND"
graphs_type = "export" # "total", "export"
layered = True
multi_graph = True
graph_identifier = f"{'multi-graph-' if multi_graph else ''}{graphs_type}{'-layered' if (layered and not multi_graph) else ''}"


train_graphs, test_graphs = get_preloaded_graphs(path=f"../../data/graphs_data/{'multi-graph/' if multi_graph else ''}{graphs_type}")

if layered:
    # Read layer embeddings
    layer_embeddings = pickle.load(open("layer_embeddings.pickle", "rb"))
    train_graphs, test_graphs = append_layer_embedding(train_graphs, test_graphs, layer_embeddings, multi_graph=multi_graph)

# Graph constants
num_classes = 1
num_features = test_graphs[0].num_features    
pos_weight = get_pos_weight(train_graphs=train_graphs)


for seed in range(1, 11):
    
    print("Seed: ", seed)

    set_seed(seed)

    model = RandomPredictor(random_seed=seed)

    print(model)

    #l, v = train3(model=model, train_graphs=train_graphs, optimizer=optimizer, criterion=criterion, \
    #                    scheduler=None, epochs=200, batch_size=-1, patience=20, save_path=f"MLP/{graphs_type}/{seed}/", random_seed=seed)

    #plot_train_curves(l, v, save_path=f"models/training/MLP/{graphs_type}/{seed}", show=False)

   
    #model.load_state_dict(torch.load(f"./models/training/MLP/{graphs_type}/{seed}/best_model.pt", weights_only=True))

    
    evaluate(model, test_graphs, save_path=f"RND-{graphs_type}-m-{seed}", show=False)