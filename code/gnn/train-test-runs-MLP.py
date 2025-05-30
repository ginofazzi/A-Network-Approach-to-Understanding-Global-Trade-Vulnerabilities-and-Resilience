### TRAINING AND TESTING

# Numpy, Pandas, etc
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.metrics import confusion_matrix, classification_report, precision_score, recall_score, f1_score, accuracy_score

# Torch and PyTorch Geometric
import torch
from torch.nn import functional as F
import torch_geometric
from torch_geometric import loader
from torch_geometric.transforms import NormalizeFeatures

# Others
from functools import total_ordering
import os
import pickle
import json

# Custom
from gnnutils import *
from models import *

##########################################################

# Settings
model_type = "MLP"
graphs_type = "total" # "total", "export"
layered = True
multi_graph = True
graph_identifier = f"{'multi-graph-' if multi_graph else ''}{graphs_type}{'-layered' if (layered and not multi_graph) else ''}"

# GPU if possible
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
print("Device: ", device)

# Best parameters from Optuna
best_params = json.load(open("./models/best_params.json", "r"))
best_params = best_params[model_type][graph_identifier]
print(f"{model_type} | {graph_identifier} :: {best_params}")

train_graphs, test_graphs = get_preloaded_graphs(path=f"../../data/graphs_data/{'multi-graph/' if multi_graph else ''}{graphs_type}")

if layered:
    # Read layer embeddings
    layer_embeddings = pickle.load(open("layer_embeddings.pickle", "rb"))
    train_graphs, test_graphs = append_layer_embedding(train_graphs, test_graphs, layer_embeddings, multi_graph=multi_graph)

# Graph constants
num_classes = 1
num_features = test_graphs[0].num_features    
pos_weight = get_pos_weight(train_graphs=train_graphs)


## Training and testing
for seed in range(1, 11):
    
    print("Seed: ", seed)
    set_seed(seed)

    MLPmodel = MLP(num_features=num_features, hidden_channels=best_params["hidden_channels"], num_classes=num_classes, dropout=best_params["dropout"])
    MLPmodel = MLPmodel.to(device)  # move model to GPU

    # Optimizer
    MLP_optimizer = torch.optim.AdamW(MLPmodel.parameters(), lr=best_params['lr'])

    # Define weighted loss
    MLP_criterion = SoftF1Loss(pos_weight=pos_weight).to(device)
    #MLP_scheduler = torch.optim.lr_scheduler.StepLR(MLP_optimizer, step_size=10, gamma=0.5)
    MLP_scheduler = None

    print(MLPmodel)


    l, v = train(model=MLPmodel, train_graphs=train_graphs, optimizer=MLP_optimizer, criterion=MLP_criterion, \
                        scheduler=MLP_scheduler, epochs=500, batch_size=-1, patience=50, \
                            save_path=f"{model_type}/{graph_identifier}/{seed}", random_seed=seed, device=device)

    plot_train_curves(l, v, save_path=f"models/training/{model_type}/{graph_identifier}/{seed}", show=False)

    # Load best model from training for evaluation
    MLPmodel.load_state_dict(torch.load(f"./models/training/{model_type}/{graph_identifier}/{seed}/best_model.pt", weights_only=True))

    # Moving to CPU for evaluation
    MLPmodel = MLPmodel.to("cpu")
    MLP_criterion = MLP_criterion.to("cpu")

    evaluate(MLPmodel, test_graphs, save_path=f"{model_type}-{graph_identifier}-{seed}", show=False, device="cpu")