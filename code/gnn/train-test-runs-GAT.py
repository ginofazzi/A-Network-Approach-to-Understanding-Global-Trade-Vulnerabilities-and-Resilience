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
model_type = "GAT"
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

# Load graphs
print("Looking for pre-loaded graphs...")
train_graphs, test_graphs = get_preloaded_graphs(path=f"../../data/graphs_data/{'multi-graph/' if multi_graph else ''}{graphs_type}")
print("Found pre-loaded graphs!")

if layered:
    print("Adding layer embeddings...")
    # Read layer embeddings
    layer_embeddings = pickle.load(open("layer_embeddings.pickle", "rb"))
    train_graphs, test_graphs = append_layer_embedding(train_graphs, test_graphs, layer_embeddings, multi_graph=multi_graph)
    print(f"New layer shape: {train_graphs[0].x.shape}")

    

# Graph constants
num_classes = 1
num_features = test_graphs[0].num_features    
pos_weight = get_pos_weight(train_graphs=train_graphs)
edge_dim = test_graphs[0].edge_attr.shape[1]

# Training and Testing seed-wise
for seed in range(1, 11):
    
    print("Seed: ", seed)
    set_seed(seed)

    GATmodel = GAT(num_features=num_features, num_classes=num_classes, hidden_channels=best_params['hidden_channels'],\
                    heads=best_params['heads'], dropout=best_params['dropout'], n_layers=best_params['n_layers'], residual=best_params['residual'], \
                            bias=best_params['bias'], edge_dim=edge_dim)
    GATmodel = GATmodel.to(device)  # move model to GPU

    # Optimizer
    if best_params['optimizer'] == "Adam":
        GAT_optimizer = torch.optim.Adam(GATmodel.parameters(), lr=best_params['lr'], weight_decay=best_params['weight_decay'])
    elif best_params['optimizer'] == "AdamW":
        GAT_optimizer = torch.optim.AdamW(GATmodel.parameters(), lr=best_params['lr'], weight_decay=best_params['weight_decay'])
    elif best_params['optimizer'] == "SGD":
        GAT_optimizer = torch.optim.SGD(GATmodel.parameters(), lr=best_params['lr'], weight_decay=best_params['weight_decay'],\
                                        momentum=best_params['momentum'])
    elif best_params['optimizer'] == "RMSprop":
        GAT_optimizer = torch.optim.RMSprop(GATmodel.parameters(), lr=best_params['lr'], weight_decay=best_params['weight_decay'],\
                                                momentum=best_params['momentum'])
    elif best_params['optimizer'] == "Adagrad":
        GAT_optimizer = torch.optim.Adagrad(GATmodel.parameters(), lr=best_params['lr'], weight_decay=best_params['weight_decay'])

    # Define weighted loss
    #GAT_criterion = SoftF1Loss(pos_weight=pos_weight).to(device)
    GAT_criterion = torch.nn.BCEWithLogitsLoss().to(device)
    #GAT_scheduler = torch.optim.lr_scheduler.StepLR(GAT_optimizer, step_size=10, gamma=0.5)
    GAT_scheduler = None # Not use it for now

    print(GATmodel)


    l, v = train(model=GATmodel, train_graphs=train_graphs, optimizer=GAT_optimizer, criterion=GAT_criterion, \
                        scheduler=GAT_scheduler, epochs=500, batch_size=-1, patience=50, \
                            save_path=f"{model_type}/{graph_identifier}/{seed}", random_seed=seed, device=device, retain_graph=True)

    plot_train_curves(l, v, save_path=f"models/training/{model_type}/{graph_identifier}/{seed}", show=False)

    # Load best model from training for evaluation
    GATmodel.load_state_dict(torch.load(f"./models/training/{model_type}/{graph_identifier}/{seed}/best_model.pt", weights_only=True))

    # Moving to CPU for evaluation
    GATmodel = GATmodel.to("cpu")
    GAT_criterion = GAT_criterion.to("cpu")

    evaluate(GATmodel, test_graphs, save_path=f"{model_type}-{graph_identifier}-{seed}", show=False, device="cpu")