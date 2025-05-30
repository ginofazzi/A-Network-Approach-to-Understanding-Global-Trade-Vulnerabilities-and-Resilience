from torch_geometric.explain import Explainer, GNNExplainer
import torch.nn.functional as F
import json
import pickle

import numpy as np
import matplotlib.pyplot as plt

from gnnutils import *
from models import *


class ExplainModel(GAT):
    def __init__(self, num_features, num_classes, hidden_channels, heads, edge_dim, dropout, n_layers=2, residual=False, bias=False):
        super().__init__(num_features, num_classes, hidden_channels, heads, edge_dim, dropout, n_layers=2, residual=False, bias=False)


    def forward(self, x, edge_index, edge_attr):

        #x, edge_index, edge_attr = data.x, data.edge_index, data.edge_attr#torch.cat((data.edge_weight.unsqueeze(1), data.edge_attr), dim=1) # Pass weights + attrs
        
        if len(self.layers) == 1:  # Handle single-layer case
            x = self.layers[0](x=x, edge_index=edge_index, edge_attr=edge_attr)
        else:
            for layer in self.layers[:-1]:
                x = layer(x=x, edge_index=edge_index, edge_attr=edge_attr)
                x = F.leaky_relu(x)

            x = self.layers[-1](x=x, edge_index=edge_index, edge_attr=edge_attr)  # Last layer
        
        return x.view(-1) ## For binary


# Settings
model_type = "GAT"
graphs_type = "export" # "total", "export"
layered = False
multi_graph = False
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
    # Read layer embeddings
    layer_embeddings = pickle.load(open("layer_embeddings.pickle", "rb"))
    train_graphs, test_graphs = append_layer_embedding(train_graphs, test_graphs, layer_embeddings, multi_graph=multi_graph)


# Graph constants
num_classes = 1
num_features = test_graphs[0].num_features    
edge_dim = test_graphs[0].edge_attr.shape[1]
pos_weight = get_pos_weight(train_graphs=train_graphs)

# Create model
GATmodel = ExplainModel(num_features=num_features, num_classes=num_classes, hidden_channels=best_params['hidden_channels'],\
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

seed = np.random.randint(1, 11)

# Load best model from training for evaluation
GATmodel.load_state_dict(torch.load(f"./models/training/{model_type}/{graph_identifier}/{seed}/best_model.pt", weights_only=True, map_location=torch.device("cpu")))

# Get it to evaluation mode
GATmodel.eval()
device = next(GATmodel.parameters()).device
data = Batch.from_data_list(test_graphs).to(device) 
print(data.num_nodes)


# Explainer
explainer = Explainer(
    model=GATmodel.eval(),
    algorithm=GNNExplainer(epochs=200, lr=best_params["lr"]),
    explanation_type='model',
    node_mask_type='attributes',   # feature masks
    edge_mask_type='attributes',   
    model_config=dict(
        mode='binary_classification',
        task_level='node',
        return_type='raw',      # use 'logits' for raw scores
    ),
)

# Explain *all* nodes in one shot:
all_idx = torch.arange(data.num_nodes, device=device)
explanation   = explainer(
    x=data.x,
    edge_index=data.edge_index,
    edge_attr=data.edge_weight,
    index=all_idx
)

# 1) pull out the raw masks
node_masks      = explanation.node_mask.detach().cpu().numpy()    # (num_nodes, num_node_features)
edge_attr_masks = explanation.edge_mask.detach().cpu().numpy()    # (num_edges, num_edge_features)

# 2) compute mean *signed* importance per attribute
#global_node_imp_div      = node_masks.mean(axis=0)   # shape: (num_node_features,)
#global_edge_attr_imp_div = edge_attr_masks.mean(axis=0)  # shape: (num_edge_features,)

#global_node_imp_div = np.mean(np.abs(node_masks), axis=0)
#global_edge_attr_imp = np.mean(np.abs(edge_attr_masks), axis=0)


with open(f"{model_type}-{graph_identifier}-node_feat.pkl", "wb") as outfile:
    pickle.dump(node_masks, outfile)

with open(f"{model_type}-{graph_identifier}-edge_attr.pkl", "wb") as outfile:
    pickle.dump(edge_attr_masks, outfile)
