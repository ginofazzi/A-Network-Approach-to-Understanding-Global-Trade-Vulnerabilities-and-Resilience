from torch_geometric.explain import Explainer, GNNExplainer
import torch.nn.functional as F
import json
import pickle

import numpy as np
import matplotlib.pyplot as plt

from gnnutils import *
from models import *


class ExplainModel(GCN):
    def __init__(self, num_features, hidden_channels, num_classes, dropout=None, n_layers=2):
        super().__init__(num_features, hidden_channels, num_classes, dropout=None, n_layers=2)

    def forward(self, x, edge_index, edge_weight=None):

        if len(self.layers) == 1:  # Handle single-layer case
            x = self.layers[0](x, edge_index, edge_weight=edge_weight)
        else:
            for layer in self.layers[:-1]:
                x = layer(x, edge_index, edge_weight=edge_weight)
                x = x.relu()
                if self.dropout:
                    x = F.dropout(x, p=self.dropout, training=self.training)

            x = self.layers[-1](x, edge_index, edge_weight=edge_weight)  # Last layer
            if self.dropout:
                x = F.dropout(x, p=self.dropout, training=self.training)

        return x.view(-1)  # For binary classification


# Settings
model_type = "GCN"
graphs_type = "export" # "total", "export"
layered = False
multi_graph = False
graph_identifier = f"{'multi-graph-' if multi_graph else ''}{graphs_type}{'-layered' if (layered and not multi_graph) else ''}"
explain_type = "phenomenon" # "phenomenon", "model"

# GPU if possible
device = "cpu"#torch.device('cuda' if torch.cuda.is_available() else 'cpu')
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
pos_weight = get_pos_weight(train_graphs=train_graphs)

# Create model
GCNmodel = ExplainModel(num_features=num_features, hidden_channels=best_params["hidden_channels"], num_classes=num_classes, \
            n_layers=best_params["n_layers"], dropout=best_params["dropout"])
GCNmodel = GCNmodel.to(device)  # move model to GPU

# Optimizer
if best_params['optimizer'] == "Adam":
    GCN_optimizer = torch.optim.Adam(GCNmodel.parameters(), lr=best_params['lr'], weight_decay=best_params['weight_decay'])
elif best_params['optimizer'] == "AdamW":
    GCN_optimizer = torch.optim.AdamW(GCNmodel.parameters(), lr=best_params['lr'], weight_decay=best_params['weight_decay'])
elif best_params['optimizer'] == "SGD":
    GCN_optimizer = torch.optim.SGD(GCNmodel.parameters(), lr=best_params['lr'], weight_decay=best_params['weight_decay'],\
                                    momentum=best_params['momentum'])
elif best_params['optimizer'] == "RMSprop":
    GCN_optimizer = torch.optim.RMSprop(GCNmodel.parameters(), lr=best_params['lr'], weight_decay=best_params['weight_decay'],\
                                            momentum=best_params['momentum'])
elif best_params['optimizer'] == "Adagrad":
    GCN_optimizer = torch.optim.Adagrad(GCNmodel.parameters(), lr=best_params['lr'], weight_decay=best_params['weight_decay'])

# Define weighted loss
GCN_criterion = SoftF1Loss(pos_weight=pos_weight).to(device)
#GCN_scheduler = torch.optim.lr_scheduler.StepLR(GCN_optimizer, step_size=10, gamma=0.5)
GCN_scheduler = None

print(GCNmodel)

seed = np.random.randint(1, 11)

# Load best model from training for evaluation
GCNmodel.load_state_dict(torch.load(f"./models/training/{model_type}/{graph_identifier}/{seed}/best_model.pt", weights_only=True, map_location=torch.device("cpu")))

# Get it to evaluation mode
GCNmodel.eval()
device = next(GCNmodel.parameters()).device
data = Batch.from_data_list(test_graphs).to(device) 
print(data.num_nodes)

# 1) Get your predicted labels (or use ground-truth for phenomenon)
with torch.no_grad():
    logits = GCNmodel(data.x, data.edge_index, data.edge_weight)
    probs = torch.sigmoid(logits)  # For binary; Convert to probabilities
    preds  = (probs > 0.5).long()

# 2) Select only the positive nodes
pos_idx = (preds == 1).nonzero(as_tuple=True)[0]  # Tensor of node-indices

model_config=dict(
        mode='binary_classification',
        task_level='node',
        return_type='raw',      # use 'logits' for raw scores
    )

# Explainer
explainer = Explainer(
    model=GCNmodel.eval(),
    algorithm=GNNExplainer(epochs=300, lr=best_params["lr"]),
    explanation_type=explain_type,
    node_mask_type='attributes',   # feature masks
    edge_mask_type='object',   
    model_config=model_config
)

# Explain *all* nodes in one shot:
#all_idx = torch.arange(data.num_nodes, device=device)

if explain_type == "phenomenon":
    explanation = explainer(
        x=data.x,
        edge_index=data.edge_index,
        edge_weight=data.edge_weight,
        index=pos_idx,
        target=data.y#[pos_idx]
    )
else:
    explanation = explainer(
        x=data.x,
        edge_index=data.edge_index,
        edge_weight=data.edge_weight,
        index=pos_idx,
    )

print(f'Generated explanations in {explanation.available_explanations}')

path = f'{model_type}-{explain_type}-feature_importance.png'
explanation.visualize_feature_importance(path, top_k=10)
print(f"Feature importance plot has been saved to '{path}'")

path = f'{model_type}-{explain_type}-subgraph.pdf'
explanation.visualize_graph(path)
print(f"Subgraph visualization plot has been saved to '{path}'")

# 1) pull out the raw masks
node_masks      = explanation.node_mask.detach().cpu().numpy()    # (num_nodes, num_node_features)
edge_attr_masks = explanation.edge_mask.detach().cpu().numpy()    # (num_edges, num_edge_features)

with open(f"{model_type}-{explain_type}-{graph_identifier}-node_feat.pkl", "wb") as outfile:
    pickle.dump(node_masks, outfile)

with open(f"{model_type}-{explain_type}-{graph_identifier}-edge_attr.pkl", "wb") as outfile:
    pickle.dump(edge_attr_masks, outfile)
