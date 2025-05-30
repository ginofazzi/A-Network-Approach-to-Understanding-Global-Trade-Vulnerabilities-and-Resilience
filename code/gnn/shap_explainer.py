
# SHAP
import numpy as np
import shap
import matplotlib.pyplot as plt
import torch
import shap
from torch_geometric.data import Data
from torch_geometric.loader import DataLoader
from gnnutils import *
from models import *

# Settings
model_type = "GCN"
graphs_type = "total" # "total", "export"
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
pos_weight = get_pos_weight(train_graphs=train_graphs)

# Create model
GCNmodel = GCN(num_features=num_features, hidden_channels=best_params["hidden_channels"], num_classes=num_classes, \
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

seed = 10

# Load best model from training for evaluation
GCNmodel.load_state_dict(torch.load(f"./models/training/{model_type}/{graph_identifier}/{seed}/best_model.pt", weights_only=True, map_location=torch.device("cpu")))

# Get it to evaluation mode
GCNmodel.eval()
device = next(GCNmodel.parameters()).device
data = Batch.from_data_list(test_graphs).to(device) 

print(test_graphs)
print(data)
import sys;sys.exit()
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
model  = GCNmodel.to(device).eval()


# Your SHAP wrapper: takes array of shape (S, N, F) → returns (S, N)
def f(x_array):
    # x_array: (S, N, F)
    S, N, F = x_array.shape
    out = np.zeros((S, N), dtype=float)
    for i in range(S):
        x = torch.from_numpy(x_array[i]).to(device).float()
        d = Data(x=x,
                 edge_index=data.edge_index,
                 edge_weight=data.edge_weight).to(device)
        with torch.no_grad():
            logits = model(d)            # [N]
            probs  = torch.sigmoid(logits)
        out[i] = probs.cpu().numpy()
    return out                            # (S, N)
                             # shape (S, N)

# Build the SHAP explainer
data_np = data.x.cpu().numpy()                # shape: (N, F)
num_bg  = 10
bg      = np.stack([data_np.copy() for _ in range(num_bg)], axis=0)
masker = shap.maskers.Independent(bg, max_samples=50)
explainer = shap.Explainer(f, masker=masker)
X = data_np[None, ...]                   # shape: (1, N, F)
shap_exp = explainer(X)


# Explain on a single “batch” of 1 graph
X = data.x.cpu().numpy()[None, :, :]         # ← add batch dim
shap_explanation = explainer(X)

# Now shap_explanation.values has shape (1, N, F)
vals = shap_explanation.values[0]             # shape (N, F)

# If you want per-feature global importances:
global_imp = np.mean(np.abs(vals), axis=0)   # shape (F,)

# Plot
plt.bar(np.arange(F), global_imp)
plt.xlabel("Feature index")
plt.ylabel("Mean |SHAP value|")
plt.title("Global feature importance via SHAP")
plt.show()


