######### MULTI-EDGE GRAPHS #########

import torch
from torch_geometric.data import Data
from gnnutils import *

# Settings
graphs_type = "total" # "total", "export"

train_graphs, test_graphs = get_preloaded_graphs(path=f"../../data/graphs_data/{graphs_type}")

print(len(train_graphs))
yearly_graphs = {}

i = 0 # Pointer for the train_graphs list
for commodity in range(96): # 96 Commodity classes
    for year in range(2012, 2021): # 9 Years training data (2012-2020)
        if year not in yearly_graphs:
            yearly_graphs[year] = []
        yearly_graphs[year].append(train_graphs[i])
        i += 1

print(len(yearly_graphs))
print(len(yearly_graphs[2012]))
print(len(yearly_graphs[2020]))



def merge_graphs(graphs, years, layer_embeddings):
    """
    - train_graphs: list of Data objects, each with .edge_index and .edge_attr for a single commodity
    - country_features: Tensor [N, F] of node features for all countries
    """
    N, F = country_features.size()
    all_edge_index = []
    all_edge_attr  = []

    merged_graphs = [] # One big graph per year

    for y in years:

        for G in train_graphs:
            # G.edge_index: [2, Ei], G.edge_attr: [Ei, 8]
            all_edge_index.append(G.edge_index)
            all_edge_attr .append(G.edge_attr)

    # concatenate along the edge dimension
    edge_index = torch.cat(all_edge_index, dim=1)     # [2, E_total]
    edge_attr  = torch.cat(all_edge_attr , dim=0)     # [E_total, 8]

    data = Data(x=country_features, 
                edge_index=edge_index, 
                edge_attr=edge_attr)
    data.num_nodes = N
    return data
