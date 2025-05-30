import pandas as pd
import json
import numpy as np
from gnnutils import *


# Generate the results for ENSEMBLE model, by majority voting from other GNN models

# For each graph typy
for graph_type in ["export", "total", "export-layered", "multi-graph-total", "multi-graph-export"]:

    results = pd.DataFrame()

    # For each seed
    for seed in range(1, 11):

        seed_results = pd.DataFrame()

        # For each GNN model
        for model in ["GCN", "GAT", "SAGE"]:
            _ = json.load(open(f"./results/{model}/{graph_type}/{model}-{graph_type}-{seed}-predictions.json"))["predictions"]
            _ = pd.DataFrame(_)
            _.columns = [model]
            seed_results = pd.concat([seed_results, _], axis=1) # Concatenate one column per model

        # Add the seed to keep track
        seed_results["SEED"] = seed
        # Add the true label
        seed_results["LABEL"] = json.load(open(f"./results/{model}/{graph_type}/{model}-{graph_type}-{seed}-predictions.json"))["labels"]
        results = pd.concat([results, seed_results])

    # Add the majority vote for ENSEMBLE
    results["ENSEMBLE"] = results[["GCN", "GAT", "SAGE"]].sum(axis=1)
    results["ENSEMBLE"] = results.ENSEMBLE.apply(lambda x: 1 if x > 1 else 0)

    # Calculate the result metrics for each seed
    for seed in range(1,11):

        results_seed = results[results.SEED == seed]

        acc = accuracy_score(results_seed["LABEL"], results_seed["ENSEMBLE"])
        f1_macro_avg = f1_score(results_seed["LABEL"], results_seed["ENSEMBLE"], average="macro")
        f1_pos = f1_score(results_seed["LABEL"], results_seed["ENSEMBLE"], pos_label=1, average="binary")

        d = classification_report(results_seed["LABEL"], results_seed["ENSEMBLE"], output_dict=True)
        
        d["Accuracy"] = acc
        d["F1 - Avg. Macro"] = f1_macro_avg
        d["F1 - Positives"] = f1_pos
        

        with open(f"./results/ENSEMBLE/{graph_type}/ENSEMBLE-{graph_type}-{seed}-report.json", "w") as file:
            json.dump(d, file, indent=4)