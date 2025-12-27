from functools import lru_cache
from collections import Counter
import pandas as pd
import random
import torch
import utils
import os


# -------------------- HELPERS --------------------

def normalize(x):
    return str(x).strip("<>")


def entities_from_triples(triples):
    """Extract entity set from triples"""
    ents = set()
    for s, _, o in triples:
        ents.add(normalize(s))
        ents.add(normalize(o))
    return ents


def fscore_entities(gt_ents, pred_ents):
    if not gt_ents or not pred_ents:
        return 0.0
    tp = len(gt_ents & pred_ents)
    prec = tp / len(pred_ents)
    rec = tp / len(gt_ents)
    if prec + rec == 0:
        return 0.0
    return 2 * prec * rec / (prec + rec)


def average_precision_entities(gt_ents, pred_ranked):
    score = 0.0
    hits = 0
    for i, ent in enumerate(pred_ranked):
        if ent in gt_ents:
            hits += 1
            score += hits / (i + 1)
    return score / len(gt_ents) if hits > 0 else 0.0


@lru_cache
def graph_stat(G):
    nodes = list(G.nodes())
    return {n: i for i, n in enumerate(nodes)}


@lru_cache(maxsize=None)
def import_top_summary(path, eid, i, k, targetentity):
    return utils.import_top_summary(path, eid, i, k, targetentity)


# -------------------- MAIN EVALUATION --------------------

def evaluate(path, entity_dataset, G, k, node_weights, z):

    entity_node_id = graph_stat(G)

    results = []
    
    print("DEBUG: Starting Evaluation Loop...")

    for index, row in entity_dataset.iterrows():

        targetentity = normalize(row["euri"])
        if targetentity not in entity_node_id:
            continue

        node_id = entity_node_id[targetentity]

        edges = list(G.out_edges(targetentity, data=True)) + \
                list(G.in_edges(targetentity, data=True))

        if not edges:
            continue

        # ---- Rank edges by learned weights ----
        scored = []
        for u, v, d in edges:
            other = normalize(v if u == targetentity else u)
            if other not in entity_node_id:
                continue
            
            # Safe Weight Access with bounds checking
            u_idx = entity_node_id[targetentity]
            v_idx = entity_node_id[other]
            
            if u_idx < node_weights.shape[0] and v_idx < node_weights.shape[1]:
                w = node_weights[u_idx][v_idx].item()
                scored.append((other, w))

        scored.sort(key=lambda x: x[1], reverse=True)

        # ---- Predicted entity ranking ----
        pred_entities_ranked = []
        for ent, _ in scored:
            if ent not in pred_entities_ranked:
                pred_entities_ranked.append(ent)
            if len(pred_entities_ranked) >= k:
                break

        pred_entities = set(pred_entities_ranked)

        if not pred_entities:
            continue

        # ---- Compare with GT ----
        # Loop reduced to 1 since we are using dynamic single ground truth from utils
        for i in range(1):
            gt_triples = import_top_summary(path, index, i, k, targetentity)
            gt_entities = entities_from_triples(gt_triples)

            f = fscore_entities(gt_entities, pred_entities)
            ap = average_precision_entities(gt_entities, pred_entities_ranked)

            results.append({
                "eid": index,
                "entity": targetentity,
                "fscore": f,
                "ap": ap
            })

    df = pd.DataFrame(results)

    if df.empty:
        print("WARNING: Empty evaluation results")
        return 0.0, 0.0, 0.0, 0.0

    f_mean = df["fscore"].mean()
    ap_mean = df["ap"].mean()

    # --- SAVE RESULTS TO FILE (NEW BLOCK) ---
    output_dir = "./results"
    os.makedirs(output_dir, exist_ok=True)
    
    # 1. Save detailed per-entity scores
    output_csv = os.path.join(output_dir, "detailed_results.csv")
    df.to_csv(output_csv, index=False)
    print(f"SUCCESS: Saved detailed results to {output_csv}")
    
    # 2. Save summary table
    summary_file = os.path.join(output_dir, "summary_report.txt")
    with open(summary_file, "w") as f:
        f.write("Evaluation Report\n")
        f.write("=================\n")
        f.write(f"Model: IRES (Unsupervised)\n")
        f.write(f"Entities Evaluated: {len(df)}\n\n")
        f.write(f"Metric       | Score\n")
        f.write(f"-------------|------\n")
        f.write(f"F-Measure @{k} | {f_mean:.4f}\n")
        f.write(f"MAP @{k}       | {ap_mean:.4f}\n")
    print(f"SUCCESS: Saved summary report to {summary_file}")
    # ----------------------------------------

    # return same values to keep main.py unchanged
    return f_mean, f_mean, ap_mean, ap_mean