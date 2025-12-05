from functools import lru_cache
import rdflib.term
from collections import Counter
import pandas as pd
import networkx as nx
import netlsd
import utils
import random
import numpy as np
import torch
from sklearn.metrics import ndcg_score

@lru_cache
def graph_stat(G):
    nodes = list(G.nodes())
    entity_node_id = {node: i for i, node in enumerate(nodes)}
    relation_names = [G[u][v]['relation'] for u, v in G.edges()]
    relation_frequency = Counter(relation_names)
    relation_frequency = {relation: 1 / frequency for relation, frequency in relation_frequency.items()}
    return entity_node_id, relation_names, relation_frequency

@lru_cache(maxsize=None)
def import_top_summary(path, eid, i, k, targetentity):
    return utils.import_top_summary(path, eid, i, k, targetentity)

def average_precision(ground_truth_summary, predicted_summary):
    relevant_count = 0
    cumulative_precision = 0
    for i, summary in enumerate(predicted_summary):
        if summary in ground_truth_summary:
            relevant_count += 1
            precision_at_i = relevant_count / (i + 1)
            cumulative_precision += precision_at_i
    if relevant_count > 0:
        return cumulative_precision / len(ground_truth_summary)
    return 0

def edge_in_gold_set(edge, gold_triples_set):
    if len(edge) < 3: return False
    u, v, data = edge
    if 'relation' not in data: return False
    relation = data['relation']
    u_str = u.n3() if isinstance(u, rdflib.term.URIRef) else str(u)
    v_str = v.n3() if isinstance(v, rdflib.term.URIRef) else str(v)
    rel_str = relation.n3() if isinstance(relation, rdflib.term.URIRef) else str(relation)
    triple = (u_str, rel_str, v_str)
    return triple in gold_triples_set

def evaluate(path, entity_dataset, G, k, node_weights, z, dataset_name="unknown"):
    entity_node_id, relation_names, relation_frequency = graph_stat(G)
    
    results = []
    results_norel = []
    
    for index, row in entity_dataset.iterrows():
        targetentity = f'<{row["euri"]}>'
        node_id = entity_node_id.get(targetentity)
        if node_id is None: continue
            
        out_edges = list(G.out_edges(targetentity, data=True))
        all_edges = []
        for edge in out_edges:
            other_node = edge[1] if edge[0] == targetentity else edge[0]
            weight = node_weights[node_id][entity_node_id[other_node]].item()
            all_edges.append((edge, weight))
            
        all_edges.sort(key=lambda x: x[1], reverse=True)
        if all_edges: weight_portion = all_edges[-1][1]
        else: weight_portion = 0
            
        for edge in G.in_edges(targetentity, data=True):
            other_node = edge[1] if edge[0] == targetentity else edge[0]
            weight = node_weights[node_id][entity_node_id[other_node]].item() + weight_portion
            all_edges.append((edge, weight))

        all_edges.sort(key=lambda x: x[1], reverse=True)
        out_edges_sorted = [edge for edge, weight in all_edges]
        
        # Selection Logic
        summary_edges = []
        added_communities = set()
        
        for edge in out_edges_sorted:
            other_node = edge[1]
            other_node_id = entity_node_id[other_node]
            other_communities = torch.topk(z[other_node_id], 1).indices.tolist()
            if not set(other_communities).issubset(added_communities):
                summary_edges.append(edge)
                added_communities.update(set(other_communities))
                
        if len(summary_edges) < k:
            for edge in out_edges_sorted:
                if len(summary_edges) >= k: break
                if edge not in summary_edges: summary_edges.append(edge)

        summary_edges = summary_edges[:k]
        
        # Metrics Prep
        pval = []
        for edge in summary_edges:
            relation = edge[2]['relation']
            tsumm = (edge[0], relation, edge[1])
            pval.append(tuple(ts.n3() if type(ts) == rdflib.term.URIRef else ts for ts in tsumm))
        
        gold_triples_set = set()
        gold_summaries_list = []
        for i in range(6): 
            tval = import_top_summary(path, index, i, k, targetentity)
            if not tval: continue
            gold_summaries_list.append(tval)
            for triple in tval: gold_triples_set.add(triple)

        # Calculate per-entity Metrics
        fscores_this_entity = []
        ap_scores_this_entity = []
        for tval in gold_summaries_list:
            fscores_this_entity.append(utils.fmeasure_score(tval, set(pval)))
            ap_scores_this_entity.append(average_precision(tval, set(pval)))
            
        final_fscore = np.mean(fscores_this_entity) if fscores_this_entity else 0.0
        final_ap = np.mean(ap_scores_this_entity) if ap_scores_this_entity else 0.0

        y_score_all = [w for _, w in all_edges]
        y_true_all = [1 if edge_in_gold_set(e, gold_triples_set) else 0 for e, _ in all_edges]
        
        if len(y_true_all) > 1 and sum(y_true_all) > 0:
            ndcg_val = ndcg_score([y_true_all], [y_score_all], k=k)
        else:
            ndcg_val = 0.0

        results.append({
            'euri': targetentity, 
            'fmeasure': final_fscore,
            'ave_precision': final_ap, 
            'ndcg': ndcg_val
        })
        
        # NoRel Metrics
        pval_norel = {(item[0], item[-1]) for item in pval}
        fscores_norel = []
        ap_norel_list = []
        for tval in gold_summaries_list:
            tval_norel = {(item[0], item[-1]) for item in tval}
            fscores_norel.append(utils.fmeasure_score(tval_norel, set(pval_norel)))
            ap_norel_list.append(average_precision(tval_norel, set(pval_norel)))
            
        final_fscore_norel = np.mean(fscores_norel) if fscores_norel else 0.0
        final_ap_norel = np.mean(ap_norel_list) if ap_norel_list else 0.0
        
        results_norel.append({
            'euri': targetentity, 
            'fmeasure_norel': final_fscore_norel,
            'ave_precision_norel': final_ap_norel
        })
    
    # --- SAVE DETAILED SCORES ---
    results_df = pd.DataFrame(results)
    results_norel_df = pd.DataFrame(results_norel)
    
    # Merge relevant columns for the detailed CSV
    detailed_df = pd.DataFrame({
        'entity': results_df['euri'],
        'fmeasure_rel': results_df['fmeasure'],
        'fmeasure_norel': results_norel_df['fmeasure_norel'],
        'ndcg': results_df['ndcg'],
        'map_rel': results_df['ave_precision'],
        'map_norel': results_norel_df['ave_precision_norel']
    })
    
    # Save to a separate CSV file with dataset name included
    detailed_filename = f"detailed_scores_{dataset_name}_top{k}.csv"
    detailed_df.to_csv(detailed_filename, index=False)
    print(f">> Detailed scores for Top-{k} saved to {detailed_filename}")
    # ---------------------------------------------

    avg_fmeasure = results_df['fmeasure'].mean()
    avg_fmeasure_norel = results_norel_df['fmeasure_norel'].mean()
    avg_ap = results_df['ave_precision'].mean()
    avg_ap_norel = results_norel_df['ave_precision_norel'].mean()
    avg_ndcg = results_df['ndcg'].mean()
    
    return (avg_fmeasure, avg_fmeasure_norel, avg_ap, avg_ap_norel, avg_ndcg)