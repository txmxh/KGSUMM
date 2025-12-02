import math 
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
import json
import os

@lru_cache
def graph_stat(G):
    # Create a dictionary that maps matrix indices to nodes.
    nodes = list(G.nodes())
    entity_node_id = {node: i for i, node in enumerate(nodes)}
    relation_names = [G[u][v]['relation'] for u, v in G.edges()]
    relation_frequency = Counter(relation_names)
    relation_frequency = {relation: 1 / frequency for relation, frequency in relation_frequency.items()}
    return entity_node_id, relation_names, relation_frequency


@lru_cache(maxsize=None)
def import_top_summary(path, eid, i, k, targetentity):
    return utils.import_top_summary(path, eid, i, k, targetentity)

def calculate_neighbor_community_sizes(G, z, entity_node_id, k):
    community_sizes = {}
    for node in list(G.nodes()):
        neighbor_nodes = list(G.neighbors(node))
        community_counts = {}
        for neighbor in neighbor_nodes:
            idx = entity_node_id[neighbor]
            top_k_communities = torch.topk(z[idx], k, largest=True).indices.tolist()
            for community in community_counts:
                if community in community_counts:
                    community_counts[community] += 1
                else:
                    community_counts[community] = 1
        community_sizes[node] = community_counts
    return community_sizes


def calculate_entropy(z):
    z = torch.sigmoid(z) 
    z_safe = torch.where(z == 0, torch.tensor(1e-10), z)
    entropy = -torch.sum(z_safe * torch.log(z_safe), dim=1)
    return entropy

def average_precision(ground_truth_summary, predicted_summary_list):
    """
    Calculates Average Precision for a single entity based on a ranked list.
    """
    relevant_count = 0
    cumulative_precision = 0
    
    for i, summary in enumerate(predicted_summary_list):
        if summary in ground_truth_summary:
            relevant_count += 1
            precision_at_i = relevant_count / (i + 1)
            cumulative_precision += precision_at_i

    if len(ground_truth_summary) > 0:
        average_precision = cumulative_precision / len(ground_truth_summary)
    else:
        average_precision=0
    return average_precision

# --- NDCG Calculation ---
def ndcg_score(ground_truth_summary, predicted_summary, k):
    """
    Calculates NDCG@k for a single entity. Relevance is binary (1 if present, 0 otherwise).
    """
    if not ground_truth_summary:
        return 0.0

    # 1. Calculate DCG (Discounted Cumulative Gain)
    dcg = 0.0
    for i in range(min(k, len(predicted_summary))):
        triple = predicted_summary[i]
        relevance = 1 if triple in ground_truth_summary else 0
        dcg += relevance / math.log2(i + 2)

    # 2. Calculate IDCG (Ideal Discounted Cumulative Gain)
    idcg = 0.0
    num_relevant = len(ground_truth_summary)
    
    for i in range(min(k, num_relevant)):
        idcg += 1 / math.log2(i + 2)

    # 3. Calculate NDCG
    if idcg == 0:
        return 0.0
    
    ndcg = dcg / idcg
    return ndcg


def evaluate(path, entity_dataset, G, k, node_weights, z, dataset_name="esbm"):
    """
    Evaluate by finding summaries for entities in a dataset and calculating F-score, MAP, and NDCG.
    """
    
    entity_node_id, relation_names, relation_frequency = graph_stat(G)

    results = []
    results_norel = []
    results_ndcg = [] 

    skipped_entities = 0
    processed_entities = 0
    
    # --- FACES Ground Truth Loading: FIXED PATH ---
    faces_ground_truth = None
    if dataset_name == 'faces':
        # ABSOLUTE PATH FIX
        gt_path = '/content/KGSUMM/Task1/Unsupervised_Learning_IRES_Model/Temp/faces_groundtruth.json'
        
        try:
            with open(gt_path, 'r') as f:
                faces_ground_truth_raw = json.load(f)
            
            faces_ground_truth = {}
            for k_raw, v in faces_ground_truth_raw.items():
                k_norm = k_raw.strip()
                if k_norm.startswith('<') and k_norm.endswith('>'):
                    k_norm = k_norm[1:-1]
                
                gt_set = set()
                for triple in v:
                    gt_set.add(tuple(triple))
                faces_ground_truth[k_norm] = gt_set
                
        except FileNotFoundError:
            print(f"[evaluate] ERROR: Cannot find FACES ground truth at {gt_path}. Ensure it was generated.")
            faces_ground_truth = {}
        except Exception as e:
            print(f"[evaluate] ERROR: Loading FACES ground truth failed: {e}")
            faces_ground_truth = {}
    # --- End Fixed Loading ---

    def make_targetentity(euri_value):
        if not isinstance(euri_value, str):
            euri_value = str(euri_value)
        euri_value = euri_value.strip()
        if euri_value.startswith('<') and euri_value.endswith('>'):
            return euri_value
        return f'<{euri_value}>'

    for index, row in entity_dataset.iterrows():
        if 'euri' not in row or pd.isna(row['euri']):
            skipped_entities += 1
            continue

        targetentity_bare = row['euri'].strip()
        targetentity_wrapped = make_targetentity(targetentity_bare)

        node_id = entity_node_id.get(targetentity_bare)
        if node_id is not None:
            targetentity = targetentity_bare
        else:
            node_id = entity_node_id.get(targetentity_wrapped)
            if node_id is not None:
                targetentity = targetentity_wrapped
            else:
                skipped_entities += 1
                continue

        processed_entities += 1
        
        out_edges = list(G.out_edges(targetentity, data=True))

        if not out_edges:
            if dataset_name == 'faces':
                results.append({'eid': index, 'euri': targetentity, 'fmeasure': 0.0, 'ave_precision': 0.0, 'Top': k})
                results_norel.append({'eid': index, 'euri': targetentity, 'fmeasure_norel': 0.0, 'ave_precision_norel': 0.0, 'Top': k})
                results_ndcg.append({'eid': index, 'euri': targetentity, 'ndcg': 0.0, 'Top': k})
            else:
                for i in range(6):
                    results.append({'eid': index, 'euri': targetentity, 'fmeasure': 0.0, 'ave_precision': 0.0, 'Top': k})
                    results_norel.append({'eid': index, 'euri': targetentity, 'fmeasure_norel': 0.0, 'ave_precision_norel': 0.0, 'Top': k})
                    results_ndcg.append({'eid': index, 'euri': targetentity, 'ndcg': 0.0, 'Top': k})
            continue


        random.shuffle(out_edges)
        all_edges = []
        for edge in out_edges:
            other_node = edge[1] if edge[0] == targetentity else edge[0]
            if other_node not in entity_node_id:
                continue
            try:
                weight = node_weights[node_id][entity_node_id[other_node]].item()
            except Exception:
                try:
                    weight = float(node_weights[node_id][entity_node_id[other_node]])
                except Exception:
                    weight = 0.0
            all_edges.append((edge, weight))

        if not all_edges:
             if dataset_name == 'faces':
                results.append({'eid': index, 'euri': targetentity, 'fmeasure': 0.0, 'ave_precision': 0.0, 'Top': k})
                results_norel.append({'eid': index, 'euri': targetentity, 'fmeasure_norel': 0.0, 'ave_precision_norel': 0.0, 'Top': k})
                results_ndcg.append({'eid': index, 'euri': targetentity, 'ndcg': 0.0, 'Top': k})
             else:
                for i in range(6):
                    results.append({'eid': index, 'euri': targetentity, 'fmeasure': 0.0, 'ave_precision': 0.0, 'Top': k})
                    results_norel.append({'eid': index, 'euri': targetentity, 'fmeasure_norel': 0.0, 'ave_precision_norel': 0.0, 'Top': k})
                    results_ndcg.append({'eid': index, 'euri': targetentity, 'ndcg': 0.0, 'Top': k})
             continue
             
        try:
            for edge in G.in_edges(targetentity, data=True):
                other_node = edge[1] if edge[0] == targetentity else edge[0]
                if other_node not in entity_node_id:
                    continue
                try:
                    weight = node_weights[node_id][entity_node_id[other_node]].item()
                except Exception:
                    weight = float(node_weights[node_id][entity_node_id[other_node]])
                all_edges.append((edge, weight))
        except Exception:
            pass

        all_edges.sort(key=lambda x: x[1], reverse=True)
        out_edges_sorted = [edge for edge, _ in all_edges]

        summary_edges = []
        added_communities = set()
        for edge in out_edges_sorted:
            other_node = edge[1]
            if other_node not in entity_node_id:
                continue
            other_node_id = entity_node_id[other_node]
            if other_node_id >= z.shape[0]:
                continue
            other_communities = torch.topk(z[other_node_id], 1).indices.tolist()
            if not set(other_communities).issubset(added_communities):
                summary_edges.append(edge)
                added_communities.update(set(other_communities))
            if len(summary_edges) >= k:
                break

        if len(summary_edges) < k:
            for edge in out_edges_sorted:
                if len(summary_edges) >= k:
                    break
                if edge not in summary_edges:
                    summary_edges.append(edge)

        summary_edges = summary_edges[:k]

        # --- Prepare Ranked LIST (S, P, O) and Unranked SET (S, P, O) ---
        pval_list = [] # RANKED LIST for MAP/NDCG
        for edge in summary_edges:
            relation = edge[2].get('relation') if isinstance(edge[2], dict) else edge[2]['relation']
            s = edge[0].strip().strip('<>')
            p = relation.strip().strip('<>')
            o = edge[1].strip().strip('<>')
            pval_list.append((s, p, o))
        
        pval_set = set(pval_list) # UNRANKED SET for F-Score
        
        # --- Prepare Ranked LIST (S, O) for NoRel Metrics ---
        pval_norel_list = [(item[0], item[2]) for item in pval_list]


        if dataset_name == 'faces':
            target_key = targetentity.strip('<>')
            tval = faces_ground_truth.get(target_key, set())
            
            if not tval:
                tval = faces_ground_truth.get(f"<{target_key}>", set())

            # MAP Rel Calculation (uses full triples)
            fscore = utils.fmeasure_score(tval, pval_set)
            Ap = average_precision(tval, pval_list) 
            ndcg = ndcg_score(tval, pval_list, k) 
            
            results.append({'eid': index, 'euri': targetentity, 'fmeasure': fscore, 'ave_precision': Ap, 'Top': k})
            results_ndcg.append({'eid': index, 'euri': targetentity, 'ndcg': ndcg, 'Top': k})

            # MAP NoRel Calculation (uses S, O pairs)
            tval_norel = {(item[0], item[-1]) for item in tval}
            fscore_norel = utils.fmeasure_score(tval_norel, set(pval_norel_list)) 
            Ap_norel = average_precision(tval_norel, pval_norel_list) 
            results_norel.append({'eid': index, 'euri': targetentity, 'fmeasure_norel': fscore_norel, 'ave_precision_norel': Ap_norel, 'Top': k})

        else:
            # ESBM evaluation logic (unchanged, but includes NDCG/MAP updates)
            for i in range(6):
                tval_set = import_top_summary(path, index, i, k, targetentity)
                tval_norel = {(item[0], item[-1]) for item in tval_set}
                
                # Rel metrics
                fscore = utils.fmeasure_score(tval_set, pval_set)
                Ap = average_precision(tval_set, pval_list) 
                ndcg = ndcg_score(tval_set, pval_list, k) 

                results.append({'eid': index, 'euri': targetentity, 'fmeasure': fscore, 'ave_precision': Ap, 'Top': k})
                results_ndcg.append({'eid': index, 'euri': targetentity, 'ndcg': ndcg, 'Top': k})
                
                # NoRel metrics
                fscore_norel = utils.fmeasure_score(tval_norel, set(pval_norel_list))
                Ap_norel = average_precision(tval_norel, pval_norel_list) 
                
                results_norel.append({'eid': index, 'euri': targetentity, 'fmeasure_norel': fscore_norel, 'ave_precision_norel': Ap_norel, 'Top': k})

    results_df = pd.DataFrame(results)
    results_norel_df = pd.DataFrame(results_norel)
    results_ndcg_df = pd.DataFrame(results_ndcg)

    if results_df.empty:
        print(f"[evaluate] No results computed. processed_entities={processed_entities}, skipped_entities={skipped_entities}")
        # MUST return 8 values now
        return 0.0, 0.0, 0.0, 0.0, 0.0, results_df, results_norel_df, results_ndcg_df

    if 'euri' not in results_df.columns:
        results_df['euri'] = results_df.get('euri', results_df.get('eid', 0)).astype(str)
    if 'euri' not in results_norel_df.columns:
        results_norel_df['euri'] = results_norel_df.get('euri', results_norel_df.get('eid', 0)).astype(str)
    if 'euri' not in results_ndcg_df.columns:
        results_ndcg_df['euri'] = results_ndcg_df.get('euri', results_ndcg_df.get('eid', 0)).astype(str)


    averages = results_df.groupby('euri')['fmeasure'].mean()
    averages_norel = results_norel_df.groupby('euri')['fmeasure_norel'].mean()
    averages_ap = results_df.groupby('euri')['ave_precision'].mean()
    averages_norel_ap = results_norel_df.groupby('euri')['ave_precision_norel'].mean()
    averages_ndcg = results_ndcg_df.groupby('euri')['ndcg'].mean()


    # FINAL RETURN: 5 mean values + 3 detailed DataFrames
    return (averages.mean(), averages_norel.mean(), averages_ap.mean(), averages_norel_ap.mean(), averages_ndcg.mean(), 
            results_df, results_norel_df, results_ndcg_df)