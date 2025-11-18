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
            for community in top_k_communities:
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

def average_precision(ground_truth_summary,predicted_summary):
    relevant_count = 0
    cumulative_precision = 0
    for i, summary in enumerate(predicted_summary):
        if summary in ground_truth_summary:
            relevant_count += 1
            precision_at_i = relevant_count / (i + 1)
            cumulative_precision += precision_at_i

    if relevant_count > 0:
        average_precision = cumulative_precision / len(ground_truth_summary)
    else:
        average_precision=0
    return average_precision

def evaluate(path, entity_dataset, G, k, node_weights, z, dataset_name="esbm"):
    """
    Evaluate by finding summaries for entities in a dataset and calculating F-score.
    """
    
    entity_node_id, relation_names, relation_frequency = graph_stat(G)

    results = []
    results_norel = []

    skipped_entities = 0
    processed_entities = 0
    
    # --- NEW FACES Ground Truth Loading (Using Absolute Path) ---
    faces_ground_truth = None
    if dataset_name == 'faces':
        # Use the absolute path where generate_and_verify.py saved the file
        gt_path = '/content/KGSUMM/Task1/Unsupervised_Learning_IRES_Model/Temp/faces_groundtruth.json'
        
        try:
            with open(gt_path, 'r') as f:
                faces_ground_truth_raw = json.load(f)
            
            # Normalize keys and convert values to sets of tuples
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
            print(f"[evaluate] ERROR: Cannot find FACES ground truth at {gt_path}")
            faces_ground_truth = {}
        except Exception as e:
            print(f"[evaluate] ERROR: Loading FACES ground truth failed: {e}")
            faces_ground_truth = {}
    # --- End New Loading ---

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

        # Normalize URI lookup
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
        
        # Get edges
        out_edges = list(G.out_edges(targetentity, data=True))

        if not out_edges:
            if dataset_name == 'faces':
                results.append({'eid': index, 'euri': targetentity, 'fmeasure': 0.0, 'ave_precision': 0.0, 'Top': k})
                results_norel.append({'eid': index, 'euri': targetentity, 'fmeasure_norel': 0.0, 'ave_precision_norel': 0.0, 'Top': k})
            else:
                for i in range(6):
                    results.append({'eid': index, 'euri': targetentity, 'fmeasure': 0.0, 'ave_precision': 0.0, 'Top': k})
                    results_norel.append({'eid': index, 'euri': targetentity, 'fmeasure_norel': 0.0, 'ave_precision_norel': 0.0, 'Top': k})
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
            else:
                for i in range(6):
                    results.append({'eid': index, 'euri': targetentity, 'fmeasure': 0.0, 'ave_precision': 0.0, 'Top': k})
                    results_norel.append({'eid': index, 'euri': targetentity, 'fmeasure_norel': 0.0, 'ave_precision_norel': 0.0, 'Top': k})
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

        pval = []
        for edge in summary_edges:
            relation = edge[2].get('relation') if isinstance(edge[2], dict) else edge[2]['relation']
            s = edge[0].strip().strip('<>')
            p = relation.strip().strip('<>')
            o = edge[1].strip().strip('<>')
            pval.append((s, p, o))
        
        pval = set(pval)

        if dataset_name == 'faces':
            target_key = targetentity.strip('<>')
            tval = faces_ground_truth.get(target_key, set())
            
            # Fallback: if key not found, try wrapping/unwrapping
            if not tval:
                 tval = faces_ground_truth.get(f"<{target_key}>", set())

            fscore = utils.fmeasure_score(tval, pval)
            Ap = average_precision(tval, pval)
            results.append({'eid': index, 'euri': targetentity, 'fmeasure': fscore, 'ave_precision': Ap, 'Top': k})

            pval_norel = {(item[0], item[-1]) for item in pval}
            tval_norel = {(item[0], item[-1]) for item in tval}
            fscore_norel = utils.fmeasure_score(tval_norel, set(pval_norel))
            Ap_norel = average_precision(tval_norel, set(pval_norel))
            results_norel.append({'eid': index, 'euri': targetentity, 'fmeasure_norel': fscore_norel, 'ave_precision_norel': Ap_norel, 'Top': k})

        else:
            for i in range(6):
                tval = import_top_summary(path, index, i, k, targetentity)
                fscore = utils.fmeasure_score(tval, pval)
                Ap = average_precision(tval, pval)
                results.append({'eid': index, 'euri': targetentity, 'fmeasure': fscore, 'ave_precision': Ap, 'Top': k})

            pval_norel = {(item[0], item[-1]) for item in pval}
            for i in range(6):
                tval = import_top_summary(path, index, i, k, targetentity)
                tval_norel = {(item[0], item[-1]) for item in tval}
                fscore_norel = utils.fmeasure_score(tval_norel, set(pval_norel))
                Ap_norel = average_precision(tval_norel, set(pval_norel))
                results_norel.append({'eid': index, 'euri': targetentity, 'fmeasure_norel': fscore_norel, 'ave_precision_norel': Ap_norel, 'Top': k})

    results_df = pd.DataFrame(results)
    results_norel_df = pd.DataFrame(results_norel)

    if results_df.empty:
        print(f"[evaluate] No results computed. processed_entities={processed_entities}, skipped_entities={skipped_entities}")
        return 0.0, 0.0, 0.0, 0.0

    if 'euri' not in results_df.columns:
        results_df['euri'] = results_df.get('euri', results_df.get('eid', 0)).astype(str)
    if 'euri' not in results_norel_df.columns:
        results_norel_df['euri'] = results_norel_df.get('euri', results_norel_df.get('eid', 0)).astype(str)

    averages = results_df.groupby('euri')['fmeasure'].mean()
    averages_norel = results_norel_df.groupby('euri')['fmeasure_norel'].mean()
    averages_ap = results_df.groupby('euri')['ave_precision'].mean()
    averages_norel_ap = results_norel_df.groupby('euri')['ave_precision_norel'].mean()

    return (averages.mean(), averages_norel.mean(), averages_ap.mean(), averages_norel_ap.mean())