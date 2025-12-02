import os
import sys
import json
import pickle
import re
from functools import lru_cache
from collections import Counter

import numpy as np
import pandas as pd
import torch
from torch.optim import Adam
import networkx as nx
import rdflib
from rdflib.graph import Graph

from pykeen.training import SLCWATrainingLoop
from pykeen.triples import TriplesFactory
from pykeen.models import TransE, TorusE

from config import Config

# --- BEGIN PICKLE FIX ---
# This block fixes the torch.load() error by creating a "fake" module
# path to match the old .pkl file, pointing it to the real TransE class.
from pykeen.models import TransE as RealTransE

class FakeUnimodaModule: pass
sys.modules['pykeen.models.unimoda'] = FakeUnimodaModule

class FakeTransEModule: pass
sys.modules['pykeen.models.unimoda.trans_e'] = FakeTransEModule

FakeTransEModule.TransE = RealTransE
# --- END PICKLE FIX ---


# ==========================================
#  HELPER FUNCTIONS (FIXED)
# ==========================================

def _compact(arr):
    """
    Removes empty strings or None values from a list.
    Fixes: ImportError: cannot import name '_compact'
    """
    return [x for x in arr if x and x.strip()]

def _extract(string):
    """
    Extracts the content inside <...> or "..." for N-Triples parsing.
    Fixes: ImportError: cannot import name '_extract'
    """
    if not isinstance(string, str):
        return string
    string = string.strip()
    if string.startswith('<') and string.endswith('>'):
        return string[1:-1]
    if string.startswith('"') and string.endswith('"'):
        return string[1:-1]
    return string

def build_dict(contents):
    """
    Builds a dictionary mapping words to unique integer IDs.
    Fixes: ImportError: cannot import name 'build_dict'
    """
    dictionary = {}
    for content in contents:
        for word in content:
            if word not in dictionary:
                dictionary[word] = len(dictionary)
    return dictionary

def build_vec(contents, dictionary):
    """
    Converts lists of words into lists of integer IDs based on the dictionary.
    Fixes: ImportError: cannot import name 'build_vec'
    """
    vecs = []
    for content in contents:
        vec = [dictionary[word] for word in content if word in dictionary]
        vecs.append(vec)
    return vecs

# ==========================================
# END HELPER FUNCTIONS
# ==========================================


def Import_ESBM(path):
    lst = os.listdir(path)
    g = Graph()
    # Assuming subdirectories are named like '1', '2', etc.
    for i_str in lst:
        g_temp = Graph()
        entity_file_name = os.path.join(path, i_str, f"{i_str}_desc.nt")
        try:
            g = g + g_temp.parse(entity_file_name, format='nt')
        except Exception as e:
            # Handle cases where a file might be missing or unreadable
            print(f"Warning: Could not parse {entity_file_name}. Error: {e}")
    return g


def IMPORT_ESBM_plus(path):
    g = Graph()
    g.parse(path, format="nt")
    triples = []
    for s, p, o in g:
        triples.append([s.n3(), p.n3(), o.n3()])
    return (g)


def Import_ESBM_lmdb(path):
    g = Graph()
    index = [i for i in range(101, 141)]
    index += [i for i in range(166, 176)]
    for i in index:
        g_temp = Graph()
        entity_file_name = os.path.join(path, str(i), f"{i}_desc.nt")
        try:
            g = g + g_temp.parse(entity_file_name, format='nt')
        except Exception as e:
            print(f"Warning: Could not parse {entity_file_name}. Error: {e}")
    return g


def Import_ESBM_dbpedia(path):
    g = Graph()
    index = [i for i in range(1, 101)]
    index += [i for i in range(141, 166)]
    for i in index:
        g_temp = Graph()
        entity_file_name = os.path.join(path, str(i), f"{i}_desc.nt")
        try:
            g = g + g_temp.parse(entity_file_name, format='nt')
        except Exception as e:
            print(f"Warning: Could not parse {entity_file_name}. Error: {e}")
    return g


def TransETraining(triples, transe_save):
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

    triples_factory = TriplesFactory.from_labeled_triples(triples)
    model=TransE(triples_factory=triples_factory, embedding_dim=50).to(device)
    optimizer = Adam(params=model.get_grad_params())
    training_loop = SLCWATrainingLoop(model=model, triples_factory=triples_factory, optimizer=optimizer)
    training_loop.train(num_epochs=200, triples_factory=triples_factory, batch_size=32)
    torch.save(model, transe_save)

    entity_to_id = triples_factory.entity_to_id
    relation_to_id = triples_factory.relation_to_id

    return entity_to_id, relation_to_id


####### EVALUATION : ###########


def directed_ground_truth(path, euri):
    label = Graph()
    triples_labels = []
    try:
        label = label.parse(path, format='nt')
        for s, p, o in label:
            triples_labels.append([s.n3(), p.n3(), o.n3()])
    except FileNotFoundError:
        print(f"Warning: Ground truth file not found at {path}")
    return triples_labels


def ground_truth(path, euri):
    label = Graph()
    triples_labels = []
    try:
        label = label.parse(path, format='nt')
        for s, p, o in label:
            triples_labels.append([s.n3(), p.n3(), o.n3()])
    except FileNotFoundError:
        print(f"Warning: Ground truth file not found at {path}")
    return triples_labels


def import_directed_top_summary(path, triplenum, sumnum, topsum, targetentity):
    base_path = "Tool" 
    path_gt = os.path.join(base_path, "ESBM-groundtruth", "groundtruth", str(triplenum),
                           f"{triplenum}_gold_top{topsum}_{sumnum}.nt")
    triples_labels = directed_ground_truth(path_gt, targetentity)
    tpeval = [tuple(teval) for teval in triples_labels]
    tval = set(tpeval)
    return tval


def import_top_summary(path, triplenum, sumnum, topsum, targetentity):
    base_path = "Tool"
    path_gt = os.path.join(base_path, "ESBM-groundtruth", "groundtruth", str(triplenum),
                           f"{triplenum}_gold_top{topsum}_{sumnum}.nt")
    triples_labels = ground_truth(path_gt, targetentity)
    tpeval = [tuple(teval) for teval in triples_labels]
    tval = set(tpeval)
    return tval


def fmeasure_score(tval, pval):
    try:
        if len(pval) == 0 or len(tval) == 0:
            return 0.0
        precision = len(pval.intersection(tval)) / len(pval)
        recall = len(pval.intersection(tval)) / len(tval)
        if precision + recall == 0:
            return 0.0
        fmeasure = (2 * precision * recall) / (precision + recall)
    except:
        fmeasure = 0
    return fmeasure


def average_top_fmeasure(triplenum, pval, topsum):
    fmeasurelist = []
    for i in range(6):
        tval = import_top_summary(triplenum, i, topsum)
        fmeasurelist.append(fmeasure_score(tval, pval))
    return fmeasurelist


def import_top_machine_summary(frequency_dict, topsum):
    tpsum = []
    for tsumm in frequency_dict:
        tpsum.append(tuple(ts.n3() if type(ts) == rdflib.term.URIRef else ts for ts in tsumm))
    pval = set(tpsum[0:topsum])
    return pval


### IMPORT DATASET:
def import_dataset():
    config = Config()
    dataset_type = config.format()
    dataset_version = config.benchmark()
    dataset_name = config.dataset()
    data_path = config.data_path() 
    output_path = config.output_path() 
    
    if dataset_version == 'esbm_plus':
        file_extension = '.tsv' if dataset_type == 'extract' else '.nt'
        path_data = f'{data_path}/ESBM_PLUS_descriptions/{dataset_name}/complete_{dataset_type}_{dataset_name}{file_extension}'
    else:
        if dataset_name == 'faces':
            # Points to the Temp directory where you saved the Faces data
            path_data = '/content/KGSUMM/Task2/Unsupervised_Learning_IRES_Model/ESBM_v1.2/Temp/'
        elif dataset_name == 'dbpedia':
            path_data = 'Dataset/ESBM_benchmark_v1.2/dbpedia_data/'
        elif dataset_name == 'lmdb':
            path_data = 'Dataset/ESBM_benchmark_v1.2/lmdb_data/'
        else:
            path_data = f'{data_path}/ESBM_descriptions/'

    path_trained = f'{output_path}/Trained_Models'
    path_embedding = f'{output_path}/Trained_Models/obtained_embedding'
    os.makedirs(path_embedding, exist_ok=True)
    path_save_entity_id = f'{path_embedding}/entity_id_{dataset_type}_{dataset_name}_{dataset_version}.pkl'
    path_save_relation_id = f'{path_embedding}/relation_id_{dataset_type}_{dataset_name}_{dataset_version}.pkl'
    save_transe_model = f'{path_trained}/{dataset_name}_{dataset_type}_{dataset_version}.pkl'

    return path_data, path_save_entity_id, save_transe_model, path_save_relation_id


def generate_entity_dataset():
    config = Config()
    dataset_type = config.format()
    dataset_name = config.dataset()
    path = 'Tool'
    last_part_euri_list = None 
    
    if dataset_name == 'faces':
        # Points to the Temp directory where you saved the Faces Groundtruth
        gt_path = '/content/KGSUMM/Task2/Unsupervised_Learning_IRES_Model/ESBM_v1.2/Temp/faces_groundtruth.json'
        
        # Check if file exists to avoid crashes
        if not os.path.exists(gt_path):
             print(f"ERROR: Faces groundtruth not found at {gt_path}")
             return pd.DataFrame(), None

        with open(gt_path, 'r') as f:
            gt_data = json.load(f)
        
        entity_ids = list(gt_data.keys())
        entity_dataset = pd.DataFrame({
            'euri': entity_ids, 
            'dataset': ['faces'] * len(entity_ids)
        })
    else:
        entitylist = pd.read_csv(os.path.join(path, 'ESBM-groundtruth', 'elist.txt'), sep='\t', index_col=0)
        if dataset_type == 'extract':
            entitylist['last_part_euri'] = entitylist['euri'].str.split('/').str[-1]
            last_part_euri_list = entitylist[entitylist.dataset == 'dbpedia']['last_part_euri'].tolist()
        
        if dataset_name == 'lmdb':
            entity_dataset = entitylist[entitylist.dataset == 'lmdb']
        elif dataset_name == 'dbpedia':
            entity_dataset = entitylist[entitylist.dataset == 'dbpedia']

    if dataset_type == "extract":
        return entity_dataset, last_part_euri_list
    else:
        return entity_dataset, None


def generate_triples():
    path_dataset, path_entity_id, transe_save, path_relation_id = import_dataset()
    config = Config()
    dataset_type = config.format()
    dataset_version = config.benchmark()
    dataset_name = config.dataset()

    triples = []
    triples_test = []

    if dataset_version == 'esbm':
        if dataset_name == 'dbpedia':
            g = Import_ESBM_dbpedia(path_dataset)
            for s, p, o in g:
                triples.append([s.n3(), p.n3(), o.n3()])
            triples = np.array(triples)
            triples_factory = TriplesFactory.from_labeled_triples(triples)
        elif dataset_name == 'lmdb':
            g = Import_ESBM_lmdb(path_dataset)
            for s, p, o in g:
                triples.append([s.n3(), p.n3(), o.n3()])
            triples = np.array(triples)
            triples_factory = TriplesFactory.from_labeled_triples(triples)
        
        elif dataset_name == 'faces':
            # Points to the Temp directory where you saved the Faces Entities
            data_file = '/content/KGSUMM/Task2/Unsupervised_Learning_IRES_Model/ESBM_v1.2/Temp/faces_entities.json'
            
            if not os.path.exists(data_file):
                 raise FileNotFoundError(f"Faces entities file not found at {data_file}")

            with open(data_file, 'r') as f:
                triples = json.load(f)
            triples = np.array(triples)
            triples_factory = TriplesFactory.from_labeled_triples(triples)

    elif dataset_version == 'esbm_plus':
        if dataset_type == 'rdf':
            g = Graph()
            g = g.parse(path_dataset)
            for s, p, o in g:
                triples_test.append([s.n3(), p.n3(), o.n3()])
            triples = np.array(triples_test)
            triples_factory = TriplesFactory.from_labeled_triples(triples)
        else:
            triples_factory = TriplesFactory.from_path(path_dataset)
    triples = triples_factory.triples

    if not os.path.exists(path_entity_id):
        entity_id, relation_id = TransETraining(triples, transe_save)
        with open(path_entity_id, "wb") as file:
            pickle.dump(entity_id, file)
        with open(path_relation_id, "wb") as file:
            pickle.dump(relation_id, file)

    return triples


def load_transe(transe_save, path_entity_id, path_relation_id):
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

    model = torch.load(transe_save, weights_only=False).to(device)
    entity_embeddings = model.entity_representations[0](torch.arange(model.num_entities))
    relation_embeddings = model.relation_representations[0](torch.arange(model.num_relations))

    entity_embeddings_numpy = entity_embeddings.cpu().detach().numpy()
    relation_embeddings_numpy = relation_embeddings.cpu().detach().numpy()

    with open(path_entity_id, 'rb') as f:
        entity_id = pickle.load(f)
    with open(path_relation_id, 'rb') as f:
        relation_id = pickle.load(f)
    return entity_id, relation_id, relation_embeddings_numpy


def generate_adj_features(triples, entity_id, transe_save, last_part_euri_list):
    config = Config()
    dataset_type = config.format()

    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

    model = torch.load(transe_save, weights_only=False).to(device)
    G = nx.DiGraph()
    for head, relation, tail in triples:
        G.add_edge(head, tail, relation=relation)
    adj = nx.adjacency_matrix(G)

    nodes = list(G.nodes())

    # Create a dictionary that maps matrix indices to nodes.
    entity_node_id = {node: i for i, node in enumerate(nodes)}

    features = []

    for node in G.nodes():
        if dataset_type == 'extract':
            if node in last_part_euri_list:
                en_id = entity_id[node]
        else:
            en_id = entity_id[node]

        en_id_tensor = torch.tensor(en_id)
        entity_embeddings = model.entity_representations[0](en_id_tensor)
        features.append(entity_embeddings.cpu().detach().numpy())

    adj = adj.toarray()

    return G, adj, features


def generate_node_relation_weight_tensor(relation_id, G, edge_index):
    relation_names = [G[u][v]['relation'] for u, v in G.edges()]
    relation_frequency = Counter(relation_names)
    
    node_relation_sum = {node: 0 for node in G.nodes()}

    for node in G.nodes():
        for neighbor in G[node]:
            relation = G[node][neighbor]['relation']
            node_relation_sum[node] += relation_frequency.get(relation, 1)

    node_relation_values = [node_relation_sum[node] for node in G.nodes()]
    return node_relation_values


def generate_node_in_relation_weight_tensor(relation_id, G, edge_index):
    relation_names = [G[u][v]['relation'] for u, v in G.edges()]
    relation_frequency = Counter(relation_names)
    
    node_relation_sum_in = {node: 0 for node in G.nodes()}

    for node in G.nodes():
        for predecessor in G.predecessors(node):
            if 'relation' in G[predecessor][node]: 
                relation = G[predecessor][node]['relation']
                node_relation_sum_in[node] += relation_frequency.get(relation, 1)

    node_relation_values_in = [node_relation_sum_in[node] for node in G.nodes()]
    return node_relation_values_in


def generate_edge_weight_tensor(G, edge_index):
    node_degrees = dict(G.degree())
    node_counter = {node:degree for node, degree in node_degrees.items()}
    node_frequency = list(node_counter.values())
    return node_frequency


def aggregate_relation_embeddings_to_list(G, transe_save, relation_id):
    model = torch.load(transe_save, weights_only=False)
    node_aggregated_embeddings = {}
    non_zero_shape = None

    for node in G.nodes():
        embeddings_sum = None

        for neighbor in G.neighbors(node):
            relation = G[node][neighbor]['relation']
            if isinstance(relation, np.ndarray):
                relation = relation.item()

            en_rel = relation_id.get(relation)
            if en_rel is None:
                continue

            relation_embedding = model.relation_representations[0](
                torch.tensor(en_rel, dtype=torch.long)).cpu().detach().numpy()

            if embeddings_sum is None:
                embeddings_sum = np.zeros_like(relation_embedding)

            embeddings_sum += relation_embedding

        node_aggregated_embeddings[node] = embeddings_sum

        if embeddings_sum is not None and non_zero_shape is None:
            non_zero_shape = embeddings_sum.shape

    embeddings_list = []
    for node in G.nodes():
        embeddings = node_aggregated_embeddings[node]
        if embeddings is None or not np.any(embeddings):
            if non_zero_shape:
                embeddings_list.append([0] * non_zero_shape[0])
            else:
                embeddings_list.append([]) 
        else:
            embeddings_list.append(embeddings.tolist())

    return embeddings_list


def create_weighted_adjacency_matrix(G):
    relation_names = [G[u][v]['relation'] for u, v in G.edges()]
    relation_frequency = Counter(relation_names)
    inverse_relation_frequency = {relation: 1 / frequency for relation, frequency in relation_frequency.items()}

    node_mapping = {node: i for i, node in enumerate(G.nodes())}
    num_nodes = len(G.nodes())
    weighted_adj_matrix = torch.zeros((num_nodes, num_nodes))

    for u, v in G.edges():
        relation = G[u][v]['relation']
        weight = inverse_relation_frequency.get(relation, 1) 
        u_idx, v_idx = node_mapping[u], node_mapping[v]
        weighted_adj_matrix[u_idx, v_idx] = weight
        weighted_adj_matrix[v_idx, u_idx] = weight 

    return weighted_adj_matrix


def generate_relation_type_adj(G, relation_id):
    nodes = list(G.nodes())
    entity_node_id = {node: i for i, node in enumerate(nodes)}
    num_nodes = len(G.nodes())
    relation_types = np.zeros((num_nodes, num_nodes), dtype=int)

    for edge in G.edges(data=True):
        node_a_id = entity_node_id[edge[0]] 
        node_b_id = entity_node_id[edge[1]]
        relation_type = relation_id[edge[2]['relation']] 
        relation_types[node_a_id, node_b_id] = relation_type
        if not G.is_directed():
            relation_types[node_b_id, node_a_id] = relation_type

    return (relation_types)


def calculate_diversity_penalty(reconstructed_adj, relation_types):
    unique_relations = torch.unique(relation_types)
    relation_frequencies = {rel_type.item(): (relation_types == rel_type).sum().item() for rel_type in unique_relations}
    edge_weights = {rel_type: 1.0 / (freq + 1e-6) for rel_type, freq in relation_frequencies.items()}

    diversity_penalty = 0.0
    for rel_type in unique_relations:
        rel_type = rel_type.item()
        edge_mask = (relation_types == rel_type)
        reconstructed_probs = reconstructed_adj[edge_mask]
        weighted_penalty = edge_weights[rel_type] * reconstructed_probs.sum()
        diversity_penalty += weighted_penalty

    return diversity_penalty