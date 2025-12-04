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
import scipy.sparse as sp

from pykeen.training import SLCWATrainingLoop
from pykeen.triples import TriplesFactory
from pykeen.models import TransE, TorusE

from config import Config

# --- BEGIN PICKLE FIX ---
from pykeen.models import TransE as RealTransE

class FakeUnimodaModule: pass
sys.modules['pykeen.models.unimoda'] = FakeUnimodaModule

class FakeTransEModule: pass
sys.modules['pykeen.models.unimoda.trans_e'] = FakeTransEModule

FakeTransEModule.TransE = RealTransE
# --- END PICKLE FIX ---


# ==========================================
# HELPER FUNCTIONS (DEFINED HERE, NO IMPORT NEEDED)
# ==========================================

def _compact(arr):
    """Removes empty strings or None values from a list."""
    return [x for x in arr if x and x.strip()]

def _extract(string):
    """Extracts the content inside <...> or "..."."""
    if not isinstance(string, str):
        return string
    string = string.strip()
    if string.startswith('<') and string.endswith('>'):
        return string[1:-1]
    if string.startswith('"') and string.endswith('"'):
        return string[1:-1]
    return string

def build_dict(contents):
    """Builds a dictionary mapping words to unique integer IDs."""
    dictionary = {}
    for content in contents:
        for word in content:
            if word not in dictionary:
                dictionary[word] = len(dictionary)
    return dictionary

def build_vec(contents, dictionary):
    """Converts lists of words into lists of integer IDs."""
    vecs = []
    for content in contents:
        vec = [dictionary[word] for word in content if word in dictionary]
        vecs.append(vec)
    return vecs

# ==========================================
# DATA LOADING FUNCTIONS
# ==========================================

def Import_ESBM(path):
    lst = os.listdir(path)
    g = Graph()
    for i_str in lst:
        g_temp = Graph()
        entity_file_name = os.path.join(path, i_str, f"{i_str}_desc.nt")
        try:
            g = g + g_temp.parse(entity_file_name, format='nt')
        except Exception as e:
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
    return triples_factory.entity_to_id, triples_factory.relation_to_id

# ==========================================
# EVALUATION HELPERS
# ==========================================

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

# ==========================================
# IMPORT DATASET
# ==========================================

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
            path_data = '/content/KGSUMM/Task1/Unsupervised_Learning_IRES_Model/Temp/'
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

# ==========================================
# ADDITIONAL TRAINING & NORMALIZATION UTILS
# ==========================================

def tensor_from_data(data, max_len):
    tensor = torch.zeros((len(data), max_len)).long()
    for i, seq in enumerate(data):
        seq_len = len(seq)
        if seq_len > 0:
            tensor[i, :seq_len] = torch.LongTensor(seq)
    return tensor

def tensor_from_weight(data, max_len):
    tensor = torch.zeros((len(data), max_len)).float()
    for i, seq in enumerate(data):
        seq_len = len(seq)
        if seq_len > 0:
            tensor[i, :seq_len] = torch.FloatTensor(seq)
    return tensor

def _eval_Fmeasure(golden, predict):
    if isinstance(golden, list): golden = set(golden)
    if isinstance(predict, list): predict = set(predict)
    if len(golden) == 0 or len(predict) == 0:
        return 0.0
    common = len(golden.intersection(predict))
    if common == 0:
        return 0.0
    precision = common / len(predict)
    recall = common / len(golden)
    if precision + recall == 0:
        return 0.0
    return 2 * (precision * recall) / (precision + recall)

def accuracy(output, labels):
    preds = output.max(1)[1].type_as(labels)
    correct = preds.eq(labels).double()
    correct = correct.sum()
    return correct / len(labels)

def _eval_ndcg_scores(gold_list, pred_scores, k):
    return 0.0

def normalize_features(mx):
    rowsum = np.array(mx.sum(1))
    r_inv = np.power(rowsum, -1).flatten()
    r_inv[np.isinf(r_inv)] = 0.
    r_mat_inv = sp.diags(r_inv)
    mx = r_mat_inv.dot(mx)
    return mx

def normalize_adj(mx):
    mx = sp.coo_matrix(mx)
    rowsum = np.array(mx.sum(1))
    d_inv_sqrt = np.power(rowsum, -0.5).flatten()
    d_inv_sqrt[np.isinf(d_inv_sqrt)] = 0.
    d_mat_inv_sqrt = sp.diags(d_inv_sqrt)
    return mx.dot(d_mat_inv_sqrt).transpose().dot(d_mat_inv_sqrt).tocoo()