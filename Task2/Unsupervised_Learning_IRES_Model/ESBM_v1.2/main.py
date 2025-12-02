#!/usr/bin/env python3
import os.path
from functools import lru_cache
from statistics import mean
import torch.nn.functional as F

import numpy as np
import optuna
import pandas as pd
import torch
from torch_geometric.data import Data
from torch_geometric.nn import GAE
import torch.optim.lr_scheduler as lr_scheduler
import time

import evaluate
from gcnencoder import RGAE_Encoder, GAE_Encoder

import loss_function as lf

import utils
from config import Config
import random

import matplotlib.pyplot as plt
import networkx as nx


@lru_cache(maxsize=None)
def load_graph():
    # Required paths
    path_dataset, path_entity_id, transe_save, path_relation_id = utils.import_dataset()

    # Return triples and graphs
    triples = utils.generate_triples()
    entity_id, relation_id, relation_embeddings_numpy = utils.load_transe(transe_save, path_entity_id, path_relation_id)
    entity_dataset, last_part_euri_list = utils.generate_entity_dataset()
    G, adj, features_transe = utils.generate_adj_features(triples, entity_id, transe_save,
                                                          last_part_euri_list)
    nodes = list(G.nodes())
    entity_node_id = {node: i for i, node in enumerate(nodes)}
    edge_relation_id = {(entity_node_id[u], entity_node_id[v]): relation_id[G[u][v]['relation']] for i, (u, v) in
                        enumerate(G.edges)}

    adj = torch.tensor(adj)
    features_transe = torch.tensor(np.array(features_transe))
    return entity_dataset, G, nodes, entity_node_id, relation_id, edge_relation_id, adj, transe_save, features_transe


def setup_features(G, relation_id, edge_relation_id, adj, transe_save, features_transe):
    edge_index = adj.nonzero(as_tuple=False).t().contiguous()
    edge_attr = [edge_relation_id[(int(u), int(v))] for u, v in edge_index.t()]

    if config.target_feature() == "transe_nodes":
        features = features_transe
    elif config.target_feature() == "transe_node_relations":
        relation_transe_node = utils.aggregate_relation_embeddings_to_list(G, transe_save, relation_id)
        relation_transe_node = torch.tensor(relation_transe_node)
        features = features_transe + relation_transe_node
    elif config.target_feature() == "node_freq":
        node_freq = utils.generate_edge_weight_tensor(G, edge_index)
        node_freq = torch.tensor(node_freq, dtype=torch.float32).view(-1, 1)
        features = node_freq
    elif config.target_feature() == "node_relation_freq":
        
        relation_out_frequency_node = utils.generate_node_relation_weight_tensor(relation_id, G, edge_index)
        relation_out_frequency_node = torch.tensor(relation_out_frequency_node, dtype=torch.float32).view(-1, 1)
        
        relation_in_frequency_node=utils.generate_node_in_relation_weight_tensor(relation_id, G, edge_index)
        relation_in_frequency_node = torch.tensor(relation_in_frequency_node, dtype=torch.float32).view(-1, 1)
        
        node_freq = utils.generate_edge_weight_tensor(G, edge_index)
        node_freq = torch.tensor(node_freq, dtype=torch.float32).view(-1, 1)
        features = torch.cat((node_freq, relation_out_frequency_node), dim=1)
    
    pg_data = Data(x=features, edge_index=edge_index, edge_attr=edge_attr)
    return pg_data, features


def prepare_data(G, pg_data, features, relation_id, adj):
    num_relations = len(G.edges)
    device = config.device()
    relation_types = utils.generate_relation_type_adj(G, relation_id)
    relation_types = torch.tensor(relation_types, device=device)
    adjacency_matrix = adj.float().to(device)
    features = features.to(device)
    pg_data = pg_data.to(device)
    return pg_data, features, adjacency_matrix, relation_types, num_relations


def create_model(num_relations, adjacency_matrix, edge_relation_id, features):
    num_features = features.shape[1]
    edge_types = None
    if config.encoder() == 'RGCN':
        model = GAE(RGAE_Encoder(num_features, config.hidden_embeddings(), num_relations)).to(config.device())
        adj_edge_index = adjacency_matrix.nonzero(as_tuple=False).t().contiguous()
        edge_types = torch.tensor([edge_relation_id[(int(u), int(v))] for u, v in adj_edge_index.t()],
                                  device=config.device())
    elif config.encoder() == 'GCN':
        model = GAE(GAE_Encoder(num_features, config.hidden_embeddings())).to(config.device())
    else:
        raise ValueError('Encoder type not supported')
    return model, edge_types


def select_optimizer(model_parameters):
    if config.optimizer() == 'ASGD':
        optimizer_func = torch.optim.ASGD
    elif config.optimizer() == 'AdamW':
        optimizer_func = torch.optim.AdamW
    elif config.optimizer() == 'Adam':
        optimizer_func = torch.optim.Adam
    else:
        raise ValueError('Optimizer not supported')
    optimizer = optimizer_func(model_parameters, lr=config.learning_rate(), weight_decay=config.weight_decay())

    if config.scheduler() == 'LinearLR':
        scheduler = lr_scheduler.LinearLR(optimizer, start_factor=1.0, end_factor=0.75, total_iters=config.num_epochs())
    elif config.scheduler() == 'OneCycleLR':
        scheduler = torch.optim.lr_scheduler.OneCycleLR(optimizer, max_lr=0.1, steps_per_epoch=config.step_size(),
                                                        epochs=config.num_epochs(), cycle_momentum=False,
                                                        three_phase=True,
                                                        pct_start=0.2, div_factor=10, final_div_factor=1500,
                                                        anneal_strategy='cos')
    elif config.scheduler() == 'StepLR':
        scheduler = torch.optim.lr_scheduler.StepLR(optimizer, step_size=config.step_size(), gamma=0.01)
    elif config.scheduler() == 'ReduceLROnPlateau':
        scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(optimizer, mode='min', factor=0.9, eps=1e-3,
                                                               patience=config.num_epochs() // 10)

    else:
        raise ValueError('Scheduler not supported')
    return optimizer, scheduler


def custom_loss(z, lambda_sim=1.0, lambda_diversity=1.0):
    z_norm = F.normalize(z, p=2, dim=1)
    similarity_matrix = torch.matmul(z_norm, z_norm.t())
    similarity_penalty = similarity_matrix.sum(dim=1).mean()
    diversity_reward = inverse_distance_sum_reward(similarity_matrix)
    total_loss = lambda_sim * similarity_penalty 
    return total_loss


def inverse_distance_sum_reward(similarity_matrix):
    epsilon = 1e-5
    distances = 1 - similarity_matrix
    inverse_distances = 1 / (distances + epsilon)
    diversity_reward = -torch.sum(inverse_distances, dim=1).mean()
    return diversity_reward


def calculate_loss(G, z, relation_types):
    Modularity2 = lf.fast_loss_modularity_trace(G, z.float(), len(G.edges)/2).to(config.device())
    print(Modularity2)
    return Modularity2


def evaluate_model(model, G, entity_dataset, features, edge_index, edge_types, k):
    model.eval()
    if config.encoder() == 'RGCN':
        z = model.encode(features, edge_index, edge_types)
    else:
        z = model.encode(features, edge_index)
    
    reconstructed_adj_weighted = torch.mm(z, z.t())
    
    # Updated to pass config.dataset()
    fscore, fscore_norel, ap, ap_norel, ndcg = evaluate.evaluate(config.data_path(), entity_dataset, G, k, reconstructed_adj_weighted, z, config.dataset())
    
    return fscore_norel


def train_model(model, G, edge_index, edge_types, features, entity_dataset, relation_types):
    optimizer, scheduler = select_optimizer(model.parameters())
    for epoch in range(config.num_epochs()):
        model.train()
        optimizer.zero_grad()
        if config.encoder() == 'RGCN':
            z = model.encode(features, edge_index, edge_types)
        elif config.encoder() == 'GCN':
            z = model.encode(features, edge_index)
        else:
            raise ValueError('Encoder type not supported')
        
        model.eval()
        if config.encoder() == 'RGCN':
            z = model.encode(features, edge_index, edge_types)
        else:
            z = model.encode(features, edge_index)

        loss = calculate_loss(G, z, relation_types)
        loss.backward()
        torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=1)
        optimizer.step()
        
        if config.scheduler() == 'ReduceLROnPlateau':
            scheduler.step(loss)
        else:
            scheduler.step()

    return model


def save_model_and_results(model, G, features, edge_index, edge_types, results_df, entity_dataset, round_num):
    model_metadata = config.model_metadata(round_num)
    result_metadata = config.result_metadata(round_num)

    if torch.cuda.is_available():
        torch.cuda.synchronize()
    start_time = time.time()

    model.eval()
    if config.encoder() == 'RGCN':
        z = model.encode(features, edge_index, edge_types) 
    else:
        z = model.encode(features, edge_index)
    
    reconstructed_adj_weighted = torch.mm(z, z.t())
    
    if torch.cuda.is_available():
        torch.cuda.synchronize()
    end_time = time.time()
    inference_runtime = end_time - start_time

    fscore5 = None
    fscore10 = None
    
    for k in [5, 10]:
        # Updated to pass config.dataset()
        fscore, fscore_norel, ap, ap_norel, ndcg = evaluate.evaluate(config.data_path(), entity_dataset, G, k, reconstructed_adj_weighted, z, config.dataset())
        
        if k == 5:
            fscore5 = fscore_norel
            map_5 = ap_norel
            print(f"Top-5 MAP: {map_5}")
        else:
            fscore10 = fscore_norel
            map_10 = ap_norel
            print(f"Top-10 MAP: {map_10}")
        
        resultsirow = {
            'dataset_version': config.benchmark(), 
            'dataset_name': config.dataset(),
            'dataset_type': config.format(),
            'round': round_num, 
            'num_epochs': config.num_epochs(),
            'feature_type': config.target_feature(), 
            'fmeasure_rel': fscore,
            'fmeasure_norel': fscore_norel, 
            'ap': ap,
            'ap_norel': ap_norel,
            'ndcg': ndcg,
            'inference_runtime': inference_runtime,
            'Top': k, 
            'learning_rate': config.learning_rate(),
            'weight_decay': config.weight_decay(), 
            'hidden_embedding': config.hidden_embeddings(),
            'optimizer_type': config.optimizer(), 
            'encoder_type': config.encoder()
        }
        results_df.loc[len(results_df)] = resultsirow
        results_df.to_csv(os.path.join(config.results_path(), result_metadata))
        
    return fscore5, fscore10 , map_5, map_10


config = Config()


def seed(seed):
    random.seed(seed)
    os.environ['PYTHONHASHSEED'] = str(seed)
    np.random.seed(seed)
    torch.manual_seed(config.seed())
    torch.cuda.manual_seed(config.seed())
    torch.cuda.manual_seed_all(config.seed())
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False


def main(trial=None, best_fscore5=None, best_fscore10=None):
    if config.seed() >= 0:
        seed(config.seed())

    entity_dataset, G, nodes, entity_node_id, relation_id, edge_relation_id, adj, transe_save, features_transe = load_graph()
    pg_data, features = setup_features(G, relation_id, edge_relation_id, adj, transe_save, features_transe)
    pg_data, features, adjacency_matrix, relation_types, num_relations = prepare_data(G, pg_data, features,
                                                                                         relation_id, adj)
    
    results_df = pd.DataFrame(
        columns=['dataset_version', 'dataset_name', 'dataset_type',
                 'round', 'num_epochs', 'feature_type', 'fmeasure_rel',
                 'fmeasure_norel','ap','ap_norel', 'ndcg', 'inference_runtime', 'Top', 'learning_rate',
                 'weight_decay', 'hidden_embedding',
                 'optimizer_type', 'encoder_type'])

    fscores_5 = []
    fscores_10 = []
    map_5=[]
    map_10=[]
    for i in range(config.num_executions()):
        model, edge_types = create_model(num_relations, adjacency_matrix, edge_relation_id, features)
        model = train_model(model, G, pg_data.edge_index, edge_types, features,entity_dataset,relation_types)
        fscore5, fscore10, map5, map10 = save_model_and_results(model, G, features, pg_data.edge_index, edge_types, results_df,
                                                entity_dataset, i)
        fscores_5.append(fscore5)
        fscores_10.append(fscore10)
        map_5.append(map5)
        map_10.append(map10)
        
        if trial and (mean(fscores_5) - best_fscore5 < -0.05 or mean(fscores_10) - best_fscore10 < -0.05):
            raise optuna.TrialPruned()

    
    print(f'Fscore_5: {mean(fscores_5)}, Fscore_10: {mean(fscores_10)}')
    print(f'MAP_5: {mean(map_5)}, MAP_10: {mean(map_10)}')
    
    return mean(fscores_5), mean(fscores_10)


if __name__ == '__main__':
    main()