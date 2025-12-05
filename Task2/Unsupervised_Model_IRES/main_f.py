#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Wed Dec  9 19:57:37 2020
"""
from __future__ import unicode_literals, print_function, division
import os
import os.path as path
import sys
import argparse
import torch
from gensim.models.keyedvectors import KeyedVectors

import numpy as np
import time
import math

# --- IMPORTS FROM YOUR _f FILES ---
from data_loader_f import split_data, load_emb
from train_f import train_iter
from generate_summary_f import generate_summary, ensembled_generating_summary

# --- FIXED PATHS (Matching your Colab Structure) ---
CURRENT_DIR = os.getcwd()
IN_ESBM_DIR = os.path.join(CURRENT_DIR, 'Dataset', 'ESBM_benchmark_v1.2')
IN_DBPEDIA_DIR = os.path.join(IN_ESBM_DIR, 'dbpedia_data')
IN_LMDB_DIR = os.path.join(IN_ESBM_DIR, 'lmdb_data')

IN_FACES_BASE = os.path.join(CURRENT_DIR, 'Dataset', 'FACES')
IN_FACES_DIR = os.path.join(IN_FACES_BASE, 'faces_data')

# Output directory for logs/results
OUT_DIR = os.path.join(CURRENT_DIR, 'Outputs')
if not os.path.exists(OUT_DIR):
    os.makedirs(OUT_DIR)

FILE_N = 6
TOP_K = [5, 10]
DS_NAME = ['dbpedia', 'lmdb', 'faces']

# --- FIXED DEVICE (Enable GPU) ---
DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")

def asHours(s):
  m = math.floor(s / 60)
  h = math.floor(m / 60)
  s -= m * 60
  m -= h * 60
  return '%dh %dm %ds' % (h, m, s)

def _read_epochs_from_log(ds_name, topk):
    log_file_path = os.path.join(OUT_DIR, 'GATES_log.txt')
    
    key = '{}-top{}'.format(ds_name, topk)
    epoch_list = None
    if os.path.exists(log_file_path):
        with open(log_file_path, 'r', encoding='utf-8') as f:
            for line in f:
                if line.startswith(key):
                    epoch_list = list(eval(line.split('\t')[1]))
    return epoch_list

def get_embeddings(word_emb_model):
    if word_emb_model == "fasttext":
        # Make sure this path exists or download it first!
        emb_path = "data/wiki-news-300d-1M.vec"
        if os.path.exists(emb_path):
            print(f"Loading FastText embeddings from {emb_path}...")
            word_emb = KeyedVectors.load_word2vec_format(emb_path)
        else:
            print(f"Warning: FastText file not found at {emb_path}. Using empty embeddings.")
            word_emb = {}
        
    elif word_emb_model=="Glove":
        word_emb = {}
        glove_path = "data/glove.6B/glove.6B.300d.txt"
        if os.path.exists(glove_path):
            with open(glove_path, 'r') as f:
                for line in f:
                    values = line.split()
                    word = values[0]
                    vector = np.asarray(values[1:], "float32")
                    word_emb[word] = vector
        else:
             print(f"Warning: Glove file not found at {glove_path}")
    else:
        print("please choose the correct word embedding model")
        sys.exit()
    return word_emb

def main(mode, emb_model, loss_type,  ent_emb_dim, pred_emb_dim, hidden_layers, nheads, lr, dropout, reg, weight_decay, n_epoch, save_every, word_emb_model, word_emb_calc, use_epoch, concat_model, weighted_edges_method): 
            
    word_emb = get_embeddings(word_emb_model)
    
    if loss_type == "BCE":
        loss_function = torch.nn.BCELoss()
    elif loss_type == "MSE":
        loss_function = torch.nn.MSELoss()
    elif loss_type == "NLLL":
        loss_function = torch.nn.NLLLoss()
    elif loss_type=="CE":
        loss_function=torch.nn.CrossEntropyLoss()
    else:
        print("please choose the correct loss fucntion")
        sys.exit()
        
    # --- FIXED LOGIC BUG ---
    # Was: weight_decay==0 (comparison), Changed to: weight_decay = 0 (assignment)
    if reg==True:
        weight_decay = weight_decay
    else:
        weight_decay = 0
        
    print('Hyper paramters:')  
    print("Loss function: {}".format(loss_function))
    print("Learning rate: {}".format(lr))
    print("Dropout: {}".format(dropout))
    if reg==True:
        print("Weight Decay: {}".format(weight_decay))
    print("n Epochs: {}".format(n_epoch))
    print("Regularization: {}".format(reg))
    print("Device: {}".format(DEVICE))
   
    if mode == "train" or mode =="test" or mode == "all":
        log_file_path = os.path.join(OUT_DIR, 'GATES_log.txt')
        if mode=="train":
            with open(log_file_path,'w') as log_file:pass
        
        # We iterate through the standard names, but paths are handled by the fixed logic above
        # If user passes -d faces, we might want to filter DS_NAME, but original code loops all.
        # We will assume single dataset execution based on typical usage or loop all.
        
        # NOTE: Original code loops through ALL datasets defined in DS_NAME. 
        # If you only want FACES, we can filter it here, but keeping original logic is safer.
        # However, to avoid errors if other datasets are missing, we can try/except.
        
        target_datasets = DS_NAME # ['dbpedia', 'lmdb', 'faces']
        
        for ds_name in target_datasets:
            if ds_name == "dbpedia":
                db_dir = IN_DBPEDIA_DIR
            elif ds_name == "lmdb":
                db_dir = IN_LMDB_DIR
            elif ds_name == "faces":
                db_dir = IN_FACES_DIR
            else:
                continue
            
            # Check if directory exists to avoid crash
            if not os.path.exists(db_dir):
                print(f"Skipping {ds_name}: Directory not found at {db_dir}")
                continue

            print('loading embeddings and dictionaries for {} ...'.format(ds_name))
            try:
                entity2vec, pred2vec, entity2ix, pred2ix = load_emb(ds_name, emb_model)
            except Exception as e:
                print(f"Error loading embeddings for {ds_name}: {e}")
                continue

            entity_dict = entity2vec
            pred_dict = pred2vec
            pred2ix_size = len(pred2ix)
            entity2ix_size = len(entity2ix)
            hidden_size = ent_emb_dim + pred_emb_dim
            start = time.time()
            
            if mode=="train" or mode=="all":
                print_to = os.path.join(OUT_DIR, 'model-training-{}.txt'.format(ds_name))
                with open(print_to, 'w+') as f:
                    f.write("Starting Training \n")
                    f.write("Training model is processed to {}\n".format(ds_name))
                    f.write("hyperparameters:\n")
                    f.write("Loss function: {}\n".format(loss_function))
                    f.write("Learning rate: {}\n".format(lr))
                    f.write("Dropout: {}\n".format(dropout))
                    if reg==True:
                        f.write("Weight Decay: {}\n".format(weight_decay))
                    f.write("n Epochs: {}\n".format(n_epoch))
                    f.write("Regularization: {}\n".format(reg))
                    f.write("nhead: {}\n".format(nheads))
                    f.write("Hidden layers: {}\n".format(hidden_layers))
                    f.write("Text embedding: {}\n".format(word_emb_model))
                    f.write("KGE model: {}\n".format(emb_model))
                    
            if mode=="test" or mode=="all":
                print_to = os.path.join(OUT_DIR, 'model-testing-dbpedia-lmdb.txt')
                if ds_name=='faces':
                   print_to = os.path.join(OUT_DIR, 'model-testing-{}.txt'.format(ds_name)) 
                   with open(print_to, 'w+') as f:
                       f.write("dsFile:true {}\n".format(IN_FACES_DIR))
                       f.write("===============================================================================\n")
                       f.write("dataset:{}\n".format(ds_name))
                        
            for topk in TOP_K:
                print(f"Processing {ds_name} Top-{topk}...")
                try:
                    train_adjs, train_facts, train_labels, val_adjs, val_facts, val_labels, test_adjs, test_facts, test_labels = split_data(ds_name, db_dir, topk, FILE_N, weighted_edges_method) 
                except Exception as e:
                    print(f"Error splitting data for {ds_name}: {e}")
                    continue

                if mode == "train" or mode=="all":
                    use_epoch = train_iter(ds_name, train_adjs, train_facts, train_labels, val_adjs, val_facts, val_labels, reg, n_epoch, save_every, DEVICE, entity_dict, \
                               pred_dict, loss_function, pred2ix_size, hidden_size, pred_emb_dim, ent_emb_dim, lr, dropout, entity2ix_size, hidden_layers, nheads, \
                               word_emb, db_dir, weight_decay, word_emb_calc, topk, FILE_N, concat_model, print_to, weighted_edges_method)
                    
                    with open(log_file_path,'a') as log_file:
                        line = '{}-top{} epoch:\t{}\n'.format(ds_name,topk, str(use_epoch))
                        log_file.write(line)
                        
                if mode == "test" or mode=="all":
                    print("Testing processes for {}@top{}".format(ds_name, topk))
                    use_epoch = _read_epochs_from_log(ds_name, topk) if mode=='test' else use_epoch # Use recently trained epoch if in 'all' mode
                    
                    if not use_epoch: 
                        print("No epochs found for testing. Skipping.")
                        continue

                    generate_summary(ds_name, test_adjs, test_facts, test_labels, pred_dict, entity_dict, pred2ix_size, pred_emb_dim, ent_emb_dim, \
                                 DEVICE, use_epoch, db_dir, dropout, entity2ix_size, hidden_layers, nheads, word_emb, word_emb_calc, topk, FILE_N, concat_model, print_to, weighted_edges_method)
                    
                    ensembled_generating_summary(ds_name, test_adjs, test_facts, test_labels, pred_dict, entity_dict, pred2ix_size, pred_emb_dim, ent_emb_dim, \
                                 DEVICE, use_epoch, db_dir, dropout, entity2ix_size, hidden_layers, nheads, word_emb, word_emb_calc, topk, FILE_N, concat_model, print_to, weighted_edges_method)
                    
            total_time = time.time()-start
            if mode=="train":
                print("Training processes time", asHours(total_time))
            elif mode=="test":
                print("Testing processes time", asHours(total_time))
            else:
                print("All processing time", asHours(total_time))
            
if __name__ == "__main__":
    parser = argparse.ArgumentParser(description='GATES: Graph Attention Network for Entity Summarization')
    parser.add_argument("--mode", type=str, default="test", help="use which mode type: train/test/all")
    parser.add_argument("--kge_model", type=str, default="ComplEx", help="use ComplEx/DistMult/ConEx")
    parser.add_argument("--loss_function", type=str, default="BCE", help="use which loss type: BCE/MSE")
    parser.add_argument("--ent_emb_dim", type=int, default=300, help="the embeddiing dimension of entity")
    parser.add_argument("--pred_emb_dim", type=int, default=300, help="the embeddiing dimension of predicate")
    parser.add_argument("--hidden_layers", type=int, default=2, help="the number of hidden layers")
    parser.add_argument("--nheads", type=int, default=1, help="the number of heads attention")
    parser.add_argument("--lr", type=float, default=0.05, help="use to define learning rate hyperparameter")
    parser.add_argument("--dropout", type=float, default=0.0, help="use to define dropout hyperparameter")
    parser.add_argument("--weight_decay", type=float, default=1e-5, help="use to define weight decay hyperparameter if the regularization set to True")
    parser.add_argument("--regularization", type=bool, default=False, help="use to define regularization: True/False")
    parser.add_argument("--save_every", type=int, default=1, help="save model in every n epochs")
    parser.add_argument("--n_epoch", type=int, default=50, help="train model in total n epochs")
    parser.add_argument("--word_emb_model", type=str, default="Glove", help="use which word embedding model: fasttext/Glove")
    parser.add_argument("--word_emb_calc", type=str, default="AVG", help="use which method to compute textual form: SUM/AVG")
    parser.add_argument("--use_epoch", type=int, nargs='+', help="how many epochs to train the model")
    parser.add_argument("--concat_model", type=int, default=1, help="use which concatenation model (1 or 2). In which, 1 refers to KGE + Word embedding, and 2 refers to KGE + (KGE/Word embeddings) depends on the object value")
    parser.add_argument("--weighted_edges_method", type=str, default="", help="use which apply the initialize weighted edges method: tf-idf")
    
    args = parser.parse_args()
    main(args.mode, args.kge_model, args.loss_function, args.ent_emb_dim, args.pred_emb_dim, args.hidden_layers, args.nheads, args.lr, args.dropout, args.regularization, args.weight_decay, \
         args.n_epoch, args.save_every, args.word_emb_model, args.word_emb_calc, args.use_epoch, args.concat_model, args.weighted_edges_method)