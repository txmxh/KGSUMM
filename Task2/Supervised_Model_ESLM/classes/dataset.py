#!/usr/bin/env python3
# -*- coding: utf-8 -*-
import os
import sys
import numpy as np
import random
import torch
import torch.nn as nn
from sklearn.model_selection import KFold
from rdflib import Graph
import glob

from classes.helpers import Utils

UTILS = Utils()

class ESBenchmark:
    def __init__(self, ds_name, file_n=6, topk=5, weighted_adjacency_matrix=False):
        self.topk = topk
        self.weighted_adjacency_matrix = weighted_adjacency_matrix
        self.in_esbm_dir = os.path.join(os.getcwd(), "datasets/ESBM_benchmark_v1.2")
        self.in_faces_dir = os.path.join(os.getcwd(), 'datasets/FACES')
        self.ds_name = ds_name

        if ds_name == "dbpedia":
            self.db_path = os.path.join(self.in_esbm_dir, "dbpedia_data")
        elif ds_name == "lmdb":
            self.db_path = os.path.join(self.in_esbm_dir, "lmdb_data")
        elif ds_name == "faces":
            self.db_path = os.path.join(self.in_faces_dir, "faces_data")
        else:
            raise ValueError("The database name must be dbpedia, lmdb, or faces")

        self.file_n = file_n

    def get_5fold_train_valid_test_elist(self, ds_name_str):
        """Get splitted data including train, valid, and test data"""
        if ds_name_str == "dbpedia":
            split_path = os.path.join(self.in_esbm_dir, "dbpedia_split")
        elif ds_name_str == "lmdb":
            split_path = os.path.join(self.in_esbm_dir, "lmdb_split")
        elif ds_name_str == "faces":
            split_path = os.path.join(self.in_faces_dir, "faces_split")
        else:
            raise ValueError("The database name must be dbpedia, lmdb, or faces")

        train_data, valid_data, test_data = [], [], []
        for i in range(5):
            fold_path = os.path.join(split_path, f'Fold{i}')
            train_eids = self.read_split(fold_path, 'train')
            valid_eids = self.read_split(fold_path, 'valid')
            test_eids = self.read_split(fold_path, 'test')
            train_data.append(train_eids)
            valid_data.append(valid_eids)
            test_data.append(test_eids)

        return train_data, valid_data, test_data

    def get_triples(self, num):
        """Get triples using Standard Graph Parser"""
        triples = []
        g = Graph()
        file_path = os.path.join(self.db_path, f"{num}", f"{num}_desc.nt")
        
        try:
            # Parse the NT file using the standard Graph API
            g.parse(file_path, format="nt")
            for s, p, o in g:
                # Convert rdflib terms to Python strings to match original format
                triples.append((s.toPython(), p.toPython(), o.toPython()))
        except Exception as e:
            # Optionally print warning if file is missing/corrupt
            # print(f"Warning: Could not parse triples for {num}: {e}")
            pass
            
        return triples

    def get_labels(self, num):
        """Get entity label from knowledge base"""
        triples = self.get_triples(num)
        endpoint = "http://dbpedia.org/sparql" if self.ds_name in ["dbpedia", "faces"] else \
                    "https://api.triplydb.com/datasets/Triply/linkedmdb/services/linkedmdb/sparql"

        triples_tuple = []
        for sub, pred, obj in triples:
            if UTILS.is_uri(obj):
                if self.ds_name == "lmdb":
                    obj_literal = UTILS.get_label_of_entity_lmdb("entity", obj, endpoint)
                else:
                    obj_literal = UTILS.get_label_of_entity(obj, endpoint)
            else:
                obj_literal = obj.title() if isinstance(obj, str) else obj

            if self.ds_name == "lmdb":
                pred_literal = UTILS.get_label_of_entity_lmdb("property", pred, endpoint)
                sub_literal = UTILS.get_label_of_entity_lmdb("entity", sub, endpoint)
            else:
                pred_literal = UTILS.get_label_of_entity(pred, endpoint)
                sub_literal = UTILS.get_label_of_entity(sub, endpoint)

            triples_tuple.append((sub_literal, pred_literal, obj_literal))
        return triples_tuple

    def get_literals(self, num):
        """Get literal value from literal txt"""
        triples_literal = []
        # Ensure path points to where your main.py expects inputs
        path = os.path.join(os.getcwd(), f"classes/data_inputs/literals/{self.ds_name}")
        file_path = os.path.join(path, f"{num}_literal.txt")
        
        try:
            with open(file_path, encoding="utf-8") as reader:
                for literal in reader:
                    values = literal.strip().split("\t")
                    if len(values) == 3:
                        triples_literal.append((values[0], values[1], values[2]))
                    elif literal.strip(): 
                        pass 
        except FileNotFoundError:
            return [] 

        return triples_literal

    def get_training_dataset(self):
        """Get all training and validation data"""
        train_eids, valid_eids, _ = self.get_5fold_train_valid_test_elist(self.ds_name)
        train_data, valid_data = [], []

        for eids_per_fold in train_eids:
            edesc = {eid: self.get_triples(eid) for eid in eids_per_fold}
            train_data.append([edesc])

        for eids_per_fold in valid_eids:
            edesc = {eid: self.get_triples(eid) for eid in eids_per_fold}
            valid_data.append([edesc])

        return train_data, valid_data

    def get_testing_dataset(self):
        """Get all testing data"""
        _, _, test_eids = self.get_5fold_train_valid_test_elist(self.ds_name)
        test_data = []

        for eids_per_fold in test_eids:
            edesc = {eid: self.get_triples(eid) for eid in eids_per_fold}
            test_data.append([edesc])

        return test_data

    def prepare_labels(self, num):
        """Create gold label dictionary using Standard Graph Parser"""
        per_entity_label_dict = {}
        
        for i in range(self.file_n):
            path = os.path.join(self.db_path, f"{num}")
            gold_file = os.path.join(path, f"{num}_gold_top{self.topk}_{i}.nt")
            
            if os.path.exists(gold_file):
                g = Graph()
                try:
                    g.parse(gold_file, format="nt")
                    for s, p, o in g:
                        pred_str = p.toPython()
                        obj_str = o.toPython()
                        UTILS.counter(per_entity_label_dict, f"{pred_str}++$++{obj_str}")
                except Exception:
                    pass

        return per_entity_label_dict

    def triples_dictionary(self, num):
        """Build triple dictionary"""
        triples_dict = {}
        triples = self.get_triples(num)
        for triple in triples:
            if triple not in triples_dict:
                triples_dict[triple] = len(triples_dict)
        return triples_dict

    def get_gold_indices(self, num, summary_index):
        """
        Get indices of gold standard triples.
        Required by generate_scores.py
        """
        triples_dict = self.triples_dictionary(num)
        gold_indices = []
        
        gold_file_path = os.path.join(self.db_path, f"{num}", f"{num}_gold_top{self.topk}_{summary_index}.nt")
        
        if os.path.exists(gold_file_path):
            g = Graph()
            try:
                g.parse(gold_file_path, format="nt")
                for s, p, o in g:
                    # Construct tuple matching triples_dictionary format
                    triple = (s.toPython(), p.toPython(), o.toPython())
                    if triple in triples_dict:
                        gold_indices.append(triples_dict[triple])
            except Exception:
                pass
        
        return gold_indices

    @staticmethod
    def read_split(fold_path, split_name):
        """Read data from split txt"""
        split_eids = []
        with open(os.path.join(fold_path, f"{split_name}.txt"), encoding='utf-8') as reader:
            for line in reader:
                line = line.strip()
                if not line:
                    continue
                split_eids.append(int(line.split('\t')[0]))
        return split_eids

    @property
    def get_ds_name(self):
        return self.ds_name

    @property
    def get_db_path(self):
        return self.db_path