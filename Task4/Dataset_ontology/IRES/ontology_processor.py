import os
import torch
import torch.nn as nn
import torch.nn.functional as F
import dgl
import csv
import json
import math
import sys
from rdflib import Graph as RDFGraph, RDF
from collections import defaultdict
from urllib.parse import unquote

# Import DGL layers safely
try:
    from dgl.nn import RelGraphConv, GraphConv
except ImportError:
    from dgl.nn.pytorch import RelGraphConv, GraphConv

# ==========================================
# 0. HELPER: CLEANER
# ==========================================
def clean_node(text):
    if not text: return ""
    text = unquote(str(text)).strip()
    if text.startswith("<") and text.endswith(">"):
        text = text[1:-1]
    return text

# ==========================================
# 1. ROBUST TRIPLE EXTRACTOR
# ==========================================
def extract_triples_from_json(data, triples_list):
    if isinstance(data, dict):
        s = data.get('subject') or data.get('s')
        p = data.get('predicate') or data.get('p') or data.get('relation')
        o = data.get('object') or data.get('o')
        if s and p and o:
            triples_list.append((str(s), str(p), str(o)))
        
        for k, v in data.items():
            if isinstance(v, list):
                for item in v:
                    if isinstance(item, list) and len(item) >= 2:
                        triples_list.append((str(k), str(item[0]), str(item[1])))
            if isinstance(v, (dict, list)):
                extract_triples_from_json(v, triples_list)

    elif isinstance(data, list):
        for item in data:
            if isinstance(item, list) and len(item) >= 3:
                triples_list.append((str(item[0]), str(item[1]), str(item[2])))
            else:
                extract_triples_from_json(item, triples_list)

# ==========================================
# 2. HIERARCHY LEARNER
# ==========================================
class UniversalHierarchyLearner:
    def __init__(self):
        self.class_counts = {}
        self.entity_types = defaultdict(list)
        self.total_entities = 0
        self.max_freq = 0

    def fit(self, dataset_path):
        print(f"   -> Learning Ontology from: {dataset_path}")
        if os.path.isfile(dataset_path) and dataset_path.endswith(".json"):
             try:
                with open(dataset_path, 'r', encoding='utf-8') as f:
                    data = json.load(f)
                    triples = []
                    extract_triples_from_json(data, triples)
                    for s, p, o in triples: self._process_triple(s, p, o)
             except: pass
        elif os.path.isdir(dataset_path):
            for root, dirs, files in os.walk(dataset_path):
                for file in files: self._scan_file(os.path.join(root, file))
        self.max_freq = max(self.class_counts.values()) if self.class_counts else 1

    def _scan_file(self, path):
        if path.endswith((".nt", ".ttl", ".xml", ".rdf")):
            try:
                g = RDFGraph()
                fmt = "nt" if path.endswith(".nt") else "turtle"
                g.parse(path, format=fmt)
                for s, p, o in g: self._process_triple(str(s), str(p), str(o))
            except: pass
        elif path.endswith((".tsv", ".csv")):
            sep = '\t' if path.endswith('.tsv') else ','
            try:
                with open(path, 'r', encoding='utf-8', errors='ignore') as f:
                    reader = csv.reader(f, delimiter=sep)
                    for row in reader:
                        if len(row) >= 3: self._process_triple(row[0], row[1], row[2])
            except: pass

    def _process_triple(self, s, p, o):
        s_clean = clean_node(s)
        p_lower = clean_node(p).lower()
        if s_clean not in self.entity_types: self.total_entities += 1
        keywords = ["type", "class", "category", "subject", "p31", "instanceof"]
        is_bad = any(x in p_lower for x in ["datatype", "mimetype", "image", "file", "label", "name", "date"])
        if (p == str(RDF.type) or any(k in p_lower for k in keywords)) and not is_bad:
            self.entity_types[s_clean].append(clean_node(o))
            self.class_counts[clean_node(o)] = self.class_counts.get(clean_node(o), 0) + 1

    def get_features(self, entity_uri):
        types = self.entity_types.get(entity_uri, [])
        if not types: return torch.tensor([0.5, 0.5]) 
        best_ic, best_depth = 0.0, 0.0
        for t in types:
            freq = self.class_counts.get(t, 0)
            prob = freq / (self.total_entities + 1e-9)
            ic = -math.log(prob)
            depth = 1.0 - (freq / (self.max_freq + 1e-9))
            if depth > best_depth: best_depth, best_ic = depth, ic
        return torch.tensor([best_ic, best_depth])

# ==========================================
# 3. GRAPH PARSER
# ==========================================
class GraphParser:
    def __init__(self):
        self.entity_to_id = {}; self.relation_to_id = {}; self.id_to_entity = {}; self.triples = []

    def parse(self, dataset_path):
        print(f"   -> Building Graph from: {dataset_path}")
        if os.path.isfile(dataset_path) and dataset_path.endswith(".json"):
             try:
                with open(dataset_path, 'r', encoding='utf-8') as f:
                    data = json.load(f)
                    triples = []
                    extract_triples_from_json(data, triples)
                    for s, p, o in triples: self._add_triple(s, p, o)
             except: pass
        elif os.path.isdir(dataset_path):
            for root, dirs, files in os.walk(dataset_path):
                for file in files: self._parse_file(os.path.join(root, file))
        
        if not self.triples: return dgl.graph(([], []))
        src = [t[0] for t in self.triples]; dst = [t[2] for t in self.triples]; rels = [t[1] for t in self.triples]
        g_dgl = dgl.graph((torch.tensor(src), torch.tensor(dst)))
        g_dgl.edata['etype'] = torch.tensor(rels)
        return g_dgl

    def _parse_file(self, path):
        if path.endswith((".nt", ".ttl")):
            try:
                g = RDFGraph(); fmt = "nt" if path.endswith(".nt") else "turtle"
                g.parse(path, format=fmt)
                for s, p, o in g: self._add_triple(str(s), str(p), str(o))
            except: pass
        elif path.endswith((".tsv", ".csv")):
            sep = '\t' if path.endswith('.tsv') else ','
            try:
                with open(path, 'r', encoding='utf-8', errors='ignore') as f:
                    reader = csv.reader(f, delimiter=sep)
                    for row in reader:
                        if len(row) >= 3: self._add_triple(row[0], row[1], row[2])
            except: pass

    def _add_triple(self, s, p, o):
        s, p, o = clean_node(s), clean_node(p), clean_node(o)
        if s not in self.entity_to_id: 
            self.entity_to_id[s] = len(self.entity_to_id); self.id_to_entity[len(self.entity_to_id)-1] = s
        if o not in self.entity_to_id: 
            self.entity_to_id[o] = len(self.entity_to_id); self.id_to_entity[len(self.entity_to_id)-1] = o
        if p not in self.relation_to_id: self.relation_to_id[p] = len(self.relation_to_id)
        self.triples.append((self.entity_to_id[s], self.relation_to_id[p], self.entity_to_id[o]))

# ==========================================
# 4. IRES MODEL (CRASH FIXED)
# ==========================================
class HierarchicalIRES(nn.Module):
    def __init__(self, num_nodes, num_rels, hidden_dim, num_anticommunities=16):
        super(HierarchicalIRES, self).__init__()
        self.structure_emb = nn.Embedding(num_nodes, hidden_dim)
        self.ontology_encoder = nn.Linear(2, hidden_dim)
        
        # EMERGENCY SWITCH: Use GraphConv if RelGraphConv is too heavy
        self.use_simple = num_rels > 1000 
        
        if self.use_simple:
            print(f"      [Optimization] High Relations ({num_rels}). Switching to GraphConv (Safe Mode).")
            # FIXED: Added allow_zero_in_degree=True to prevent crashes on WIKIES
            self.conv1 = GraphConv(hidden_dim * 2, hidden_dim, allow_zero_in_degree=True)
            self.conv2 = GraphConv(hidden_dim, num_anticommunities, allow_zero_in_degree=True)
        else:
            print(f"      [Optimization] Standard RelGraphConv with {10} bases.")
            self.conv1 = RelGraphConv(hidden_dim * 2, hidden_dim, num_rels, num_bases=10)
            self.conv2 = RelGraphConv(hidden_dim, num_anticommunities, num_rels, num_bases=10)
            
        self.dropout = nn.Dropout(0.2)
        
    def forward(self, g, ontology_features, edge_types):
        struct_h = self.structure_emb.weight
        onto_h = F.relu(self.ontology_encoder(ontology_features))
        h = torch.cat([struct_h, onto_h], dim=1)
        
        if self.use_simple:
            # Simple GCN (No edge types)
            h = self.conv1(g, h)
            h = F.elu(h)
            h = self.dropout(h)
            c_matrix = self.conv2(g, h)
        else:
            # Full RGCN (With edge types)
            h = self.conv1(g, h, edge_types)
            h = F.elu(h)
            h = self.dropout(h)
            c_matrix = self.conv2(g, h, edge_types)
            
        return F.softmax(c_matrix, dim=1), h

def select_summary_dpp(c_matrix, ontology_features, entity_ids, k=5):
    quality = torch.max(c_matrix, dim=1).values 
    norm_onto = F.normalize(ontology_features, p=2, dim=1)
    L_sim = torch.mm(norm_onto, norm_onto.t())
    L = torch.outer(quality, quality) * L_sim
    selected = []
    indices = list(range(len(entity_ids)))
    for _ in range(min(k, len(entity_ids))):
        best = -1; best_g = -1.0
        for i in indices:
            if i in selected: continue
            pen = sum(L[i, x].item() for x in selected)
            gain = quality[i].item() - (0.5 * pen)
            if gain > best_g: best_g = gain; best = i
        if best != -1: selected.append(best)
    return selected

# ==========================================
# 5. TARGET LOADING
# ==========================================
def get_target_entities(dataset_name, parser):
    targets = set()
    target_file = None
    if dataset_name == "FACES":
        target_file = "/content/KGSUMM/Task4/Dataset_ontology/FACES/elist.txt"
    elif dataset_name == "DBpedia":
        target_file = "/content/KGSUMM/Task4/Dataset_ontology/ESBM/Tool/ESBM-groundtruth/elist.txt"
    elif dataset_name == "LMDB":
        target_file = "/content/KGSUMM/Task4/Dataset_ontology/ESBM/Tool/ESBM-groundtruth/elist.txt"
    elif dataset_name == "WIKIES":
        target_file = "/content/KGSUMM/Task4/Dataset_ontology/WIKIES/gold.json"

    print(f"   -> Loading Target Entities from: {target_file}")
    if not target_file or not os.path.exists(target_file):
        return list(range(min(50, len(parser.entity_to_id))))

    if target_file.endswith(".json"):
        try:
            with open(target_file, 'r', encoding='utf-8') as f:
                data = json.load(f)
                keys = data.keys() if isinstance(data, dict) else [x.get('entity') for x in data]
                for k in keys:
                    clean_k = clean_node(k)
                    if clean_k in parser.entity_to_id: targets.add(parser.entity_to_id[clean_k])
        except: pass
    else:
        try:
            with open(target_file, 'r', encoding='utf-8') as f:
                for line in f:
                    parts = line.strip().split()
                    if len(parts) < 2: continue
                    eid = clean_node(parts[0])
                    if eid in parser.entity_to_id:
                        targets.add(parser.entity_to_id[eid]); continue
                    uri_candidates = [p for p in parts if "http" in p]
                    for uri in uri_candidates:
                        clean_uri = clean_node(uri)
                        if clean_uri in parser.entity_to_id:
                            targets.add(parser.entity_to_id[clean_uri]); break 
        except: pass

    if not targets:
        print("      ⚠️ No targets matched! Summarizing first 50 nodes.")
        return list(range(min(50, len(parser.entity_to_id))))
    print(f"      ✅ Targeting {len(targets)} entities.")
    return list(targets)

def run_hires_pipeline():
    datasets = {
        "DBpedia": "/content/KGSUMM/Task4/Dataset_ontology/ESBM/Datasets/dbpedia_data",
        "LMDB": "/content/KGSUMM/Task4/Dataset_ontology/ESBM/Datasets/lmdb_data",
        "FACES": "/content/KGSUMM/Task4/Dataset_ontology/FACES/faces_data",
        "WIKIES": "/content/KGSUMM/Task4/Dataset_ontology/WIKIES/data/data.json"
    }
    output_base = "/content/KGSUMM/Task4/Dataset_ontology/Outputs"
    os.makedirs(output_base, exist_ok=True)
    hidden_dim = 64
    
    for name, path in datasets.items():
        if not os.path.exists(path): continue
        print(f"\n{'='*10} PROCESSING DATASET: {name} {'='*10}")
        
        epochs = 3 if name == "WIKIES" else 30
        
        ontology = UniversalHierarchyLearner()
        ontology.fit(path)
        
        parser = GraphParser()
        g_dgl = parser.parse(path)
        num_nodes = g_dgl.num_nodes()
        if num_nodes == 0: continue
        num_rels = len(parser.relation_to_id)
        
        print(f"   -> Graph Built: {num_nodes} Nodes, {num_rels} Relations")
        
        features = [ontology.get_features(parser.id_to_entity[i]) for i in range(num_nodes)]
        onto_tensor = torch.stack(features) 
        
        model = HierarchicalIRES(num_nodes, num_rels, hidden_dim)
        optimizer = torch.optim.Adam(model.parameters(), lr=0.01)
        
        print("   -> Training H-IRES...")
        model.train()
        for epoch in range(epochs):
            c_matrix, _ = model(g_dgl, onto_tensor, g_dgl.edata['etype'])
            loss = -torch.mean(torch.sum(c_matrix * torch.log(c_matrix + 1e-9), dim=1))
            optimizer.zero_grad(); loss.backward(); optimizer.step()
        
        target_ids = get_target_entities(name, parser)
        output_file = os.path.join(output_base, f"{name}_output.txt")
        print(f"   -> Saving summaries to: {output_file}")
        
        model.eval()
        with torch.no_grad():
            c_matrix, _ = model(g_dgl, onto_tensor, g_dgl.edata['etype'])
            with open(output_file, "w") as f_out:
                f_out.write(f"--- SUMMARY OUTPUT FOR {name} ---\n")
                for i in target_ids:
                    entity_uri = parser.id_to_entity[i]
                    f_out.write(f"\nENTITY: {entity_uri}\n")
                    neighbors = g_dgl.successors(i).tolist()
                    if not neighbors: continue
                    unique_indices = list({parser.id_to_entity[nid]: nid for nid in neighbors}.values())
                    c_subset = c_matrix[unique_indices]
                    onto_subset = onto_tensor[unique_indices]
                    summary_indices = select_summary_dpp(c_subset, onto_subset, unique_indices, k=5)
                    for idx in summary_indices:
                        f_out.write(f"   -> {parser.id_to_entity[unique_indices[idx]]}\n")

if __name__ == "__main__":
    run_hires_pipeline()
