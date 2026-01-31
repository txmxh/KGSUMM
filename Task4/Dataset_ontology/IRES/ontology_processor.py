import os
import torch
import torch.nn as nn
import torch.nn.functional as F
import dgl
import numpy as np
import math
import csv
from rdflib import Graph as RDFGraph, RDF
from collections import defaultdict

# --- FIX: Robust Import Strategy ---
try:
    from dgl.nn import RelGraphConv
except ImportError:
    try:
        from dgl.nn.pytorch import RelGraphConv
    except ImportError:
        from dgl.nn.pytorch.conv import RelGraphConv

# ==========================================
# 1. UNIVERSAL HIERARCHY LEARNER
# ==========================================
class UniversalHierarchyLearner:
    def __init__(self):
        self.class_counts = {}
        self.entity_types = defaultdict(list)
        self.total_entities = 0
        self.max_freq = 0

    def fit(self, dataset_path):
        print(f"   -> Learning Ontology from: {dataset_path}")
        global_subjects = set()
        
        for root, dirs, files in os.walk(dataset_path):
            for file in files:
                file_path = os.path.join(root, file)
                
                if file.endswith((".nt", ".ttl", ".xml", ".rdf")):
                    try:
                        g = RDFGraph()
                        fmt = "nt" if file.endswith(".nt") else "turtle" if file.endswith(".ttl") else "xml"
                        g.parse(file_path, format=fmt)
                        for s, p, o in g: self._process_triple(str(s), str(p), str(o), global_subjects)
                    except: continue
                
                elif file.endswith(".csv"):
                    try:
                        with open(file_path, 'r', encoding='utf-8', errors='ignore') as f:
                            reader = csv.reader(f)
                            # FIX: Attempt to skip header if it looks like "Unnamed" or "id"
                            try:
                                header = next(reader)
                                if "Unnamed" not in str(header[0]) and "id" not in str(header[0]):
                                    # If it wasn't a header, reset (this is rough, but safer for mixed files)
                                    f.seek(0)
                            except: pass
                            
                            for row in reader:
                                if len(row) >= 3: self._process_triple(row[0], row[1], row[2], global_subjects)
                    except: continue
                    
        print(f"      [Done] Entities: {self.total_entities}, Unique Classes: {len(self.class_counts)}")

    def _process_triple(self, s, p, o, global_subjects):
        if s not in global_subjects:
            global_subjects.add(s)
            self.total_entities += 1
        is_type = (p == str(RDF.type)) or ("type" in p.lower())
        if is_type:
            self.class_counts[o] = self.class_counts.get(o, 0) + 1
            if self.class_counts[o] > self.max_freq: self.max_freq = self.class_counts[o]
            self.entity_types[s].append(o)

    def get_features(self, entity_uri):
        types = self.entity_types.get(entity_uri, [])
        if not types: return torch.tensor([0.0, 0.0])
        best_ic, best_depth = 0.0, 0.0
        for t in types:
            freq = self.class_counts.get(t, 0)
            prob = freq / (self.total_entities + 1e-9)
            ic = -math.log(prob)
            depth = 1.0 - (freq / (self.max_freq + 1e-9))
            if depth > best_depth: best_depth, best_ic = depth, ic
        return torch.tensor([best_ic, best_depth])

# ==========================================
# 2. GRAPH PARSER
# ==========================================
class GraphParser:
    def __init__(self):
        self.entity_to_id = {}; self.relation_to_id = {}; self.id_to_entity = {}; self.triples = []

    def parse(self, dataset_path):
        print(f"   -> Building Graph from: {dataset_path}")
        for root, dirs, files in os.walk(dataset_path):
            for file in files:
                file_path = os.path.join(root, file)
                if file.endswith((".nt", ".ttl")):
                    try:
                        g = RDFGraph(); fmt = "nt" if file.endswith(".nt") else "turtle"
                        g.parse(file_path, format=fmt)
                        for s, p, o in g: self._add_triple(str(s), str(p), str(o))
                    except: continue
                elif file.endswith(".csv"):
                    try:
                        with open(file_path, 'r', encoding='utf-8', errors='ignore') as f:
                            reader = csv.reader(f)
                            try:
                                header = next(reader)
                                if "Unnamed" not in str(header[0]) and "id" not in str(header[0]):
                                    f.seek(0)
                            except: pass
                            
                            for row in reader:
                                if len(row) >= 3: self._add_triple(row[0], row[1], row[2])
                    except: continue
        
        if not self.triples: return dgl.graph(([], []))
        src = [t[0] for t in self.triples]; dst = [t[2] for t in self.triples]; rels = [t[1] for t in self.triples]
        g_dgl = dgl.graph((torch.tensor(src), torch.tensor(dst)))
        g_dgl.edata['etype'] = torch.tensor(rels)
        return g_dgl

    def _add_triple(self, s, p, o):
        if s not in self.entity_to_id: self.entity_to_id[s] = len(self.entity_to_id); self.id_to_entity[len(self.entity_to_id)-1] = s
        if o not in self.entity_to_id: self.entity_to_id[o] = len(self.entity_to_id); self.id_to_entity[len(self.entity_to_id)-1] = o
        if p not in self.relation_to_id: self.relation_to_id[p] = len(self.relation_to_id)
        self.triples.append((self.entity_to_id[s], self.relation_to_id[p], self.entity_to_id[o]))

# ==========================================
# 3. HIERARCHICAL IRES MODEL
# ==========================================
class HierarchicalIRES(nn.Module):
    def __init__(self, num_nodes, num_rels, hidden_dim, num_anticommunities=16):
        super(HierarchicalIRES, self).__init__()
        self.structure_emb = nn.Embedding(num_nodes, hidden_dim)
        self.ontology_encoder = nn.Linear(2, hidden_dim)
        self.rgcn1 = RelGraphConv(hidden_dim * 2, hidden_dim, num_rels)
        self.rgcn2 = RelGraphConv(hidden_dim, num_anticommunities, num_rels)
        self.dropout = nn.Dropout(0.2)
    def forward(self, g, ontology_features, edge_types):
        struct_h = self.structure_emb.weight
        onto_h = F.relu(self.ontology_encoder(ontology_features))
        h = torch.cat([struct_h, onto_h], dim=1)
        h = self.rgcn1(g, h, edge_types)
        h = F.elu(h)
        h = self.dropout(h)
        c_matrix = self.rgcn2(g, h, edge_types)
        return F.softmax(c_matrix, dim=1), h

# ==========================================
# 4. SEMANTIC DPP SELECTION
# ==========================================
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
# 5. MAIN EXECUTION (FIXED SAVING & DE-DUPLICATION)
# ==========================================
def run_hires_pipeline():
    datasets = {
        "DBpedia": "/content/KGSUMM/Task4/Dataset_ontology/ESBM/Datasets/dbpedia_data",
        "LMDB": "/content/KGSUMM/Task4/Dataset_ontology/ESBM/Datasets/lmdb_data",
        "FACES": "/content/KGSUMM/Task4/Dataset_ontology/FACES/faces_data",
        "WIKIES": "/content/KGSUMM/Task4/Dataset_ontology/WIKIES/data/seed_nodes"
    }
    
    output_base = "/content/KGSUMM/Task4/Dataset_ontology/Outputs"
    os.makedirs(output_base, exist_ok=True)
    hidden_dim = 64
    epochs = 50 
    
    for name, path in datasets.items():
        if not os.path.exists(path):
            print(f"⚠️ SKIPPING {name}: Path not found.")
            continue
            
        print(f"\n{'='*10} PROCESSING DATASET: {name} {'='*10}")
        
        # A. Learn Ontology
        ontology = UniversalHierarchyLearner()
        ontology.fit(path)
        
        # B. Build Graph
        parser = GraphParser()
        g_dgl = parser.parse(path)
        num_nodes = g_dgl.num_nodes()
        if num_nodes == 0: continue
        num_rels = len(parser.relation_to_id)
        
        # C. Prepare Features
        features = []
        for i in range(num_nodes):
            uri = parser.id_to_entity[i]
            features.append(ontology.get_features(uri))
        onto_tensor = torch.stack(features) 
        
        # D. Train Model
        model = HierarchicalIRES(num_nodes, num_rels, hidden_dim)
        optimizer = torch.optim.Adam(model.parameters(), lr=0.01)
        
        print("   -> Training H-IRES...")
        model.train()
        for epoch in range(epochs):
            c_matrix, embeddings = model(g_dgl, onto_tensor, g_dgl.edata['etype'])
            loss = -torch.mean(torch.sum(c_matrix * torch.log(c_matrix + 1e-9), dim=1))
            optimizer.zero_grad(); loss.backward(); optimizer.step()
        
        # E. GENERATE & SAVE SUMMARIES (NO DUPLICATES)
        output_file = os.path.join(output_base, f"{name}_output.txt")
        print(f"   -> Saving Summaries to: {output_file}")
        
        model.eval()
        with torch.no_grad():
            c_matrix, _ = model(g_dgl, onto_tensor, g_dgl.edata['etype'])
            
            with open(output_file, "w") as f_out:
                f_out.write(f"--- SUMMARY OUTPUT FOR {name} ---\n")
                
                for i in range(min(50, num_nodes)):
                    entity_uri = parser.id_to_entity[i]
                    f_out.write(f"\nENTITY: {entity_uri}\n")
                    
                    neighbors = g_dgl.successors(i).tolist()
                    if not neighbors: 
                        f_out.write("   (No neighbors to summarize)\n")
                        continue
                    
                    # --- FIX: REMOVE DUPLICATE NEIGHBORS (BY STRING) ---
                    # We group neighbor IDs by their string value to prevent repeats
                    unique_neighbors_map = {}
                    for nid in neighbors:
                        obj_str = parser.id_to_entity[nid]
                        if obj_str not in unique_neighbors_map:
                            unique_neighbors_map[obj_str] = nid
                    
                    # Use unique list for selection
                    unique_neigh_indices = list(unique_neighbors_map.values())
                    
                    # Select
                    c_subset = c_matrix[unique_neigh_indices]
                    onto_subset = onto_tensor[unique_neigh_indices]
                    summary_indices = select_summary_dpp(c_subset, onto_subset, unique_neigh_indices, k=3)
                    
                    for idx in summary_indices:
                        actual_node_id = unique_neigh_indices[idx]
                        fact_obj = parser.id_to_entity[actual_node_id]
                        f_out.write(f"   -> {fact_obj}\n")

if __name__ == "__main__":
    run_hires_pipeline()
