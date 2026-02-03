import os
import torch
import torch.nn as nn
import torch.nn.functional as F
import dgl
import numpy as np
import math
import csv
import json
from rdflib import Graph as RDFGraph, RDF
from collections import defaultdict
from urllib.parse import unquote

# Import DGL layers safely
try:
    from dgl.nn import RelGraphConv
except ImportError:
    try:
        from dgl.nn.pytorch import RelGraphConv
    except ImportError:
        from dgl.nn.pytorch.conv import RelGraphConv

# ==========================================
# 0. HELPER: URI NORMALIZER
# ==========================================
def normalize(text):
    if not text: return ""
    text = unquote(str(text))
    # Remove standard prefixes to match simple names
    for p in ["http://dbpedia.org/resource/", "http://data.linkedmdb.org/resource/film/", 
              "http://data.linkedmdb.org/resource/actor/", "<", ">", "http://xmlns.com/foaf/0.1/"]:
        text = text.replace(p, "")
    # Lowercase, remove underscores, strip whitespace
    return text.strip().replace("_", " ").lower()

# ==========================================
# 0. HELPER: ROBUST JSON PARSER
# ==========================================
def find_triples_recursive(data, triples_list):
    """Recursively searches JSON for any structure resembling a triple."""
    if isinstance(data, dict):
        # Check for Explicit Keys (s, p, o)
        s = data.get('subject') or data.get('s') or data.get('head')
        p = data.get('predicate') or data.get('p') or data.get('relation') or data.get('type')
        o = data.get('object') or data.get('o') or data.get('tail')
        
        if s and p and o:
            triples_list.append((str(s), str(p), str(o)))
        
        # Check for Adjacency Dict {"Elvis": {"bornIn": "USA"}}
        for k, v in data.items():
            if isinstance(v, (str, int, float)) and k not in ['subject', 's', 'predicate', 'p', 'object', 'o']:
                # Heuristic: If key is a verb-like string, assume adjacency triple? 
                # Safer to just recurse unless we are sure.
                pass 
            find_triples_recursive(v, triples_list)
            
    elif isinstance(data, list):
        for item in data:
            # Check for List Triple ["s", "p", "o"]
            if isinstance(item, list) and len(item) == 3:
                triples_list.append((str(item[0]), str(item[1]), str(item[2])))
            else:
                find_triples_recursive(item, triples_list)

# ==========================================
# 1. HIERARCHY LEARNER
# ==========================================
class UniversalHierarchyLearner:
    def __init__(self):
        self.class_counts = {}
        self.entity_types = defaultdict(list)
        self.total_entities = 0
        self.max_freq = 0

    def fit(self, dataset_path):
        print(f"   -> Learning Ontology from: {dataset_path}")
        
        # --- HANDLE SINGLE JSON FILE (WIKIES UPDATE) ---
        if os.path.isfile(dataset_path) and dataset_path.endswith(".json"):
            try:
                with open(dataset_path, 'r', encoding='utf-8') as f:
                    data = json.load(f)
                    triples = []
                    find_triples_recursive(data, triples)
                    
                    if not triples:
                        print(f"      ⚠️ Warning: JSON loaded but 0 triples found. Check structure.")
                    
                    for s, p, o in triples:
                        self._process_rdf_triple(s, p, o)
            except Exception as e:
                print(f"      ⚠️ Error parsing JSON ontology: {e}")

        # --- HANDLE DIRECTORY WALK ---
        elif os.path.isdir(dataset_path):
            for root, dirs, files in os.walk(dataset_path):
                for file in files:
                    file_path = os.path.join(root, file)
                    if file.endswith((".nt", ".ttl", ".xml", ".rdf")):
                        try:
                            g = RDFGraph()
                            fmt = "nt" if file.endswith(".nt") else "turtle" if file.endswith(".ttl") else "xml"
                            g.parse(file_path, format=fmt)
                            for s, p, o in g: self._process_rdf_triple(str(s), str(p), str(o))
                        except: continue
                    elif file.endswith(".csv"):
                        try:
                            with open(file_path, 'r', encoding='utf-8', errors='ignore') as f:
                                reader = csv.reader(f)
                                headers = next(reader, None)
                                if not headers: continue
                                header_str = str(headers).lower()
                                if "level3_main_occ" in header_str or "occupation" in header_str:
                                    try:
                                        id_idx = headers.index("name") if "name" in headers else 0
                                        class_idx = headers.index("level3_main_occ") 
                                        for row in reader:
                                            if len(row) > class_idx:
                                                self._register_class(row[id_idx], row[class_idx])
                                    except: pass
                        except: continue
        
        self.max_freq = max(self.class_counts.values()) if self.class_counts else 1
        print(f"      [Ontology] Found {self.total_entities} entities with {len(self.class_counts)} unique classes.")

    def _process_rdf_triple(self, s, p, o):
        if s not in self.entity_types: self.total_entities += 1
        p_lower = p.lower()
        keywords = ["type", "class", "category", "subject", "p31", "instanceof"]
        is_bad = any(x in p_lower for x in ["datatype", "mimetype", "image", "file", "label", "name", "date"])
        if (p == str(RDF.type) or any(k in p_lower for k in keywords)) and not is_bad:
            self._register_class(s, o)

    def _register_class(self, entity, cls_name):
        self.class_counts[cls_name] = self.class_counts.get(cls_name, 0) + 1
        self.entity_types[entity].append(cls_name)

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
# 2. GRAPH PARSER
# ==========================================
class GraphParser:
    def __init__(self):
        self.entity_to_id = {}; self.relation_to_id = {}; self.id_to_entity = {}; self.triples = []

    def parse(self, dataset_path):
        print(f"   -> Building Graph from: {dataset_path}")
        
        # --- HANDLE JSON FILE ---
        if os.path.isfile(dataset_path) and dataset_path.endswith(".json"):
             try:
                with open(dataset_path, 'r', encoding='utf-8') as f:
                    data = json.load(f)
                    triples = []
                    find_triples_recursive(data, triples)
                    for s, p, o in triples:
                        self._add_triple(s, p, o)
             except Exception as e:
                 print(f"      ⚠️ Error parsing JSON graph: {e}")

        # --- HANDLE DIRECTORY ---
        elif os.path.isdir(dataset_path):
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
                        if "wikies" in dataset_path.lower() and "train" not in file: continue 
                        try:
                            with open(file_path, 'r', encoding='utf-8', errors='ignore') as f:
                                reader = csv.reader(f)
                                for row in reader:
                                    if len(row) < 3: continue
                                    s, p, o = row[0], row[1], row[2]
                                    if "Unnamed" in s: continue
                                    self._add_triple(s, p, o)
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
# 3. IRES MODEL
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
# 4. TARGET LOADING
# ==========================================
def get_target_entities(dataset_name, dataset_path, parser):
    targets = set()
    target_file = None
    
    # --- 1. USE YOUR SPECIFIC PATHS ---
    if dataset_name == "FACES":
        target_file = "/content/KGSUMM/Task1/Unsupervised_Learning_IRES_Model/FACES/Temp/faces_groundtruth.json"
    elif dataset_name == "DBpedia":
        target_file = "/content/KGSUMM/Task1/Unsupervised_Learning_IRES_Model/ESBM_v1.2/Temp/dbpedia_groundtruth.json"
    elif dataset_name == "LMDB":
        target_file = "/content/KGSUMM/Task1/Unsupervised_Learning_IRES_Model/ESBM_v1.2/Temp/lmdb_groundtruth.json"
    elif dataset_name == "WIKIES":
        target_file = "/content/KGSUMM/Task4/Dataset_ontology/WIKIES/gold.json"

    print(f"   -> Loading Target Entities for {dataset_name} from: {target_file}")
    
    if not target_file or not os.path.exists(target_file):
        print(f"      ❌ Target file not found! Fallback to first 50 nodes.")
        return list(range(min(50, len(parser.entity_to_id))))

    # --- 2. SPECIAL WIKIES ID MAPPING ---
    id_map = {}
    if dataset_name == "WIKIES":
        # Scan WIKIES directory for CSVs to build ID map
        scan_dir = os.path.dirname(dataset_path) if os.path.isfile(dataset_path) else dataset_path
        for root, _, files in os.walk(scan_dir):
            for file in files:
                if file.endswith(".csv") and "test" not in file:
                    try:
                        with open(os.path.join(root, file), 'r') as f:
                            reader = csv.reader(f)
                            headers = next(reader, [])
                            try:
                                id_idx = headers.index('id') if 'id' in headers else 0
                                name_idx = headers.index('name') if 'name' in headers else (2 if len(headers)>2 else 1)
                                for row in reader:
                                    if len(row) > max(id_idx, name_idx):
                                        i, n = normalize(row[id_idx]), normalize(row[name_idx])
                                        id_map[n] = i
                            except: pass
                    except: pass

    # --- 3. READ THE TARGET FILE ---
    try:
        with open(target_file, 'r', encoding='utf-8') as f:
            # Handle JSON list/dict
            if target_file.endswith(".json"):
                data = json.load(f)
                keys = []
                if isinstance(data, dict): keys = list(data.keys())
                elif isinstance(data, list): keys = [x.get('entity') for x in data if 'entity' in x]
                
                for k in keys:
                    norm_k = normalize(k)
                    if k in parser.entity_to_id: targets.add(parser.entity_to_id[k])
                    elif dataset_name == "WIKIES" and norm_k in id_map:
                        mapped_id = id_map[norm_k]
                        if mapped_id in parser.entity_to_id: targets.add(parser.entity_to_id[mapped_id])
                    else:
                        for name, nid in parser.entity_to_id.items():
                            if normalize(name) == norm_k: targets.add(nid); break
            
            # Handle TSV/TXT
            else:
                # (Existing logic for TSV parsing if needed)
                pass 
    except Exception as e:
        print(f"      ⚠️ Error reading targets: {e}")

    if not targets:
        print("      ⚠️ No targets matched! Summarizing first 50 nodes as fallback.")
        return list(range(min(50, len(parser.entity_to_id))))
    
    print(f"      ✅ Targeting {len(targets)} specific entities.")
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
    epochs = 30
    
    for name, path in datasets.items():
        if not os.path.exists(path): continue
        print(f"\n{'='*10} PROCESSING DATASET: {name} {'='*10}")
        
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
        
        target_ids = get_target_entities(name, path, parser)
        
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
