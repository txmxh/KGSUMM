
print(">>> LOADING CSV-OUTPUT METRIC.PY <<<")
import os
import csv
import json
import numpy as np
from sklearn.metrics import ndcg_score
from urllib.parse import unquote

# ==========================================
# 1. PATHS
# ==========================================
base_dir = "/content/KGSUMM/Task4/Dataset_ontology"
output_dir = os.path.join(base_dir, "Outputs")
score_file_path = os.path.join(output_dir, "final_scores.csv")  # CHANGED TO CSV

PATHS = {
    "WIKIES": {
        "gt": "/content/KGSUMM/Task4/Dataset_ontology/WIKIES/gold.json",
        "map_source": "/content/KGSUMM/Task4/Dataset_ontology/WIKIES/data/data.json", 
        "csv_backup": "/content/KGSUMM/Task4/Dataset_ontology/WIKIES/data/seed_nodes",
        "type": "wikies"
    },
    "DBpedia": {
        "gt": "/content/KGSUMM/Task1/Unsupervised_Learning_IRES_Model/ESBM_v1.2/Temp/dbpedia_groundtruth.json",
        "map_source": "/content/KGSUMM/Task4/Dataset_ontology/ESBM/Tool/ESBM-groundtruth/elist.txt",
        "type": "esbm"
    },
    "LMDB": {
        "gt": "/content/KGSUMM/Task1/Unsupervised_Learning_IRES_Model/ESBM_v1.2/Temp/lmdb_groundtruth.json",
        "map_source": "/content/KGSUMM/Task4/Dataset_ontology/ESBM/Tool/ESBM-groundtruth/elist.txt",
        "type": "esbm"
    },
    "FACES": {
        "gt": "/content/KGSUMM/Task1/Unsupervised_Learning_IRES_Model/FACES/Temp/faces_groundtruth.json",
        "map_source": "/content/KGSUMM/Task4/Dataset_ontology/FACES/elist.txt",
        "type": "esbm"
    }
}

# ==========================================
# 2. UTILS
# ==========================================
def normalize(text):
    if not text: return ""
    text = unquote(str(text))
    for p in ["http://dbpedia.org/resource/", "http://data.linkedmdb.org/resource/film/", 
              "http://data.linkedmdb.org/resource/actor/", "http://xmlns.com/foaf/0.1/", "<", ">"]:
        text = text.replace(p, "")
    return text.strip().replace("_", " ").lower()

def calculate_metrics_at_k(pred_list, gt_set, k):
    norm_preds = [normalize(p) for p in pred_list[:k]]
    norm_gt = {normalize(g) for g in gt_set}
    
    if not norm_preds or not norm_gt: return 0.0, 0.0, 0.0
    
    # HIT Calculation
    hits = len(set(norm_preds).intersection(norm_gt))
    
    # Precision & Recall
    prec = hits / len(norm_preds)
    rec = hits / len(norm_gt)
    f1 = 2 * (prec * rec) / (prec + rec) if (prec + rec) > 0 else 0.0
    
    # MAP
    ap = 0.0; hit_count = 0
    for i, p in enumerate(norm_preds):
        if p in norm_gt:
            hit_count += 1; ap += hit_count / (i + 1)
    map_score = ap / min(len(norm_gt), k) if min(len(norm_gt), k) > 0 else 0.0

    # NDCG
    relevance = [1 if p in norm_gt else 0 for p in norm_preds]
    if len(relevance) < k: relevance += [0] * (k - len(relevance))
    ideal = sorted(relevance, reverse=True)
    ndcg = ndcg_score([ideal], [relevance], k=k) if sum(ideal) > 0 else 0.0
    
    return f1, map_score, ndcg

# ==========================================
# 3. ID MAPPING
# ==========================================
def load_id_map(dataset_name, config):
    mapping = {}
    
    # WIKIES: Hybrid CSV + JSON scanning
    if config["type"] == "wikies":
        # Check CSVs first
        csv_path = config.get("csv_backup")
        if csv_path and os.path.exists(csv_path):
            for root, _, files in os.walk(csv_path):
                for file in files:
                    if file.endswith(".csv"):
                        try:
                            with open(os.path.join(root, file), 'r', encoding='utf-8', errors='ignore') as f:
                                reader = csv.reader(f)
                                headers = next(reader, [])
                                try:
                                    id_idx = headers.index('id') if 'id' in headers else 0
                                    name_idx = headers.index('name') if 'name' in headers else (2 if len(headers)>2 else 1)
                                    for row in reader:
                                        if len(row) > max(id_idx, name_idx):
                                            mapping[normalize(row[id_idx])] = normalize(row[name_idx])
                                except: pass
                        except: pass
        
    # ESBM/FACES: elist.txt
    elif config["type"] == "esbm":
        path = config["map_source"]
        if os.path.exists(path):
            with open(path, 'r', encoding='utf-8', errors='ignore') as f:
                for line in f:
                    parts = line.strip().split()
                    if len(parts) < 2 or "eid" in parts[0]: continue
                    eid = normalize(parts[0])
                    uri = next((p for p in parts if p.startswith("http")), None)
                    if uri: mapping[eid] = normalize(uri)

    print(f"   [{dataset_name}] Mapped {len(mapping)} IDs.")
    return mapping

# ==========================================
# 4. LOADERS
# ==========================================
def load_predictions(filepath, id_map):
    preds = {}
    curr_key = None
    if not os.path.exists(filepath): return {}
    with open(filepath, 'r', encoding='utf-8', errors='ignore') as f:
        for line in f:
            line = line.strip()
            if line.startswith("ENTITY:"):
                raw = normalize(line.split(":", 1)[-1])
                curr_key = id_map.get(raw, raw) 
                preds[curr_key] = []
            elif line.startswith("->") and curr_key:
                raw_val = line.replace("->", "").strip()
                # If prediction is an ID, translate it. If it's a name, normalize it.
                val = id_map.get(normalize(raw_val), normalize(raw_val))
                preds[curr_key].append(val)
    return preds

def load_ground_truth_json(path, id_map):
    gt = {}
    if not os.path.exists(path): return {}
    try:
        with open(path, 'r', encoding='utf-8') as f:
            data = json.load(f)
            
            def add_entry(k, v_list):
                # Translate Key (ID -> Name)
                norm_k = normalize(k)
                final_k = id_map.get(norm_k, norm_k)
                
                clean_set = set()
                for item in v_list:
                    # CASE A: Item is a Triple ["S", "P", "O"] -> Extract "O"
                    if isinstance(item, list) and len(item) >= 3:
                        target = item[2]
                    # CASE B: Item is a String -> Use as is
                    else:
                        target = item
                    
                    # Translate Object (ID -> Name)
                    norm_target = normalize(target)
                    final_target = id_map.get(norm_target, norm_target)
                    clean_set.add(final_target)
                
                gt[final_k] = clean_set

            if isinstance(data, dict):
                for k, v in data.items(): add_entry(k, v)
            elif isinstance(data, list):
                for item in data:
                    k = item.get('entity') or item.get('subject')
                    v = item.get('summary') or item.get('objects')
                    if k and v: add_entry(k, v)
    except: pass
    return gt

# ==========================================
# 5. EXECUTION
# ==========================================
def run_evaluation():
    # SETUP CSV WRITER
    with open(score_file_path, "w", newline='', encoding='utf-8') as csvfile:
        fieldnames = ['Dataset', 'K', 'F1', 'MAP', 'NDCG', 'Matched_Entities']
        writer = csv.DictWriter(csvfile, fieldnames=fieldnames)
        writer.writeheader()
        
        print(f"\n{'='*40}\n      FINAL EVALUATION RESULTS\n{'='*40}\n")

        for name, config in PATHS.items():
            print(f"\n>> PROCESSING DATASET: {name}")
            
            id_map = load_id_map(name, config)
            preds = load_predictions(f"{output_dir}/{name}_output.txt", id_map)
            gt = load_ground_truth_json(config["gt"], id_map)
            
            # Match Keys
            common = set(preds.keys()).intersection(set(gt.keys())) - {"", "unnamed: 0"}
            msg = f"   Matching Entities: {len(common)} (Preds: {len(preds)}, GT: {len(gt)})"
            print(msg)
            
            if not common: 
                writer.writerow({'Dataset': name, 'K': 5, 'F1': 0.0, 'MAP': 0.0, 'NDCG': 0.0, 'Matched_Entities': 0})
                writer.writerow({'Dataset': name, 'K': 10, 'F1': 0.0, 'MAP': 0.0, 'NDCG': 0.0, 'Matched_Entities': 0})
                continue

            s5 = {'f1':[], 'map':[], 'ndcg':[]}
            s10 = {'f1':[], 'map':[], 'ndcg':[]}
            
            for entity in common:
                f, m, n = calculate_metrics_at_k(preds[entity], gt[entity], k=5)
                s5['f1'].append(f); s5['map'].append(m); s5['ndcg'].append(n)
                f, m, n = calculate_metrics_at_k(preds[entity], gt[entity], k=10)
                s10['f1'].append(f); s10['map'].append(m); s10['ndcg'].append(n)
            
            f1_5, map_5, ndcg_5 = np.mean(s5['f1']), np.mean(s5['map']), np.mean(s5['ndcg'])
            f1_10, map_10, ndcg_10 = np.mean(s10['f1']), np.mean(s10['map']), np.mean(s10['ndcg'])

            print(f"   [TOP-5]  F1: {f1_5:.4f} | MAP: {map_5:.4f} | NDCG: {ndcg_5:.4f}")
            print(f"   [TOP-10] F1: {f1_10:.4f} | MAP: {map_10:.4f} | NDCG: {ndcg_10:.4f}")

            writer.writerow({
                'Dataset': name, 'K': 5, 
                'F1': f1_5, 'MAP': map_5, 'NDCG': ndcg_5, 
                'Matched_Entities': len(common)
            })
            writer.writerow({
                'Dataset': name, 'K': 10, 
                'F1': f1_10, 'MAP': map_10, 'NDCG': ndcg_10, 
                'Matched_Entities': len(common)
            })

    print(f"\n📄 Scores saved to: {score_file_path}")

if __name__ == "__main__":
    run_evaluation()
