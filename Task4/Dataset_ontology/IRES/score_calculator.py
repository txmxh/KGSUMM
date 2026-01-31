import os
import csv
import json
import xml.etree.ElementTree as ET
import numpy as np
from sklearn.metrics import ndcg_score
from urllib.parse import unquote

# ==========================================
# 1. SETUP & UTILS
# ==========================================
base_dir = "/content/KGSUMM/Task4/Dataset_ontology"
output_dir = os.path.join(base_dir, "Outputs")
score_file_path = os.path.join(output_dir, "final_scores.txt")

def normalize_uri(uri):
    """Standardizes URIs to matching strings."""
    if not uri: return ""
    uri = unquote(uri)
    for prefix in ["http://dbpedia.org/resource/", "http://data.linkedmdb.org/resource/film/", 
                   "http://data.linkedmdb.org/resource/actor/", "http://xmlns.com/foaf/0.1/", "<", ">"]:
        uri = uri.replace(prefix, "")
    return uri.strip().replace("_", " ")

# ==========================================
# 2. METRIC MATH
# ==========================================
def calculate_metrics_at_k(pred_list, gt_set, k):
    norm_preds = [normalize_uri(p) for p in pred_list[:k]]
    norm_gt = {normalize_uri(g) for g in gt_set}
    
    if not norm_preds or not norm_gt: return 0.0, 0.0, 0.0
    
    # F1
    hits = len(set(norm_preds).intersection(norm_gt))
    prec = hits / len(norm_preds)
    rec = hits / len(norm_gt)
    f1 = 2 * (prec * rec) / (prec + rec) if (prec + rec) > 0 else 0.0
    
    # MAP
    ap = 0.0
    hit_count = 0
    for i, p in enumerate(norm_preds):
        if p in norm_gt:
            hit_count += 1
            ap += hit_count / (i + 1)
    map_score = ap / min(len(norm_gt), k) if min(len(norm_gt), k) > 0 else 0.0

    # NDCG
    relevance = [1 if p in norm_gt else 0 for p in norm_preds]
    if len(relevance) < k: relevance += [0] * (k - len(relevance))
    ideal = sorted(relevance, reverse=True)
    ndcg = ndcg_score([ideal], [relevance], k=k) if sum(ideal) > 0 else 0.0
        
    return f1, map_score, ndcg

# ==========================================
# 3. LOADERS
# ==========================================
def load_predictions(filepath):
    preds = {}
    current_entity = None
    if not os.path.exists(filepath): return {}
    with open(filepath, 'r', encoding='utf-8', errors='ignore') as f:
        for line in f:
            line = line.strip()
            if line.startswith("ENTITY:"):
                raw = line.split(":", 1)[-1].strip()
                if "Unnamed" in raw or "name" in raw: 
                    current_entity = None
                    continue
                current_entity = normalize_uri(raw)
                preds[current_entity] = []
            elif line.startswith("->") and current_entity:
                val = line.replace("->", "").strip()
                preds[current_entity].append(val)
    return preds

def load_gold_standard(filepath, dataset_type):
    gt = {}
    if not filepath or not os.path.exists(filepath): 
        return {}
    
    print(f"   🔎 Parsing Gold Standard: {os.path.basename(filepath)}")
    
    try:
        # A. CSV Loader (WIKIES)
        if filepath.endswith(".csv") or dataset_type == "WIKIES":
            # If it's a directory (WIKIES folder), scan for test.csv
            if os.path.isdir(filepath):
                for root, dirs, files in os.walk(filepath):
                    for file in files:
                        if "test.csv" in file:
                            with open(os.path.join(root, file), 'r') as f:
                                reader = csv.reader(f)
                                for row in reader:
                                    if len(row) >= 3:
                                        s, o = normalize_uri(row[0]), row[2]
                                        if s not in gt: gt[s] = set()
                                        gt[s].add(o)
            # If it's a single file
            else:
                with open(filepath, 'r') as f:
                    reader = csv.reader(f)
                    for row in reader:
                        if len(row) >= 3:
                            s, o = normalize_uri(row[0]), row[2]
                            if s not in gt: gt[s] = set()
                            gt[s].add(o)

        # B. XML Loader (DBpedia / LMDB / ESBM)
        elif filepath.endswith(".xml"):
            tree = ET.parse(filepath)
            root = tree.getroot()
            for topic in root.findall(".//topic") or root.findall(".//entity"):
                uri = topic.attrib.get('uri') or topic.attrib.get('id')
                if uri:
                    norm_s = normalize_uri(uri)
                    gt[norm_s] = set()
                    for triple in topic.findall(".//triple") or topic.findall(".//statement"):
                        obj = triple.attrib.get('object') or triple.text
                        if obj: gt[norm_s].add(normalize_uri(obj))

        # C. JSON Loader (FACES)
        elif filepath.endswith(".json"):
            with open(filepath, 'r') as f:
                data = json.load(f)
                if isinstance(data, list):
                    for item in data:
                        s = item.get('entity') or item.get('subject')
                        summ = item.get('summary') or item.get('objects')
                        if s and summ:
                            gt[normalize_uri(s)] = set(normalize_uri(x) for x in summ)
                elif isinstance(data, dict):
                    for s, summ in data.items():
                        gt[normalize_uri(s)] = set(normalize_uri(x) for x in summ)

    except Exception as e:
        print(f"   ⚠️ Error reading GT file: {e}")
        
    return gt

# ==========================================
# 4. EXECUTION
# ==========================================
def run_evaluation():
    # --- IMPORTANT: UPDATE THESE PATHS ---
    # You must provide the EXACT path to the Gold Standard file for each dataset.
    # For WIKIES, the folder path works because it scans for test.csv.
    # For others, point to the specific .xml or .json file.
    
    datasets = {
        "WIKIES": {
            "gt_path": f"{base_dir}/WIKIES/data/seed_nodes" 
        },
        "DBpedia": {
            # FIND YOUR XML FILE (e.g., gold_standard.xml) AND PASTE PATH HERE:
            "gt_path": f"{base_dir}/ESBM/Datasets/dbpedia_data/gold_standard.xml" 
        },
        "FACES": {
            # FIND YOUR JSON FILE AND PASTE PATH HERE:
            "gt_path": f"{base_dir}/FACES/faces_data/gold.json" 
        },
        "LMDB": {
            # FIND YOUR XML FILE AND PASTE PATH HERE:
            "gt_path": f"{base_dir}/ESBM/Datasets/lmdb_data/gold_standard.xml" 
        }
    }
    
    with open(score_file_path, "w") as out_f:
        header = "\n" + "="*40 + "\n      FINAL EVALUATION RESULTS\n" + "="*40 + "\n"
        print(header)
        out_f.write(header)

        for name, info in datasets.items():
            print(f"\n>> PROCESSING DATASET: {name}")
            out_f.write(f"\n>> PROCESSING DATASET: {name}\n")
            
            # 1. Load Preds
            pred_file = f"{output_dir}/{name}_output.txt"
            preds = load_predictions(pred_file)
            if not preds:
                msg = f"   [Error] Predictions not found at {pred_file}\n"
                print(msg); out_f.write(msg)
                continue

            # 2. Load GT
            gt = load_gold_standard(info['gt_path'], name)

            # 3. Score
            if gt and preds:
                scores_5 = {'f1': [], 'map': [], 'ndcg': []}
                scores_10 = {'f1': [], 'map': [], 'ndcg': []}
                
                common = set(preds.keys()).intersection(set(gt.keys()))
                msg_match = f"   Matching Entities: {len(common)} (Preds: {len(preds)}, GT: {len(gt)})\n"
                print(msg_match); out_f.write(msg_match)
                
                if not common:
                    msg_fail = "   ❌ No overlapping entities found (IDs might not match).\n"
                    print(msg_fail); out_f.write(msg_fail)
                    continue

                for entity in common:
                    # @5
                    f1_5, map_5, ndcg_5 = calculate_metrics_at_k(preds[entity], gt[entity], k=5)
                    scores_5['f1'].append(f1_5); scores_5['map'].append(map_5); scores_5['ndcg'].append(ndcg_5)
                    # @10
                    f1_10, map_10, ndcg_10 = calculate_metrics_at_k(preds[entity], gt[entity], k=10)
                    scores_10['f1'].append(f1_10); scores_10['map'].append(map_10); scores_10['ndcg'].append(ndcg_10)
                
                results = (
                    f"   ----------------------------------------\n"
                    f"   [TOP-5]\n"
                    f"   ✅ F1:   {np.mean(scores_5['f1']):.4f} | MAP:  {np.mean(scores_5['map']):.4f} | NDCG: {np.mean(scores_5['ndcg']):.4f}\n"
                    f"   [TOP-10]\n"
                    f"   ✅ F1:   {np.mean(scores_10['f1']):.4f} | MAP:  {np.mean(scores_10['map']):.4f} | NDCG: {np.mean(scores_10['ndcg']):.4f}\n"
                    f"   ----------------------------------------\n"
                )
                print(results); out_f.write(results)
            else:
                msg_skip = f"   ⚠️ Skipping: Could not load Gold Standard from {info['gt_path']}\n"
                print(msg_skip); out_f.write(msg_skip)

    print(f"\n📄 Scores saved to: {score_file_path}")

if __name__ == "__main__":
    run_evaluation()
