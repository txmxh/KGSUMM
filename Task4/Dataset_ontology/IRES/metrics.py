import os
import csv
import json
import xml.etree.ElementTree as ET
import numpy as np
from sklearn.metrics import ndcg_score
from urllib.parse import unquote

# ==========================================
# 1. SETUP
# ==========================================
base_dir = "/content/KGSUMM/Task4/Dataset_ontology"
output_dir = os.path.join(base_dir, "Outputs")
score_file_path = os.path.join(output_dir, "final_scores.txt")

# ==========================================
# 2. HELPER: URI NORMALIZER
# ==========================================
def normalize_uri(uri):
    if not uri: return ""
    uri = unquote(uri)
    for prefix in ["http://dbpedia.org/resource/", "http://data.linkedmdb.org/resource/film/", 
                   "http://data.linkedmdb.org/resource/actor/", "http://xmlns.com/foaf/0.1/", "<", ">"]:
        uri = uri.replace(prefix, "")
    return uri.strip().replace("_", " ")

# ==========================================
# 3. HELPER: FILE HUNTER
# ==========================================
def find_ground_truth(dataset_name, root_path):
    """Recursively searches for a potential Ground Truth file."""
    print(f"   🔎 Searching for {dataset_name} Gold Standard in: {root_path}...")
    
    candidates = []
    for root, dirs, files in os.walk(root_path):
        for file in files:
            # WIKIES: Look for test.csv
            if dataset_name == "WIKIES" and "test.csv" in file:
                return os.path.join(root, file) # Return the first test file found
            
            # OTHERS: Look for "gold", "ground", "standard"
            if dataset_name != "WIKIES":
                lower_f = file.lower()
                if ("gold" in lower_f or "ground" in lower_f or "standard" in lower_f) and \
                   (file.endswith(".xml") or file.endswith(".json")):
                    return os.path.join(root, file)
    
    return None

# ==========================================
# 4. SCORING MATH
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
# 5. EXECUTION
# ==========================================
def run_evaluation():
    datasets = ["WIKIES", "DBpedia", "FACES", "LMDB"]
    
    with open(score_file_path, "w") as out_f:
        header = "\n" + "="*40 + "\n      FINAL EVALUATION RESULTS\n" + "="*40 + "\n"
        print(header)
        out_f.write(header)

        for name in datasets:
            print(f"\n>> PROCESSING DATASET: {name}")
            out_f.write(f"\n>> PROCESSING DATASET: {name}\n")
            
            # 1. Load Predictions
            pred_file = f"{output_dir}/{name}_output.txt"
            if not os.path.exists(pred_file):
                msg = "   [Error] Prediction file not found. Did you run the training script?\n"
                print(msg); out_f.write(msg)
                continue
                
            preds = {}
            current_entity = None
            with open(pred_file, 'r', encoding='utf-8', errors='ignore') as f:
                for line in f:
                    line = line.strip()
                    if line.startswith("ENTITY:"):
                        current_entity = normalize_uri(line.split(":", 1)[-1].strip())
                        preds[current_entity] = []
                    elif line.startswith("->") and current_entity:
                        preds[current_entity].append(line.replace("->", "").strip())

            # 2. Find & Load Ground Truth
            # We search in the main dataset folder for this specific dataset
            search_path = f"{base_dir}/{name}" if name == "WIKIES" else f"{base_dir}"
            if name == "DBpedia" or name == "LMDB": search_path += "/ESBM"
            elif name == "FACES": search_path += "/FACES"
            
            gt_file = find_ground_truth(name, search_path)
            gt = {}
            
            if gt_file:
                print(f"   ✅ Found GT File: {gt_file}")
                try:
                    if name == "WIKIES": # CSV Loader
                        # For Wikies, we might need to scan ALL test.csv files, not just one.
                        # This simpler version tries the one it found.
                        with open(gt_file, 'r') as f:
                            reader = csv.reader(f)
                            for row in reader:
                                if len(row) >= 3:
                                    s = normalize_uri(row[0]); o = row[2]
                                    if s not in gt: gt[s] = set()
                                    gt[s].add(o)
                    elif gt_file.endswith(".xml"): # XML Loader
                        tree = ET.parse(gt_file)
                        root = tree.getroot()
                        for topic in root.findall(".//topic") or root.findall(".//entity"):
                            uri = topic.attrib.get('uri') or topic.attrib.get('id')
                            if uri:
                                norm_s = normalize_uri(uri)
                                gt[norm_s] = set()
                                for triple in topic.findall(".//triple") or topic.findall(".//statement"):
                                    obj = triple.attrib.get('object') or triple.text
                                    if obj: gt[norm_s].add(normalize_uri(obj))
                    elif gt_file.endswith(".json"): # JSON Loader
                        with open(gt_file, 'r') as f:
                            data = json.load(f)
                            if isinstance(data, dict):
                                for s, summ in data.items():
                                    gt[normalize_uri(s)] = set(normalize_uri(x) for x in summ)
                except Exception as e:
                    print(f"   ⚠️ Error reading GT file: {e}")
            else:
                msg = "   ❌ CRITICAL: No Gold Standard file found (gold*.xml/json) in dataset folder.\n"
                print(msg); out_f.write(msg)
                continue

            # 3. Calculate Scores
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

    print(f"\n📄 Scores saved to: {score_file_path}")

if __name__ == "__main__":
    run_evaluation()