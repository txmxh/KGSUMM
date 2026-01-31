import os
import csv
import torch
import numpy as np
import dgl
from sklearn.metrics import ndcg_score
from urllib.parse import unquote
import shutil

# ==========================================
# PART 1: DIAGNOSTIC & SETUP
# ==========================================
base_dir = "/content/KGSUMM/Task4/Dataset_ontology"
output_dir = os.path.join(base_dir, "Outputs")

print(f"📍 Checking Directory: {output_dir}")
if os.path.exists(output_dir):
    files = os.listdir(output_dir)
    print(f"✅ Found {len(files)} files: {files}")
else:
    print("❌ Output directory not found! Creating it now...")
    os.makedirs(output_dir, exist_ok=True)

# ==========================================
# PART 2: THE "FIXED" SCORING LOGIC
# ==========================================
def normalize_uri(uri):
    """Cleans URIs so 'http://dbpedia.org/resource/Baywatch' matches 'Baywatch'"""
    if not uri: return ""
    uri = unquote(uri)
    # Strip common prefixes found in your data
    for prefix in ["http://dbpedia.org/resource/", "http://data.linkedmdb.org/resource/film/", 
                   "http://data.linkedmdb.org/resource/actor/", "<", ">"]:
        uri = uri.replace(prefix, "")
    # Remove file extensions if they appear in IDs (common in some datasets)
    if uri.endswith(".jpg"): uri = uri.replace(".jpg", "")
    return uri.strip()

def calculate_metrics(pred_list, gt_set, k=5):
    # Normalize everything
    norm_preds = [normalize_uri(p) for p in pred_list[:k]]
    norm_gt = {normalize_uri(g) for g in gt_set}
    
    hits = len(set(norm_preds).intersection(norm_gt))
    if not norm_preds or not norm_gt: return 0.0, 0.0
    
    # F1
    prec = hits / len(norm_preds)
    rec = hits / len(norm_gt)
    f1 = 2 * (prec * rec) / (prec + rec) if (prec + rec) > 0 else 0.0
    
    # NDCG
    relevance = [1 if p in norm_gt else 0 for p in norm_preds]
    if len(relevance) < k: relevance += [0] * (k - len(relevance))
    
    ideal = sorted(relevance, reverse=True)
    ndcg = ndcg_score([ideal], [relevance]) if sum(ideal) > 0 else 0.0
    
    return f1, ndcg

def load_predictions(filepath):
    preds = {}
    current_entity = None
    if not os.path.exists(filepath): return {}
    
    with open(filepath, 'r', encoding='utf-8', errors='ignore') as f:
        for line in f:
            line = line.strip()
            if line.startswith("ENTITY:"):
                raw = line.split(":", 1)[-1].strip()
                # Skip the "Unnamed" garbage from bad CSV parsing
                if "Unnamed" in raw or "name" in raw: 
                    current_entity = None
                    continue
                current_entity = normalize_uri(raw)
                preds[current_entity] = []
            elif line.startswith("->") and current_entity:
                val = line.replace("->", "").strip()
                preds[current_entity].append(val)
    return preds

def load_wikies_gt(base_path):
    gt = {}
    print(f"🔎 Scanning Ground Truth in: {base_path}")
    for root, dirs, files in os.walk(base_path):
        for file in files:
            # Load Test CSVs as Ground Truth
            if "test.csv" in file:
                try:
                    with open(os.path.join(root, file), 'r') as f:
                        reader = csv.reader(f)
                        for row in reader:
                            if len(row) >= 3:
                                s = normalize_uri(row[0])
                                o = row[2] # Object
                                if s not in gt: gt[s] = set()
                                gt[s].add(o)
                except: continue
    return gt

# ==========================================
# PART 3: RUN EVALUATION
# ==========================================
# Define datasets
datasets = {
    "WIKIES": {
        "pred": f"{output_dir}/WIKIES_output.txt",
        "gt": f"{base_dir}/WIKIES/data/seed_nodes",
        "type": "wikies"
    },
    "DBpedia": {
        "pred": f"{output_dir}/DBpedia_output.txt",
        "gt": None, # Add path if you have gold_standard.json
        "type": "esbm"
    }
}

print("\n" + "="*40)
print("       EVALUATION RESULTS")
print("="*40)

for name, info in datasets.items():
    print(f"\n>> DATASET: {name}")
    
    # 1. Check if file exists (Double Check)
    if not os.path.exists(info['pred']):
        print(f"   ⚠️ Prediction file missing: {info['pred']}")
        continue

    # 2. Load
    preds = load_predictions(info['pred'])
    print(f"   Loaded {len(preds)} summaries.")
    
    # 3. Load GT
    gt = {}
    if info['type'] == 'wikies':
        gt = load_wikies_gt(info['gt'])
        print(f"   Loaded {len(gt)} Ground Truth entities.")
    
    # 4. Score
    if gt and preds:
        f1s, ndcgs = [], []
        # Find matches
        common = set(preds.keys()).intersection(set(gt.keys()))
        print(f"   Matching Entities: {len(common)}")
        
        for entity in common:
            f1, ndcg = calculate_metrics(preds[entity], gt[entity], k=5)
            f1s.append(f1)
            ndcgs.append(ndcg)
            
        if f1s:
            print(f"   ✅ F1-Score: {np.mean(f1s):.4f}")
            print(f"   ✅ NDCG@5:   {np.mean(ndcgs):.4f}")
        else:
            print("   ❌ No common entities found (Check ID formats).")
            if len(preds) > 0: print(f"      Sample Pred ID: {list(preds.keys())[0]}")
            if len(gt) > 0:    print(f"      Sample GT ID:   {list(gt.keys())[0]}")
    else:
        print("   (Skipping scores: Missing Ground Truth or Predictions)")