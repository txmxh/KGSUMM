import os
import csv
import json
import xml.etree.ElementTree as ET
import numpy as np
from sklearn.metrics import ndcg_score
from urllib.parse import unquote

base_dir = "/content/KGSUMM/Task4/Dataset_ontology"
output_dir = os.path.join(base_dir, "Outputs")
score_file_path = os.path.join(output_dir, "final_scores.txt")

def normalize_uri(uri):
    if not uri: return ""
    uri = unquote(uri)
    for prefix in ["http://dbpedia.org/resource/", "http://data.linkedmdb.org/resource/film/", 
                   "http://data.linkedmdb.org/resource/actor/", "http://xmlns.com/foaf/0.1/", "<", ">"]:
        uri = uri.replace(prefix, "")
    # KEY FIX: Lowercase and replace underscores with spaces for flexible matching
    return uri.strip().replace("_", " ").lower()

def calculate_metrics_at_k(pred_list, gt_set, k):
    norm_preds = [normalize_uri(p) for p in pred_list[:k]]
    norm_gt = {normalize_uri(g) for g in gt_set}
    
    if not norm_preds or not norm_gt: return 0.0, 0.0, 0.0
    
    hits = len(set(norm_preds).intersection(norm_gt))
    prec = hits / len(norm_preds)
    rec = hits / len(norm_gt)
    f1 = 2 * (prec * rec) / (prec + rec) if (prec + rec) > 0 else 0.0
    
    ap = 0.0
    hit_count = 0
    for i, p in enumerate(norm_preds):
        if p in norm_gt:
            hit_count += 1
            ap += hit_count / (i + 1)
    map_score = ap / min(len(norm_gt), k) if min(len(norm_gt), k) > 0 else 0.0

    relevance = [1 if p in norm_gt else 0 for p in norm_preds]
    if len(relevance) < k: relevance += [0] * (k - len(relevance))
    ideal = sorted(relevance, reverse=True)
    ndcg = ndcg_score([ideal], [relevance], k=k) if sum(ideal) > 0 else 0.0
        
    return f1, map_score, ndcg

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

def load_wikies_recursive(root_path):
    gt = {}
    print(f"   🔎 Scanning WIKIES test files in: {root_path}")
    for root, dirs, files in os.walk(root_path):
        for file in files:
            if "test.csv" in file:
                try:
                    with open(os.path.join(root, file), 'r') as f:
                        reader = csv.reader(f)
                        for row in reader:
                            if len(row) >= 3:
                                # Store BOTH Name and ID to ensure matching works
                                s_name = normalize_uri(row[2])
                                s_id = normalize_uri(row[0]) 
                                
                                if s_name not in gt: gt[s_name] = set()
                                for item in row: gt[s_name].add(normalize_uri(item))
                                
                                # Alias the ID to point to the same set
                                if s_id not in gt: gt[s_id] = gt[s_name]
                except: continue
    return gt

def load_standard_gold(filepath):
    gt = {}
    if not filepath or not os.path.exists(filepath): return {}
    try:
        if filepath.endswith(".xml"):
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
        elif filepath.endswith(".json"):
            with open(filepath, 'r') as f:
                data = json.load(f)
                if isinstance(data, dict):
                    for s, summ in data.items():
                        gt[normalize_uri(s)] = set(normalize_uri(x) for x in summ)
    except: pass
    return gt

def run_evaluation():
    datasets = {
        "WIKIES": {
            "pred": f"{output_dir}/WIKIES_output.txt",
            "path": f"{base_dir}/WIKIES/data/seed_nodes",
            "mode": "wikies"
        },
        "DBpedia": {
            "pred": f"{output_dir}/DBpedia_output.txt",
            "path": f"{base_dir}/ESBM/Datasets/dbpedia_data/gold_standard.xml",
            "mode": "standard"
        },
        "FACES": {
            "pred": f"{output_dir}/FACES_output.txt",
            "path": f"{base_dir}/FACES/faces_data/gold.json",
            "mode": "standard"
        },
        "LMDB": {
            "pred": f"{output_dir}/LMDB_output.txt",
            "path": f"{base_dir}/ESBM/Datasets/lmdb_data/gold_standard.xml",
            "mode": "standard"
        }
    }
    
    with open(score_file_path, "w") as out_f:
        header = "\n" + "="*40 + "\n      FINAL EVALUATION RESULTS\n" + "="*40 + "\n"
        print(header); out_f.write(header)

        for name, info in datasets.items():
            print(f"\n>> PROCESSING DATASET: {name}")
            out_f.write(f"\n>> PROCESSING DATASET: {name}\n")
            
            preds = load_predictions(info['pred'])
            if not preds:
                print("   ⚠️ Predictions missing."); continue

            if info['mode'] == "wikies":
                gt = load_wikies_recursive(info['path'])
            else:
                gt = load_standard_gold(info['path'])

            if not gt:
                msg = f"   ⚠️ Skipping: MISSING REAL GOLD STANDARD FILE in {info['path']}\n"
                print(msg); out_f.write(msg)
                continue

            common = set(preds.keys()).intersection(set(gt.keys()))
            msg_match = f"   Matching Entities: {len(common)} (Preds: {len(preds)}, GT: {len(gt)})\n"
            print(msg_match); out_f.write(msg_match)
            
            if not common:
                print("   ❌ No matches. Check if IDs align."); continue

            s5 = {'f1':[],'map':[],'ndcg':[]}; s10 = {'f1':[],'map':[],'ndcg':[]}
            for entity in common:
                f, m, n = calculate_metrics_at_k(preds[entity], gt[entity], k=5)
                s5['f1'].append(f); s5['map'].append(m); s5['ndcg'].append(n)
                f, m, n = calculate_metrics_at_k(preds[entity], gt[entity], k=10)
                s10['f1'].append(f); s10['map'].append(m); s10['ndcg'].append(n)
            
            results = (
                f"   [TOP-5]  F1: {np.mean(s5['f1']):.4f} | MAP: {np.mean(s5['map']):.4f} | NDCG: {np.mean(s5['ndcg']):.4f}\n"
                f"   [TOP-10] F1: {np.mean(s10['f1']):.4f} | MAP: {np.mean(s10['map']):.4f} | NDCG: {np.mean(s10['ndcg']):.4f}\n"
            )
            print(results); out_f.write(results)

if __name__ == "__main__":
    run_evaluation()