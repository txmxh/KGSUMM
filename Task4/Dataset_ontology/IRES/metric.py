
print(">>> LOADING 'RAW MATCH' METRIC.PY <<<")
import os
import json
import numpy as np
from sklearn.metrics import ndcg_score
from urllib.parse import unquote

# ==========================================
# 1. CONFIG
# ==========================================
base_dir = "/content/KGSUMM/Task4/Dataset_ontology"
output_dir = os.path.join(base_dir, "Outputs")
score_file_path = os.path.join(output_dir, "final_scores.txt")

# We only need the Output file and the GT file. No maps.
PATHS = {
    "WIKIES": {
        "gt": "/content/KGSUMM/Task4/Dataset_ontology/WIKIES/gold.json",
        "pred": os.path.join(output_dir, "WIKIES_output.txt")
    },
    "FACES": {
        "gt": "/content/KGSUMM/Task1/Unsupervised_Learning_IRES_Model/FACES/Temp/faces_groundtruth.json",
        "pred": os.path.join(output_dir, "FACES_output.txt")
    },
    "DBpedia": {
        "gt": "/content/KGSUMM/Task1/Unsupervised_Learning_IRES_Model/ESBM_v1.2/Temp/dbpedia_groundtruth.json",
        "pred": os.path.join(output_dir, "DBpedia_output.txt")
    },
    "LMDB": {
        "gt": "/content/KGSUMM/Task1/Unsupervised_Learning_IRES_Model/ESBM_v1.2/Temp/lmdb_groundtruth.json",
        "pred": os.path.join(output_dir, "LMDB_output.txt")
    }
}

# ==========================================
# 2. HELPER: NORMALIZE STRINGS
# ==========================================
def normalize(text):
    """
    Strips URIs to ensure 'http://dbpedia.org/resource/Barack_Obama' matches 'Barack_Obama'.
    Also handles 'Q123' staying 'Q123'.
    """
    if not text: return ""
    text = unquote(str(text))
    # Remove common prefixes
    prefixes = [
        "http://dbpedia.org/resource/", "http://data.linkedmdb.org/resource/film/", 
        "http://data.linkedmdb.org/resource/actor/", "http://xmlns.com/foaf/0.1/", 
        "http://www.w3.org/2000/01/rdf-schema#", "http://", "https://"
    ]
    for p in prefixes:
        text = text.replace(p, "")
    
    # Remove brackets < >
    text = text.replace("<", "").replace(">", "")
    
    # Lowercase and strip
    return text.strip().replace("_", " ").lower()

# ==========================================
# 3. LOADERS
# ==========================================
def load_predictions(filepath):
    preds = {}
    curr_key = None
    if not os.path.exists(filepath): return {}
    
    with open(filepath, 'r', encoding='utf-8', errors='ignore') as f:
        for line in f:
            line = line.strip()
            if line.startswith("ENTITY:"):
                # Clean the Entity ID (e.g., http://.../Obama -> obama)
                raw = line.split(":", 1)[-1]
                curr_key = normalize(raw)
                preds[curr_key] = []
            elif line.startswith("->") and curr_key:
                # Clean the Summary Item
                val = line.replace("->", "").strip()
                preds[curr_key].append(normalize(val))
    return preds

def load_ground_truth(path):
    gt = {}
    if not os.path.exists(path): return {}
    
    try:
        with open(path, 'r', encoding='utf-8') as f:
            data = json.load(f)
            
            # Inner helper to process a single entity's summary list
            def process_summary(summary_list):
                clean_set = set()
                for item in summary_list:
                    target = ""
                    # If Triple ["S", "P", "O"], take "O"
                    if isinstance(item, list) and len(item) >= 3:
                        target = item[2]
                    # If String, take String
                    elif isinstance(item, str):
                        target = item
                    
                    if target:
                        clean_set.add(normalize(target))
                return clean_set

            # Dict format: {"Entity": ["fact", ...]}
            if isinstance(data, dict):
                for k, v in data.items():
                    gt[normalize(k)] = process_summary(v)
            
            # List format: [{"entity": "...", "summary": [...]}]
            elif isinstance(data, list):
                for item in data:
                    k = item.get('entity') or item.get('subject')
                    v = item.get('summary') or item.get('objects')
                    if k and v:
                        gt[normalize(k)] = process_summary(v)
    except: pass
    return gt

# ==========================================
# 4. SCORING MATH
# ==========================================
def calculate_scores(pred_list, gt_set, k):
    # Safe slicing
    current_preds = pred_list[:k]
    
    if not current_preds or not gt_set:
        return 0.0, 0.0, 0.0

    # F1
    hits = len(set(current_preds).intersection(gt_set))
    prec = hits / len(current_preds)
    rec = hits / len(gt_set)
    f1 = 2 * (prec * rec) / (prec + rec) if (prec + rec) > 0 else 0.0
    
    # MAP
    ap = 0.0
    hit_count = 0
    for i, p in enumerate(current_preds):
        if p in gt_set:
            hit_count += 1
            ap += hit_count / (i + 1)
    # Standard MAP definition: divide by min(len(gt), k) OR len(gt)
    # We use min(len(gt), k) to correspond to Precision@K constraints
    map_score = ap / min(len(gt_set), k) if min(len(gt_set), k) > 0 else 0.0

    # NDCG
    relevance = [1 if p in gt_set else 0 for p in current_preds]
    # Pad with zeros if prediction is shorter than k
    if len(relevance) < k: relevance += [0] * (k - len(relevance))
    
    ideal = sorted(relevance, reverse=True)
    current_ndcg = ndcg_score([ideal], [relevance], k=k) if sum(ideal) > 0 else 0.0
    
    return f1, map_score, current_ndcg

# ==========================================
# 5. MAIN RUNNER
# ==========================================
def run_evaluation():
    with open(score_file_path, "w") as out_f:
        header = "\n" + "="*40 + "\n      RAW MATCH EVALUATION RESULTS\n" + "="*40 + "\n"
        print(header); out_f.write(header)

        for name, paths in PATHS.items():
            print(f"\n>> PROCESSING DATASET: {name}")
            out_f.write(f"\n>> PROCESSING DATASET: {name}\n")
            
            # 1. Load
            preds = load_predictions(paths["pred"])
            gt = load_ground_truth(paths["gt"])
            
            if not preds: print("   ⚠️ Predictions empty or missing."); continue
            if not gt: print("   ⚠️ Ground Truth empty or missing."); continue

            # 2. Intersect Keys
            # Filter out garbage keys like 'unnamed: 0' or empty strings
            common_entities = set(preds.keys()).intersection(set(gt.keys()))
            common_entities = {x for x in common_entities if x and "unnamed" not in x}
            
            msg = f"   Matching Entities: {len(common_entities)} (Preds: {len(preds)}, GT: {len(gt)})\n"
            print(msg); out_f.write(msg)

            if not common_entities:
                # DEBUG: Show what the keys look like so we can debug mismatch
                p_sample = list(preds.keys())[:3]
                g_sample = list(gt.keys())[:3]
                print(f"   [DEBUG] Pred Keys: {p_sample}")
                print(f"   [DEBUG] GT Keys:   {g_sample}")
                continue

            # 3. Calculate Metrics
            s5 = {'f1':[], 'map':[], 'ndcg':[]}
            s10 = {'f1':[], 'map':[], 'ndcg':[]}
            
            for entity in common_entities:
                # Score @ 5
                f, m, n = calculate_scores(preds[entity], gt[entity], k=5)
                s5['f1'].append(f); s5['map'].append(m); s5['ndcg'].append(n)
                
                # Score @ 10
                f, m, n = calculate_scores(preds[entity], gt[entity], k=10)
                s10['f1'].append(f); s10['map'].append(m); s10['ndcg'].append(n)
            
            # 4. Report
            res = (
                f"   [TOP-5]  F1: {np.mean(s5['f1']):.4f} | MAP: {np.mean(s5['map']):.4f} | NDCG: {np.mean(s5['ndcg']):.4f}\n"
                f"   [TOP-10] F1: {np.mean(s10['f1']):.4f} | MAP: {np.mean(s10['map']):.4f} | NDCG: {np.mean(s10['ndcg']):.4f}\n"
            )
            print(res); out_f.write(res)

    print(f"\n📄 Scores saved to: {score_file_path}")

if __name__ == "__main__":
    run_evaluation()
