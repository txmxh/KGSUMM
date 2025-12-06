import sys
import os
import pandas as pd
import numpy as np
from sklearn.metrics import ndcg_score

# =============================================================
# 1. SETUP PATHS & IMPORTS (Auto-Finding)
# =============================================================

CURRENT_DIR = os.getcwd()

# Helper to find the folder containing 'classes' (for imports)
def find_classes_folder(start_dir):
    for root, dirs, files in os.walk(start_dir):
        if "classes" in dirs and os.path.exists(os.path.join(root, "classes", "dataset.py")):
            return root
    return None

# Attempt to find and append the code folder
code_root = find_classes_folder(CURRENT_DIR)
if code_root:
    if code_root not in sys.path:
        sys.path.append(code_root)
else:
    # Fallback to hardcoded structure
    sys.path.append(os.path.join(CURRENT_DIR, "Supervised_Model_ESLM", "ESBM_v1.2"))

try:
    from classes.dataset import ESBenchmark
except ImportError:
    print("CRITICAL ERROR: Could not import 'classes.dataset'.")
    print("Ensure you are running this from the Task2 directory.")
    sys.exit(1)

# =============================================================
# 2. CONFIGURATION
# =============================================================

# Filename to save the results to
OUTPUT_CSV_FILE = "ESLM_score_metrices.csv"

# Where to look for prediction CSVs
PRED_SEARCH_DIRS = [
    os.path.join(CURRENT_DIR, "Supervised_Model_ESLM", "Supervised_Scores"),
    os.path.join(CURRENT_DIR, "Supervised_Model_ESLM", "Predictions"),
    os.path.join(CURRENT_DIR, "Supervised_Model_ESLM", "ESBM_v1.2", "Outputs"),
    os.path.join(CURRENT_DIR, "Outputs"),
    os.path.join(CURRENT_DIR, "Supervised_Predictions")
]

DATASETS = ["dbpedia", "faces", "lmdb"]

# Helper to find the actual data folders (for Gold Standard)
def find_data_path(ds_name):
    target = f"{ds_name}_data"
    for root, dirs, files in os.walk(CURRENT_DIR):
        if target in dirs:
            return os.path.dirname(os.path.join(root, target))
    return None

FILES = [
    ("dbpedia", "predictions_ESLM_t5_dbpedia_top5_10.csv"),
    ("lmdb",    "predictions_ESLM_t5_lmdb_top5_10.csv"),
    ("faces",   "predictions_ESLM_t5_faces_top5_10.csv")
]

# =============================================================
# 3. HELPERS
# =============================================================

def discover_prediction_files():
    found = {ds: [] for ds in DATASETS}

    for d in PRED_SEARCH_DIRS:
        if not os.path.exists(d):
            continue

        for f in os.listdir(d):
            if not f.endswith(".csv"):
                continue
            if "predictions" not in f.lower():
                continue

            for ds in DATASETS:
                if ds in f.lower():
                    found[ds].append(os.path.join(d, f))
    return found

def parse_preds(s):
    if pd.isna(s): return []
    s = str(s).strip()
    if s in ["", "[]", "nan"]: return []
    s = s.replace('[', '').replace(']', '')
    try:
        return list(map(int, s.split(",")))
    except ValueError:
        return []

def calc_map(preds, gold, k):
    if not gold: return 0.0
    hits = 0
    score = 0.0
    active_preds = preds[:k]
    for i, p in enumerate(active_preds):
        if p in gold:
            hits += 1
            score += hits / (i + 1)
    return score / min(len(gold), k)

# =============================================================
# 4. MAIN EXECUTION
# =============================================================

def run_metrics():
    pred_files = discover_prediction_files()

    # List to store results for CSV
    results_data = []

    print(f"\n{'DATASET':<10} | {'TOP-K':<5} | {'NDCG':<8} | {'MAP':<8}")
    print("-" * 45)

    for ds_name in DATASETS:
        files = pred_files[ds_name]
        if not files:
            continue

        real_db_path = find_data_path(ds_name)
        if not real_db_path:
            print(f"WARNING: Could not find data folder for {ds_name}. Scores will be 0.0")

        for csv_path in files:
            try:
                df = pd.read_csv(csv_path)
            except: continue

            if "predicted_indices" not in df.columns:
                continue
            
            for k in [5, 10]:
                if 'topk' in df.columns:
                    subset = df[df['topk'] == k]
                else:
                    subset = df 
                
                if subset.empty: continue

                ndcg_vals, map_vals = [], []
                
                try:
                    benchmark = ESBenchmark(ds_name, 6, k, False)
                    if real_db_path:
                        if ds_name == 'faces':
                            benchmark.db_path = os.path.join(real_db_path, "faces_data")
                        elif ds_name == 'dbpedia':
                            benchmark.db_path = os.path.join(real_db_path, "dbpedia_data")
                        elif ds_name == 'lmdb':
                            benchmark.db_path = os.path.join(real_db_path, "lmdb_data")
                except:
                    continue

                for _, row in subset.iterrows():
                    preds = parse_preds(row["predicted_indices"])
                    eid = row["entity_id"]

                    gold = set()
                    for hop in range(6):
                        try:
                            if hasattr(benchmark, 'get_gold_indices'):
                                gold.update(benchmark.get_gold_indices(eid, hop))
                        except: pass

                    if not preds or not gold:
                        ndcg_vals.append(0.0)
                        map_vals.append(0.0)
                        continue

                    relevance = [1 if p in gold else 0 for p in preds[:k]]
                    if len(relevance) > 0 and sum(relevance) > 0:
                        scores = list(range(len(preds[:k]), 0, -1))
                        val_ndcg = ndcg_score([relevance], [scores], k=k)
                    else:
                        val_ndcg = 0.0

                    val_map = calc_map(preds, gold, k)
                    
                    ndcg_vals.append(val_ndcg)
                    map_vals.append(val_map)

                if ndcg_vals:
                    mean_ndcg = np.mean(ndcg_vals)
                    mean_map = np.mean(map_vals)
                    
                    # Print to console
                    print(f"{ds_name:<10} | {k:<5} | {mean_ndcg:.4f}   | {mean_map:.4f}")
                    
                    # Add to list for saving
                    results_data.append({
                        "Dataset": ds_name,
                        "Top-K": k,
                        "NDCG": round(mean_ndcg, 4),
                        "MAP": round(mean_map, 4)
                    })

    # --- SAVE TO CSV ---
    if results_data:
        final_df = pd.DataFrame(results_data)
        final_df.to_csv(OUTPUT_CSV_FILE, index=False)
        print(f"\n✅ Results saved to: {OUTPUT_CSV_FILE}")

if __name__ == "__main__":
    run_metrics()