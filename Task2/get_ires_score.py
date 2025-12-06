import pandas as pd
import os
import numpy as np

# ==========================================
# CONFIGURATION
# ==========================================
# Path to your Unsupervised Scores folder
SCORE_DIR = "Unsupervised_Model_IRES/Unsupervised_Scores"
OUTPUT_FILENAME = "IRES_score_metrices.csv"

# Filename for FACES (which combines Top-5 and Top-10)
FACES_COMBINED_FILE = "detailed_scores_faces_top5_10.csv"
# ==========================================

def get_metrics(df):
    """Calculates mean metrics handling different column names"""
    # 1. F-Measure
    f1 = 0.0
    for col in ['fmeasure_norel', 'fmeasure', 'fmeasure_rel', 'F-Measure']:
        if col in df.columns:
            f1 = df[col].mean()
            break
            
    # 2. NDCG
    ndcg = 0.0
    for col in ['ndcg', 'NDCG', 'ndcg_score']:
        if col in df.columns:
            ndcg = df[col].mean()
            break
        
    # 3. MAP (Added 'map_norel' and 'map_rel' to the list)
    map_score = 0.0
    for col in ['map_norel', 'map_rel', 'ave_precision_norel', 'map', 'ap', 'ap_norel', 'MAP']:
        if col in df.columns:
            map_score = df[col].mean()
            break
            
    return f1, ndcg, map_score

print(f"\n{'DATASET':<10} | {'TOP-K':<5} | {'F-MEASURE':<10} | {'NDCG':<8} | {'MAP':<8}")
print("-" * 60)

if not os.path.exists(SCORE_DIR):
    print(f"CRITICAL ERROR: Directory not found: {SCORE_DIR}")
    # Fallback to current directory check
    SCORE_DIR = "." 

datasets = ['dbpedia', 'lmdb', 'faces']
results_data = []

for ds in datasets:
    for k in [5, 10]:
        df = None
        
        # Strategy A: Look for specific file (e.g., detailed_scores_dbpedia_top5.csv)
        filename = f"detailed_scores_{ds}_top{k}.csv"
        fpath = os.path.join(SCORE_DIR, filename)
        
        if os.path.exists(fpath):
            try:
                df = pd.read_csv(fpath)
            except: pass
        
        # Strategy B: Look for FACES combined file if Strategy A failed
        if df is None and ds == 'faces':
            alt_path = os.path.join(SCORE_DIR, FACES_COMBINED_FILE)
            if not os.path.exists(alt_path):
                 # Try looking in current dir
                 alt_path = FACES_COMBINED_FILE
            
            if os.path.exists(alt_path):
                try:
                    full_df = pd.read_csv(alt_path)
                    # Filter for K
                    top_col = 'Top' if 'Top' in full_df.columns else 'topk'
                    if top_col in full_df.columns:
                        df = full_df[full_df[top_col] == k]
                except: pass

        # Process Data
        if df is not None and not df.empty:
            f1, ndcg, map_s = get_metrics(df)
            print(f"{ds:<10} | {k:<5} | {f1:.4f}     | {ndcg:.4f}   | {map_s:.4f}")
            
            # Add to list for CSV
            results_data.append({
                "Dataset": ds,
                "Top-K": k,
                "F-Measure": round(f1, 4),
                "NDCG": round(ndcg, 4),
                "MAP": round(map_s, 4)
            })
        else:
             print(f"{ds:<10} | {k:<5} | -- MISSING --")

# --- SAVE TO CSV ---
if results_data:
    final_df = pd.DataFrame(results_data)
    final_df.to_csv(OUTPUT_FILENAME, index=False)
    print(f"\n✅ Results saved to: {OUTPUT_FILENAME}")