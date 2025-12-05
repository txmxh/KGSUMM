import pandas as pd
from scipy import stats
import os
import re

# --- CONFIGURATION ---
UNSUP_DIR = os.path.join("Unsupervised_Model_IRES", "Unsupervised_Scores")
SUP_DIR   = os.path.join("Supervised_Model_ESLM", "Supervised_Scores")

ELIST_PATH = "/content/KGSUMM/Task2/Unsupervised_Model_IRES/Tool/ESBM-groundtruth/elist.txt"
FACES_DATA_DIR = "/content/KGSUMM/Task2/Supervised_Model_ESLM/datasets/FACES/faces_data" 

datasets = ["dbpedia", "lmdb", "faces"]
top_ks = [5, 10]
# ---------------------

def load_elist_mapping(elist_path):
    if not os.path.exists(elist_path): return {}
    mapping = {}
    with open(elist_path, 'r', encoding='utf-8') as f:
        header = next(f)
        for line in f:
            parts = line.strip().split('\t')
            if len(parts) >= 4:
                mapping[str(parts[0])] = parts[3]
    return mapping

def load_faces_mapping(faces_dir):
    mapping = {}
    if not os.path.exists(faces_dir): return {}
    entity_folders = [f for f in os.listdir(faces_dir) if os.path.isdir(os.path.join(faces_dir, f))]
    for eid in entity_folders:
        try:
            with open(os.path.join(faces_dir, eid, f"{eid}_desc.nt"), 'r', encoding='utf-8') as f:
                match = re.match(r'<([^>]+)>', f.readline())
                if match: mapping[str(eid)] = match.group(1)
        except: pass
    return mapping

# Load Mappings
print("Loading ID mappings...")
db_lmdb_map = load_elist_mapping(ELIST_PATH)
faces_map = load_faces_mapping(FACES_DATA_DIR)

# Prepare list to store data for CSV
results_data = []

print(f"\n{'DATASET':<10} | {'TOP-K':<5} | {'IRES (Mean)':<12} | {'ESLM (Mean)':<12} | {'P-VALUE':<10} | {'RESULT'}")
print("-" * 85)

# Pre-load special FACES file if exists
faces_ires_df = None
faces_ires_path = "detailed_scores_faces_top5_10.csv"
if not os.path.exists(faces_ires_path): faces_ires_path = os.path.join(UNSUP_DIR, "detailed_scores_faces_top5_10.csv")
if os.path.exists(faces_ires_path):
    faces_ires_df = pd.read_csv(faces_ires_path)
    if 'Top' in faces_ires_df.columns: faces_ires_df.rename(columns={'Top': 'topk'}, inplace=True)

for ds in datasets:
    for k in top_ks:
        # 1. Load IRES
        df_ires = None
        if ds == 'faces' and faces_ires_df is not None:
            df_ires = faces_ires_df[faces_ires_df['topk'] == k].copy()
        else:
            ires_path = os.path.join(UNSUP_DIR, f"detailed_scores_{ds}_top{k}.csv")
            if os.path.exists(ires_path): df_ires = pd.read_csv(ires_path)

        # 2. Load ESLM
        eslm_path = os.path.join(SUP_DIR, f"detailed_scores_{ds}_top{k}.csv")
        if df_ires is None or df_ires.empty or not os.path.exists(eslm_path):
            continue
            
        df_eslm = pd.read_csv(eslm_path)

        # 3. Merge
        ires_col = 'euri' if 'euri' in df_ires.columns else 'entity'
        eslm_col = 'entity' if 'entity' in df_eslm.columns else 'euri'
        
        df_ires['clean_key'] = df_ires[ires_col].astype(str).apply(lambda x: x.strip('<>').strip())
        mapping = faces_map if ds == 'faces' else db_lmdb_map
        df_eslm['clean_key'] = df_eslm[eslm_col].apply(lambda x: mapping.get(str(x), str(x)).strip('<>').strip())

        merged = pd.merge(df_ires, df_eslm, on='clean_key', suffixes=('_ires', '_eslm'))
        if len(merged) == 0: continue

        # 4. Stats
        col_ires = 'fmeasure_norel' if 'fmeasure_norel' in merged.columns else 'fmeasure'
        col_eslm = 'fmeasure' if 'fmeasure' in merged.columns else 'fmeasure_eslm'
        
        try:
            stat, p_val = stats.wilcoxon(merged[col_ires], merged[col_eslm])
        except ValueError: p_val = 1.0 
        
        res_str = "SIGNIFICANT" if p_val < 0.05 else "Not Sig."
        
        # Print
        print(f"{ds:<10} | {k:<5} | {merged[col_ires].mean():.4f}       | {merged[col_eslm].mean():.4f}       | {p_val:.1e}  | {res_str}")
        
        # 5. Save Data for CSV
        results_data.append({
            'Dataset': ds,
            'Top-K': k,
            'IRES_Mean_FMeasure': round(merged[col_ires].mean(), 4),
            'ESLM_Mean_FMeasure': round(merged[col_eslm].mean(), 4),
            'P-Value': p_val,
            'Significant': res_str
        })

# --- SAVE TO FILE ---
if results_data:
    final_df = pd.DataFrame(results_data)
    final_df.to_csv("final_results_table.csv", index=False)
    print("\n✅ Table saved to: final_results_table.csv")
    print("(Download this file to use in your report!)")