import sys
import os
import pandas as pd
import numpy as np

# ==========================================
# CONFIGURATION
# ==========================================
DATASET_NAME = "dbpedia"  # Change to 'lmdb' or 'faces'
TOP_K = 5

# Define Paths
CURRENT_DIR = os.getcwd()
IRES_FOLDER = "Unsupervised_Model_IRES"
ESLM_FOLDER = "Supervised_Model_ESLM"

# --- FIX: Point directly to the folder, don't add ESBM_v1.2 ---
ESLM_ROOT = os.path.abspath(ESLM_FOLDER) 
# --------------------------------------------------------------

# Path to Score Files
UNSUP_SCORES_DIR = os.path.join(IRES_FOLDER, "Unsupervised_Scores")
SUP_SCORES_DIR   = os.path.join(ESLM_FOLDER, "Supervised_Scores")

# Path to Mapping File (Absolute path is safest)
ELIST_PATH = "/content/KGSUMM/Task2/Unsupervised_Model_IRES/Tool/ESBM-groundtruth/elist.txt"

# ==========================================
# HELPER FUNCTIONS
# ==========================================
def load_elist_mapping(elist_path):
    # Check if absolute path exists
    if not os.path.exists(elist_path): 
        # Fallback: Try relative path if absolute fails
        rel_path = os.path.join(IRES_FOLDER, "/content/KGSUMM/Task2/Unsupervised_Model_IRES/Tool/ESBM-groundtruth/elist.txt")
        if os.path.exists(rel_path):
            elist_path = rel_path
        else:
            print(f"WARNING: elist.txt not found at {elist_path}")
            return {}
            
    mapping = {}
    with open(elist_path, 'r', encoding='utf-8') as f:
        next(f) # Skip header
        for line in f:
            parts = line.strip().split('\t')
            if len(parts) >= 4:
                eid_int = parts[0]
                euri_str = parts[3]
                mapping[str(eid_int)] = euri_str
    return mapping

def get_literal_text(eid, dataset_name, eslm_root):
    """Reads the literal file to get the raw text candidates"""
    # Try multiple possible paths for the 'classes' folder to be safe
    possible_paths = [
        os.path.join(eslm_root, "ESBM_v1.2", "classes", "data_inputs", "literals", dataset_name, f"{eid}_literal.txt"), # Deep nesting
        os.path.join(eslm_root, "classes", "data_inputs", "literals", dataset_name, f"{eid}_literal.txt")               # Direct nesting
    ]
    
    for literal_file in possible_paths:
        if os.path.exists(literal_file):
            with open(literal_file, 'r', encoding='utf-8') as f:
                return [l.strip() for l in f.readlines()]
    
    return []

def run_error_analysis():
    print(f"\n{'='*60}")
    print(f"ERROR ANALYSIS: {DATASET_NAME.upper()} (Top-{TOP_K})")
    print(f"{'='*60}\n")

    # 1. Load Score Files
    ires_file = f"detailed_scores_{DATASET_NAME}_top{TOP_K}.csv"
    
    # Handle special FACES filename
    if DATASET_NAME == 'faces' and TOP_K == 5:
         alt_path = os.path.join(UNSUP_SCORES_DIR, "detailed_IRES_FACES_metrics_R0 (1).csv")
         if os.path.exists(alt_path): ires_path = alt_path
         else: ires_path = os.path.join(UNSUP_SCORES_DIR, ires_file)
    else:
         ires_path = os.path.join(UNSUP_SCORES_DIR, ires_file)

    eslm_path = os.path.join(SUP_SCORES_DIR, ires_file)

    if not os.path.exists(ires_path) or not os.path.exists(eslm_path):
        print(f"CRITICAL: Score files not found.")
        print(f"IRES: {ires_path}")
        print(f"ESLM: {eslm_path}")
        return

    df_ires = pd.read_csv(ires_path)
    df_eslm = pd.read_csv(eslm_path)

    # 2. Prepare Merge Keys
    ires_col = 'euri' if 'euri' in df_ires.columns else 'entity'
    eslm_col = 'entity' if 'entity' in df_eslm.columns else 'euri'

    df_ires['clean_key'] = df_ires[ires_col].astype(str).apply(lambda x: x.strip('<>').strip())
    
    if DATASET_NAME == 'faces':
        df_eslm['clean_key'] = df_eslm[eslm_col].astype(str)
    else:
        id_map = load_elist_mapping(ELIST_PATH)
        print(f"Loaded {len(id_map)} ID mappings.")
        df_eslm['clean_key'] = df_eslm[eslm_col].apply(
            lambda x: id_map.get(str(x), str(x)).strip('<>').strip()
        )

    # 3. Merge
    merged = pd.merge(df_ires, df_eslm, on='clean_key', suffixes=('_ires', '_eslm'))
    
    print(f"Successfully merged {len(merged)} entities.")
    
    if len(merged) == 0:
        print("ERROR: Merge failed.")
        return

    # Identify Score Columns
    col_ires = 'fmeasure_norel' if 'fmeasure_norel' in merged.columns else 'fmeasure'
    col_eslm = 'fmeasure' if 'fmeasure' in merged.columns else 'fmeasure_eslm'
    
    if col_eslm in merged.columns and col_eslm + "_eslm" in merged.columns:
        col_eslm = col_eslm + "_eslm"
    elif col_eslm not in merged.columns and "fmeasure" in merged.columns:
        if "fmeasure_eslm" in merged.columns: col_eslm = "fmeasure_eslm"
        else: col_eslm = "fmeasure"

    # 4. Find Interesting Cases
    LOW = 0.1
    HIGH = 0.4
    
    hard_cases = merged[(merged[col_ires] <= LOW) & (merged[col_eslm] <= LOW)]
    eslm_fails = merged[(merged[col_eslm] <= LOW) & (merged[col_ires] >= HIGH)]
    ires_fails = merged[(merged[col_ires] <= LOW) & (merged[col_eslm] >= HIGH)]

    print(f"\n--- STATISTICS ---")
    print(f"Total Entities: {len(merged)}")
    print(f"Hard Cases (Both < {LOW}): {len(hard_cases)}")
    print(f"ESLM Fail / IRES Good:    {len(eslm_fails)}")
    print(f"IRES Fail / ESLM Good:    {len(ires_fails)}")

    # 5. Inspect a Hard Case
    if not hard_cases.empty:
        sample = hard_cases.iloc[0]
        uri = sample['clean_key']
        print(f"\n\n{'='*20} INSPECTING HARD FAILURE {'='*20}")
        print(f"Entity: {uri}")
        print(f"IRES Score: {sample[col_ires]:.4f}")
        print(f"ESLM Score: {sample[col_eslm]:.4f}")
        
        original_id = sample[eslm_col + "_eslm"] if eslm_col + "_eslm" in sample else sample[eslm_col]
        
        # Pass the fixed root
        candidates = get_literal_text(original_id, DATASET_NAME, ESLM_ROOT)
        print(f"\n[CANDIDATE FACTS] (Total: {len(candidates)})")
        
        if len(candidates) == 0:
            print("  (Text file not found or empty. This explains the 0 score!)")
        elif len(candidates) < TOP_K:
            print(f"  -> SPARSE DATA WARNING: Only {len(candidates)} candidates available.")
            for c in candidates: print(f"  * {c}")
        else:
            print("  -> First 5 candidates:")
            for c in candidates[:5]: print(f"  * {c}")
            
    # 6. Inspect an ESLM Failure
    if not eslm_fails.empty:
        sample = eslm_fails.iloc[0]
        uri = sample['clean_key']
        print(f"\n\n{'='*20} INSPECTING ESLM FAILURE {'='*20}")
        print(f"Entity: {uri}")
        print(f"IRES Score: {sample[col_ires]:.4f} (Good)")
        print(f"ESLM Score: {sample[col_eslm]:.4f} (Bad)")
        
        original_id = sample[eslm_col + "_eslm"] if eslm_col + "_eslm" in sample else sample[eslm_col]
        candidates = get_literal_text(original_id, DATASET_NAME, ESLM_ROOT)
        
        if candidates:
            print(f"\n[CANDIDATE FACTS] (First 5 of {len(candidates)})")
            for c in candidates[:5]: print(f"  * {c}")
        else:
            print("  (Text file missing)")

if __name__ == "__main__":
    run_error_analysis()
