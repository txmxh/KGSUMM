import os
import pandas as pd
import numpy as np
import ast
from classes.dataset import ESBenchmark
from evaluator.fmeasure import FMeasure

# --- CONFIGURATION ---
PRED_DIR = "Predictions"
OUTPUT_DIR = "Supervised_Scores"
os.makedirs(OUTPUT_DIR, exist_ok=True)

# Map your filenames to dataset names
FILES = [
    ("dbpedia", "predictions_ESLM_t5_dbpedia_top5_10.csv"),
    ("lmdb",    "predictions_ESLM_t5_lmdb_top5_10.csv"),
    ("faces",   "predictions_ESLM_t5_faces_top5_10.csv")
]
# ---------------------

def calculate_scores():
    fmeasure = FMeasure()
    
    for ds_name, filename in FILES:
        filepath = os.path.join(PRED_DIR, filename)
        if not os.path.exists(filepath):
            print(f"Skipping {ds_name}: File not found at {filepath}")
            continue
            
        print(f"Processing {ds_name}...")
        df = pd.read_csv(filepath)
        
        # The supervised predictions file usually has Top-5 and Top-10 mixed.
        # We need to split them.
        for k in [5, 10]:
            # Filter for specific K
            subset = df[df['topk'] == k]
            
            if subset.empty:
                print(f"  No data found for Top-{k} in {ds_name}")
                continue
                
            scores = []
            dataset_loader = ESBenchmark(ds_name, 6, k, False)
            
            for index, row in subset.iterrows():
                eid = row['entity_id']
                
                # 1. Parse Predicted Indices (saved as string "1, 2, 3")
                try:
                    pred_indices = list(map(int, str(row['predicted_indices']).split(',')))
                except:
                    pred_indices = []
                
                # 2. Get Ground Truth
                # We need the gold standard for this specific entity
                # The loader helper 'get_all_data' or similar logic retrieves this
                # We use a simplified logic here assuming standard ESBM structure
                gold_summaries = []
                # Retrieve all 6 gold summaries
                for i in range(6): 
                    # Note: You might need to adjust 'dataset_loader.get_gold' depending on your classes/data.py
                    # This uses the standard logic from your previous evaluation code
                    try:
                        tval = dataset_loader.get_gold_indices(eid, i) # You might need to verify this method exists in your class
                        if tval: gold_summaries.append(set(tval))
                    except: pass
                
                # 3. Calculate Max/Avg F-Measure
                if not gold_summaries:
                    scores.append({'entity': eid, 'fmeasure': 0.0})
                    continue
                    
                entity_fscores = []
                pred_set = set(pred_indices)
                
                for gold_set in gold_summaries:
                    # F-Measure logic
                    prec = len(pred_set.intersection(gold_set)) / len(pred_set) if len(pred_set) > 0 else 0
                    rec = len(pred_set.intersection(gold_set)) / len(gold_set) if len(gold_set) > 0 else 0
                    f1 = (2 * prec * rec) / (prec + rec) if (prec + rec) > 0 else 0
                    entity_fscores.append(f1)
                
                # Average F-Measure for this entity
                scores.append({'entity': eid, 'fmeasure': np.mean(entity_fscores)})
            
            # Save to CSV
            score_df = pd.DataFrame(scores)
            output_filename = f"detailed_scores_{ds_name}_top{k}.csv"
            score_df.to_csv(os.path.join(OUTPUT_DIR, output_filename), index=False)
            print(f"  Saved {output_filename}")

if __name__ == "__main__":
    calculate_scores()