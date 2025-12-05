import sys
import os
import pandas as pd
import torch
import numpy as np
import networkx as nx
import importlib.util

# ==========================================
# CONFIGURATION
# ==========================================
DATASET_NAME = "dbpedia"  # Change to 'lmdb' or 'faces' as needed
ENTITIES_TO_INSPECT = [1, 5, 10]
TOP_K = 5

# Define Paths relative to where you run this script
CURRENT_DIR = os.getcwd()
IRES_FOLDER = "Unsupervised_Model_IRES"
ESLM_FOLDER = "Supervised_Model_ESLM"

IRES_ROOT = os.path.abspath(IRES_FOLDER)
ESLM_ROOT = os.path.abspath(ESLM_FOLDER)

# Output File for the Qualitative Analysis Text
OUTPUT_TXT = os.path.join(CURRENT_DIR, f"Qualitative_Analysis_{DATASET_NAME}.txt")

# Specific File Paths
IRES_MODEL_PATH = os.path.join(IRES_ROOT, "/content/KGSUMM/Task2/Unsupervised_Model_IRES/output/models/dbpedia/esbm/IRES_model_rdf.pth")
ESLM_PRED_FILE = os.path.join(ESLM_ROOT, "/content/KGSUMM/Task2/Supervised_Model_ESLM/Predictions/predictions_ESLM_t5_dbpedia_top5_10.csv")

# ==========================================
# MAIN LOGIC
# ==========================================

def run_comparison():
    # Open the log file for writing
    log_file = open(OUTPUT_TXT, "w", encoding="utf-8")

    # Helper function to print to BOTH console and file
    def log(msg=""):
        print(msg)
        log_file.write(str(msg) + "\n")

    log(f"{'='*60}")
    log(f"QUALITATIVE ANALYSIS: {DATASET_NAME.upper()} (Top-{TOP_K})")
    log(f"{'='*60}\n")

    # Initialize variables
    original_cwd = os.getcwd()
    ires_main = None
    utils = None
    config = None
    
    # --- 1. Load IRES Data & Model ---
    log(f">> Loading Unsupervised (IRES) Model from: {IRES_ROOT}")
    
    if not os.path.exists(os.path.join(IRES_ROOT, "config.py")):
        log(f"CRITICAL ERROR: config.py not found in {IRES_ROOT}")
        return

    if IRES_ROOT not in sys.path:
        sys.path.insert(0, IRES_ROOT)
    
    try:
        os.chdir(IRES_ROOT)
        import config
        import utils
        import main as ires_main
        
        conf = config.Config()
        conf.args.dataset = DATASET_NAME
        conf.args.benchmark = "esbm" 
        # FIX: Force Encoder to GCN to match saved model
        conf.args.encoder = "GCN"
        conf.args.data_path = os.path.join(IRES_ROOT, "Tool")
        
        log("   - Loading Graph Data...")
        entity_dataset, G, nodes, entity_node_id, relation_id, edge_relation_id, adj, transe_save, features_transe = ires_main.load_graph()
        pg_data, features = ires_main.setup_features(G, relation_id, edge_relation_id, adj, transe_save, features_transe)
        
    except Exception as e:
        log(f"CRITICAL ERROR LOADING IRES MODULES: {e}")
        return
    finally:
        os.chdir(original_cwd)
        if IRES_ROOT in sys.path:
            sys.path.remove(IRES_ROOT)

    if not os.path.exists(IRES_MODEL_PATH):
        log(f"ERROR: IRES model file not found at {IRES_MODEL_PATH}")
        return

    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    try:
        ires_model = torch.load(IRES_MODEL_PATH, map_location=device, weights_only=False)
        ires_model.eval()
    except Exception as e:
        log(f"Error loading .pth file: {e}")
        return

    log(">> Generating IRES Scores...")
    with torch.no_grad():
        x = features.to(device)
        edge_index = pg_data.edge_index.to(device)
        
        if conf.args.encoder == 'RGCN': 
             if isinstance(pg_data.edge_attr, list):
                 edge_attr = torch.tensor(pg_data.edge_attr).to(device)
             else:
                 edge_attr = pg_data.edge_attr.to(device)
             z = ires_model.encode(x, edge_index, edge_attr)
        else:
             z = ires_model.encode(x, edge_index)
             
        reconstructed_adj = torch.mm(z, z.t())

    # --- 2. Load ESLM Predictions ---
    log(">> Loading Supervised (ESLM) Predictions...")
    if not os.path.exists(ESLM_PRED_FILE):
        log(f"ERROR: ESLM predictions not found at {ESLM_PRED_FILE}")
        return
    df_eslm = pd.read_csv(ESLM_PRED_FILE)

    # --- 3. Compare Entities ---
    for eid in ENTITIES_TO_INSPECT:
        log(f"\n\n{'='*20} ENTITY ID: {eid} {'='*20}")
        
        # A. Get Gold Standard
        gold_facts = set()
        os.chdir(IRES_ROOT)
        try:
            target_row = entity_dataset.iloc[eid]
            target_uri = f'<{target_row["euri"]}>'
            log(f"URI: {target_uri}")

            for i in range(6):
                tval = utils.import_top_summary(conf.data_path(), eid, i, TOP_K, target_uri)
                if tval:
                    for t in tval:
                        gold_facts.add(f"{t[1]} -> {t[2]}")
        except Exception as e:
            log(f"Error fetching gold standard: {e}")
        finally:
            os.chdir(original_cwd)

        log(f"\n[GOLD STANDARD] (Union of human summaries)")
        for g in list(gold_facts)[:5]: 
            log(f"  * {g}")
        if len(gold_facts) > 5: log("  ... (and more)")

        # B. Get IRES Prediction
        log(f"\n[UNSUPERVISED IRES]")
        node_id = entity_node_id.get(target_uri)
        if node_id is not None:
            out_edges = list(G.out_edges(target_uri, data=True))
            ranked_edges = []
            for edge in out_edges:
                other_node = edge[1]
                other_id = entity_node_id[other_node]
                score = reconstructed_adj[node_id][other_id].item()
                rel = edge[2]['relation'] if 'relation' in edge[2] else "unknown"
                ranked_edges.append((score, f"{rel} -> {other_node}"))
            
            ranked_edges.sort(key=lambda x: x[0], reverse=True)
            for i, (score, text) in enumerate(ranked_edges[:TOP_K]):
                match = "✅" if text in gold_facts else " "
                log(f"  {match} {text}  (Score: {score:.4f})")
        else:
            log("  (Entity not found in IRES graph)")

        # C. Get ESLM Prediction
        log(f"\n[SUPERVISED ESLM]")
        row = df_eslm[(df_eslm['entity_id'] == eid) & (df_eslm['topk'] == TOP_K)]
        if not row.empty:
            pred_str = str(row.iloc[0]['predicted_indices'])
            if pred_str.lower() == 'nan':
                pred_indices = []
            else:
                clean_str = pred_str.replace('[', '').replace(']', '')
                if clean_str.strip():
                    pred_indices = list(map(int, clean_str.split(',')))
                else:
                    pred_indices = []
            
            literal_file = os.path.join(ESLM_ROOT, f"classes/data_inputs/literals/{DATASET_NAME}/{eid}_literal.txt")
            if os.path.exists(literal_file):
                with open(literal_file, 'r') as f:
                    all_literals = [l.strip().split('\t') for l in f.readlines()]
                
                for idx in pred_indices:
                    if idx < len(all_literals):
                        lit_parts = all_literals[idx]
                        if len(lit_parts) >= 3:
                            text = f"{lit_parts[1]} -> {lit_parts[2]}"
                        else:
                            text = str(lit_parts)
                        
                        match = "✅" if text in gold_facts else " "
                        log(f"  {match} {text}")
            else:
                log("  (Literal file not found to decode indices)")
        else:
            log("  (No prediction found in CSV)")

    # Close file
    log_file.close()
    print(f"\n✅ Results saved to: {OUTPUT_TXT}")

if __name__ == "__main__":
    run_comparison()