import os
import json
import pandas as pd
from wikes_toolkit import WikESToolkit, WikESVersions, PandasWikESGraph

# --- HELPER: Auto-detect column names ---
def get_column_map(df):
    """Finds the correct column names for subject/predicate/object."""
    cols = df.columns.tolist()
    mapping = {}
    
    # Subject/Head
    if 'subject' in cols: mapping['s'] = 'subject'
    elif 'head' in cols: mapping['s'] = 'head'
    elif 'source' in cols: mapping['s'] = 'source'
    else: mapping['s'] = cols[0] # Fallback
    
    # Predicate/Relation
    if 'predicate' in cols: mapping['p'] = 'predicate'
    elif 'relation' in cols: mapping['p'] = 'relation'
    elif 'label' in cols: mapping['p'] = 'label'
    else: mapping['p'] = cols[1] # Fallback
    
    # Object/Tail
    if 'object' in cols: mapping['o'] = 'object'
    elif 'tail' in cols: mapping['o'] = 'tail'
    elif 'target' in cols: mapping['o'] = 'target'
    else: mapping['o'] = cols[2] # Fallback
    
    return mapping

# 1. Setup paths
output_dir = "./data/wikies_small"
if not os.path.exists(output_dir):
    os.makedirs(output_dir)

print("--- Starting WikiES-SMALL (Cinema) Download ---")

# 2. Initialize Toolkit
toolkit = WikESToolkit(save_path="./wikes_cache") 

# 3. Load the "WikiCinema" Small dataset 
print("Loading graph (this might take a moment)...")
dataset_version = WikESVersions.V1.WikiCinema.SMALL
graph = toolkit.load_graph(PandasWikESGraph, dataset_version)

# 4. Convert to IRES/ESBM JSON Format
print("Converting data to IRES format...")

data_json = []
gold_json = []

# Get root entities
root_ids = graph.root_entity_ids()
print(f"Found {len(root_ids)} root entities to process.")

# Fetch ALL triples once (efficient)
all_triples_df = graph.triples()
print(f"Loaded {len(all_triples_df)} total triples.")

# Detect column names for triples
t_map = get_column_map(all_triples_df)
print(f"DEBUG: Using triple columns: {t_map}")

for i, entity_id in enumerate(root_ids):
    if i % 50 == 0:
        print(f"Processing entity {i}/{len(root_ids)}...")

    # --- 4a. Build 'data.json' (The Neighborhood) ---
    # Filter global triples for this entity
    # We look for rows where the entity is the Subject OR Object
    entity_mask = (all_triples_df[t_map['s']] == entity_id) | (all_triples_df[t_map['o']] == entity_id)
    entity_specific_triples = all_triples_df[entity_mask]
    
    formatted_triples = []
    for _, row in entity_specific_triples.iterrows():
        formatted_triples.append([
            str(row[t_map['s']]), 
            str(row[t_map['p']]), 
            str(row[t_map['o']])
        ])
        
    data_json.append({
        "entity": str(entity_id),
        "triples": formatted_triples
    })

    # --- 4b. Build 'gold.json' (The Summary) ---
    # FIX: Use 'ground_truths(entity_id)' method found in dir()
    try:
        summary_df = graph.ground_truths(entity_id)
        
        # Detect columns for summary (might differ from triples)
        if i == 0: # Do this check only once
            s_map = get_column_map(summary_df)
            print(f"DEBUG: Using summary columns: {s_map}")

        formatted_summary = []
        for _, row in summary_df.iterrows():
            formatted_summary.append([
                str(row[s_map['s']]), 
                str(row[s_map['p']]), 
                str(row[s_map['o']])
            ])
            
        gold_json.append({
            "entity": str(entity_id),
            "summary": formatted_summary
        })
        
    except Exception as e:
        print(f"WARNING: Could not fetch summary for {entity_id}: {e}")
        continue

# 5. Save files
print(f"Saving to {output_dir}...")

with open(os.path.join(output_dir, "data.json"), "w") as f:
    json.dump(data_json, f, indent=4)

with open(os.path.join(output_dir, "gold.json"), "w") as f:
    json.dump(gold_json, f, indent=4)

print("--- Download and Conversion Complete! ---")
print(f"Files ready at: {output_dir}/data.json and {output_dir}/gold.json")