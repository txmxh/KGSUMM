import os
import json
import glob
from rdflib import Graph

# --- Input Paths (where the .nt files are) ---
DBPEDIA_DATA_DIR = "/content/Unsupervised_Learning_IRES_Model/ESBM_v1.2/Dataset/ESBM_benchmark_v1.2/dbpedia_data"
LMDB_DATA_DIR = "/content/Unsupervised_Learning_IRES_Model/ESBM_v1.2/Dataset/ESBM_benchmark_v1.2/lmdb_data"

# --- Output Path (where IRES will read from) ---
IRES_DATA_DIR = "ESBM_v1.2/Temp"

# --- FIX: Create the output directory if it doesn't exist ---
os.makedirs(IRES_DATA_DIR, exist_ok=True)
# -------------------------------------------------------------

print("Starting ESBM v1.2 conversion...")

# --- 1. Process DBpedia ---
dbpedia_triples = []
dbpedia_gt = {}
print("Processing DBpedia files...")

# Find all entity description files
desc_files_dbpedia = glob.glob(os.path.join(DBPEDIA_DATA_DIR, "*", "*_desc.nt"))
for f in desc_files_dbpedia:
    g = Graph()
    try:
        g.parse(f, format="nt")
        for s, p, o in g:
            dbpedia_triples.append([str(s), str(p), str(o)])
    except Exception as e:
        print(f"Error parsing {f}: {e}")

# Find all ground truth files
gt_files_dbpedia = glob.glob(os.path.join(DBPEDIA_DATA_DIR, "*", "*_gold_top*.nt"))
for f_path in gt_files_dbpedia:
    entity_id = os.path.basename(os.path.dirname(f_path))
    if entity_id not in dbpedia_gt:
        dbpedia_gt[entity_id] = []

    g_gt = Graph()
    try:
        g_gt.parse(f_path, format="nt")
        for s, p, o in g_gt:
            triple_str = [str(s), str(p), str(o)]
            if triple_str not in dbpedia_gt[entity_id]:
                dbpedia_gt[entity_id].append(triple_str)
    except Exception as e:
        print(f"Error parsing {f_path}: {e}")

# --- 2. Process LMDB ---
lmdb_triples = []
lmdb_gt = {}
print("Processing LMDB files...")

# Find all entity description files
desc_files_lmdb = glob.glob(os.path.join(LMDB_DATA_DIR, "*", "*_desc.nt"))
for f in desc_files_lmdb:
    g = Graph()
    try:
        g.parse(f, format="nt")
        for s, p, o in g:
            lmdb_triples.append([str(s), str(p), str(o)])
    except Exception as e:
        print(f"Error parsing {f}: {e}")

# Find all ground truth files
gt_files_lmdb = glob.glob(os.path.join(LMDB_DATA_DIR, "*", "*_gold_top*.nt"))
for f_path in gt_files_lmdb:
    entity_id = os.path.basename(os.path.dirname(f_path))
    if entity_id not in lmdb_gt:
        lmdb_gt[entity_id] = []

    g_gt = Graph()
    try:
        g_gt.parse(f_path, format="nt")
        for s, p, o in g_gt:
            triple_str = [str(s), str(p), str(o)]
            if triple_str not in lmdb_gt[entity_id]:
                lmdb_gt[entity_id].append(triple_str)
    except Exception as e:
        print(f"Error parsing {f_path}: {e}")

# --- 3. Save all 4 JSON files ---
print("Saving JSON files...")
with open(os.path.join(IRES_DATA_DIR, "dbpedia.json"), 'w') as f:
    json.dump(dbpedia_triples, f)
with open(os.path.join(IRES_DATA_DIR, "dbpedia_groundtruth.json"), 'w') as f:
    json.dump(dbpedia_gt, f)
with open(os.path.join(IRES_DATA_DIR, "lmdb.json"), 'w') as f:
    json.dump(lmdb_triples, f)
with open(os.path.join(IRES_DATA_DIR, "lmdb_groundtruth.json"), 'w') as f:
    json.dump(lmdb_gt, f)

print(f"Conversion complete. All files saved to {IRES_DATA_DIR}")
