import os
import glob
import pandas as pd
import subprocess
import re
import sys

# --- 1. Define All Project Paths ---
JAR_FILE_DIR = "/content/Unsupervised_Learning_IRES_Model/ESBM_v1.2/Evaluation"
ELIS_FILE_PATH = "/content/Unsupervised_Learning_IRES_Model/ESBM_v1.2/Tool/ESBM-groundtruth/elist.txt"
BENCHMARK_DIR = "/content/Unsupervised_Learning_IRES_Model/ESBM_v1.2/Dataset/ESBM_benchmark_v1.2"
SUMMARY_DIR = "/content/Unsupervised_Learning_IRES_Model/ESBM_v1.2/Dataset/output_summaries"
IRES_RESULTS_DIR = "/content/Unsupervised_Learning_IRES_Model/ESBM_v1.2/output/results"


# Helper function to find scores in text
def parse_score(text, pattern):
    """Uses regex to find a score in text. Returns float or 'N/A'."""
    match = re.search(pattern, text)
    if match:
        try:
            return float(match.group(1))
        except:
            return "Error"
    return "N/A"

print("--- Step 1: Preparing list files for Java evaluator... ---")
try:
    # --- Create the list files using awk ---
    cmd_dbpedia = f"awk -F '\\t' '/dbpedia/ {{print $1}}' \"{ELIS_FILE_PATH}\" > \"{BENCHMARK_DIR}/dbpedia_data\""
    cmd_lmdb = f"awk -F '\\t' '/lmdb/ {{print $1}}' \"{ELIS_FILE_PATH}\" > \"{BENCHMARK_DIR}/lmdb_data\""
    
    os.system(cmd_dbpedia)
    os.system(cmd_lmdb)
    print("--- List files created successfully. ---")
except Exception as e:
    print(f"!!! CRITICAL ERROR: Could not create list files: {e} !!!")
    sys.exit()


print("--- Step 2: Running GATES (Baseline) evaluation... ---")
# --- Run the Java evaluator and capture its output ---
java_command = [
    "java", "-jar", 
    os.path.join(JAR_FILE_DIR, "esummeval_v1.2.jar"),
    BENCHMARK_DIR,
    SUMMARY_DIR
]

result = subprocess.run(java_command, capture_output=True, text=True)
java_output = result.stdout
if result.returncode != 0:
    print("!!! JAVA EVALUATION FAILED !!!")
    print(result.stderr)
else:
    print("--- GATES evaluation complete. Parsing scores... ---")

# --- Parse the GATES scores from the Java output ---
gates_scores = {
    'dbpedia_f5': parse_score(java_output, r"Results\(dbpedia@top5\):\s*F-measure=([\d\.]+)"),
    'dbpedia_ndcg5': parse_score(java_output, r"Results\(dbpedia@top5\).*NDCG=([\d\.]+)"),
    'dbpedia_f10': parse_score(java_output, r"Results\(dbpedia@top10\):\s*F-measure=([\d\.]+)"),
    'dbpedia_ndcg10': parse_score(java_output, r"Results\(dbpedia@top10\).*NDCG=([\d\.]+)"),
    
    'lmdb_f5': parse_score(java_output, r"Results\(lmdb@top5\):\s*F-measure=([\d\.]+)"),
    'lmdb_ndcg5': parse_score(java_output, r"Results\(lmdb@top5\).*NDCG=([\d\.]+)"),
    'lmdb_f10': parse_score(java_output, r"Results\(lmdb@top10\):\s*F-measure=([\d\.]+)"),
    'lmdb_ndcg10': parse_score(java_output, r"Results\(lmdb@top10\).*NDCG=([\d\.]+)"),
}


print("--- Step 3: Reading IRES (Your Model) results... ---")
# --- Find and read your IRES .csv result files ---
try:
    ires_dbpedia_file = glob.glob(os.path.join(IRES_RESULTS_DIR, '*dbpedia*.csv'))[0]
    ires_lmdb_file = glob.glob(os.path.join(IRES_RESULTS_DIR, '*lmdb*.csv'))[0]
    
    print(f"Found IRES DBpedia results: {os.path.basename(ires_dbpedia_file)}")
    print(f"Found IRES LMDB results: {os.path.basename(ires_lmdb_file)}")

    df_dbpedia = pd.read_csv(ires_dbpedia_file)
    df_lmdb = pd.read_csv(ires_lmdb_file)
    
    # Extract scores using the CORRECT column names and filtering by 'Top'
    ires_scores = {
        'dbpedia_f5':   df_dbpedia[df_dbpedia['Top'] == 5]['fmeasure_rel'].iloc[-1],
        'dbpedia_map5': df_dbpedia[df_dbpedia['Top'] == 5]['ap'].iloc[-1],
        'dbpedia_f10':  df_dbpedia[df_dbpedia['Top'] == 10]['fmeasure_rel'].iloc[-1],
        'dbpedia_map10':df_dbpedia[df_dbpedia['Top'] == 10]['ap'].iloc[-1],
        
        'lmdb_f5':   df_lmdb[df_lmdb['Top'] == 5]['fmeasure_rel'].iloc[-1],
        'lmdb_map5': df_lmdb[df_lmdb['Top'] == 5]['ap'].iloc[-1],
        'lmdb_f10':  df_lmdb[df_lmdb['Top'] == 10]['fmeasure_rel'].iloc[-1],
        'lmdb_map10':df_lmdb[df_lmdb['Top'] == 10]['ap'].iloc[-1],
    }
    print("--- IRES scores loaded successfully. ---")
except Exception as e:
    print(f"!!! CRITICAL ERROR: Could not read IRES .csv files: {e} !!!")
    print(f"Searched in: {IRES_RESULTS_DIR}")
    print("Please check the path and column names.")
    sys.exit()


# --- 4. Print the final, dynamic tables ---
print("\n\n" + "="*60)
print("     DYNAMICALLY GENERATED QUANTITATIVE RESULTS (Task 1)")
print("="*60)

print("\n## DBpedia Results")
print(f"""
| Model | F-Measure@5 | MAP@5 | NDCG@5 | F-Measure@10 | MAP@10 | NDCG@10 | Runtime |
| :--- | :--- | :--- | :--- | :--- | :--- | :--- | :--- |
| **GATES** | {gates_scores['dbpedia_f5']:.4f} | N/A* | {gates_scores['dbpedia_ndcg5']:.4f} | {gates_scores['dbpedia_f10']:.4f} | N/A* | {gates_scores['dbpedia_ndcg10']:.4f} | N/A |
| **IRES** | {ires_scores['dbpedia_f5']:.4f} | {ires_scores['dbpedia_map5']:.4f} | N/A** | {ires_scores['dbpedia_f10']:.4f} | {ires_scores['dbpedia_map10']:.4f} | N/A** | (See Log) |
""")

print("\n## LMDB Results")
print(f"""
| Model | F-Measure@5 | MAP@5 | NDCG@5 | F-Measure@10 | MAP@10 | NDCG@10 | Runtime |
| :--- | :--- | :--- | :--- | :--- | :--- | :--- | :--- |
| **GATES** | {gates_scores['lmdb_f5']:.4f} | N/A* | {gates_scores['lmdb_ndcg5']:.4f} | {gates_scores['lmdb_f10']:.4f} | N/A* | {gates_scores['lmdb_ndcg10']:.4f} | N/A |
| **IRES** | {ires_scores['lmdb_f5']:.4f} | {ires_scores['lmdb_map5']:.4f} | N/A** | {ires_scores['lmdb_f10']:.4f} | {ires_scores['lmdb_map10']:.4f} | N/A** | (See Log) |
""")

print("\n---")
print("*N/A: The esummeval_v1.2.jar tool did not report MAP scores in its output.")
print("**N/A: The IRES model does not generate ranked summary files, so NDCG cannot be calculated.")
print("\n---")
print("Runtimes for IRES must be manually read from the training log.")
