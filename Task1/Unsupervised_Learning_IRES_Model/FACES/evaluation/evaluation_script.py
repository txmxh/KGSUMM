import os
import time
import numpy as np
from sklearn.metrics import f1_score
from scipy.stats import wilcoxon, ttest_rel
import glob
import re
import pandas as pd
from collections import defaultdict

# ================================
# CONFIGURATION
# ================================
GOLD_SUMMARIES_ROOT = "/content/KGSUMM/Task1/Unsupervised_Learning_IRES_Model/FACES/Dataset/output_summaries_ensembled/faces"
IRES_OUTPUT_ROOT = "/content/KGSUMM/Task1/Unsupervised_Learning_IRES_Model/FACES/Dataset/output_summaries/faces"
BASELINE_OUTPUT_ROOT = "/content/KGSUMM/Task1/Unsupervised_Learning_IRES_Model/FACES/Dataset/output_summaries/faces_baseline"

OUTPUT_RESULT_FILE = "/content/KGSUMM/Task1/evaluation_results.txt"
OUTPUT_CSV_FILE = "/content/KGSUMM/Task1/evaluation_per_entity.csv"

TOP_K_SIZES = [5, 10]
EPSILON = 1e-5  # Tiny perturbation if needed

# ================================
# HELPER FUNCTIONS
# ================================
def load_output_summaries(root_folder, k):
    summaries = []
    search_path = os.path.join(root_folder, '*', f'*_top{k}.nt')
    file_paths = sorted(glob.glob(search_path))

    def extract_entity_id(path):
        match = re.search(r'/(\d+)/', path)
        return int(match.group(1)) if match else 0

    file_paths.sort(key=extract_entity_id)

    for file_path in file_paths:
        try:
            with open(file_path, "r", encoding="utf-8") as f:
                entities = set(line.strip() for line in f if line.strip())
                summaries.append(entities)
        except Exception as e:
            print(f"Error reading {file_path}: {e}")
    return summaries

def compute_average_precision(y_true, y_pred):
    score_sum = 0.0
    hits = 0
    for i, val in enumerate(y_pred):
        if val == 1 and y_true[i] == 1:
            hits += 1
            score_sum += hits / (i + 1)
    return score_sum / max(1, sum(y_true))

def run_evaluation(gold_summaries, model_summaries, baseline_summaries):
    f_scores = []
    map_scores = []

    if not gold_summaries or len(gold_summaries) != len(model_summaries):
        return None, None, None, None, None

    # Compute per-entity F1 and MAP
    for gold, pred in zip(gold_summaries, model_summaries):
        all_entities = list(gold.union(pred))
        y_true = np.array([1 if e in gold else 0 for e in all_entities])
        y_pred = np.array([1 if e in pred else 0 for e in all_entities])
        f_scores.append(f1_score(y_true, y_pred))
        map_scores.append(compute_average_precision(y_true, y_pred))

    avg_f1 = np.mean(f_scores)
    avg_map = np.mean(map_scores)

    # Statistical test
    stat = None
    pval = None
    test_name = None
    if baseline_summaries and len(baseline_summaries) == len(gold_summaries):
        baseline_f1 = []
        for gold, pred in zip(gold_summaries, baseline_summaries):
            all_entities = list(gold.union(pred))
            y_true = np.array([1 if e in gold else 0 for e in all_entities])
            y_pred = np.array([1 if e in pred else 0 for e in all_entities])
            baseline_f1.append(f1_score(y_true, y_pred))

        diffs = np.array(f_scores) - np.array(baseline_f1)
        try:
            # Wilcoxon fails if all diffs are zero
            if np.all(diffs == 0):
                baseline_f1 = [b + np.random.uniform(0, EPSILON) for b in baseline_f1]

            wil_res = wilcoxon(f_scores, baseline_f1)
            stat, pval = wil_res.statistic, wil_res.pvalue
            test_name = "Wilcoxon"
        except ValueError:
            t_res = ttest_rel(f_scores, baseline_f1)
            stat, pval = t_res.statistic, t_res.pvalue
            test_name = "Paired t-test"

    return avg_f1, avg_map, (stat, pval, test_name), f_scores, map_scores

# ================================
# MAIN EXECUTION
# ================================
final_results = defaultdict(dict)
per_entity_records = []
start_time = time.time()

baseline_available = os.path.isdir(BASELINE_OUTPUT_ROOT)
baseline_summaries_k = {}
if baseline_available:
    for k in TOP_K_SIZES:
        baseline_summaries_k[k] = load_output_summaries(BASELINE_OUTPUT_ROOT, k)

for k in TOP_K_SIZES:
    gold_standard = load_output_summaries(GOLD_SUMMARIES_ROOT, k)
    model_output = load_output_summaries(IRES_OUTPUT_ROOT, k)
    if not gold_standard or not model_output or len(gold_standard) != len(model_output):
        continue

    current_baseline = baseline_summaries_k.get(k)
    avg_f1, avg_map, stat_pval, f_scores, map_scores = run_evaluation(
        gold_standard, model_output, current_baseline
    )

    final_results[k]['f1'] = avg_f1
    final_results[k]['map'] = avg_map
    final_results[k]['stat_pval'] = stat_pval

    for idx, (f, mp) in enumerate(zip(f_scores, map_scores), 1):
        per_entity_records.append({'TopK': k, 'EntityID': idx, 'F1': f, 'MAP': mp})

runtime_seconds = time.time() - start_time

# ================================
# SAVE RESULTS
# ================================
with open(OUTPUT_RESULT_FILE, "w", encoding="utf-8") as f:
    f.write("="*40 + "\n")
    f.write(" QUANTITATIVE EVALUATION RESULTS (IRES vs. Baseline)\n")
    f.write("="*40 + "\n")
    for k in TOP_K_SIZES:
        res = final_results.get(k)
        if res and res['f1'] is not None:
            f.write(f"\n## Metrics @ Top {k}\n")
            f.write(f"Average F1-Measure: {res['f1']:.4f}\n")
            f.write(f"Average MAP: {res['map']:.4f}\n")
            f.write(f"Average NDCG: N/A (Unranked Set)\n")
            if res['stat_pval']:
                f.write(f"Test used: {res['stat_pval'][2]}\n")
                f.write(f"Test statistic: {res['stat_pval'][0]:.4f}\n")
                f.write(f"p-value: {res['stat_pval'][1]:.4f}\n")
            elif baseline_available:
                f.write("Statistical test not performed (data issue).\n")
            else:
                f.write("Baseline output missing, test not performed.\n")
    f.write(f"\nTotal Evaluation Runtime (seconds): {runtime_seconds:.2f}\n")

# Save per-entity CSV
pd.DataFrame(per_entity_records).to_csv(OUTPUT_CSV_FILE, index=False)
