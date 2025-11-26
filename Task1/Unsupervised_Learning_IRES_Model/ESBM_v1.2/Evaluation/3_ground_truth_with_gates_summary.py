%%bash

# --- 1. Print the "Ground Truth" (the correct answer) for entity 1 ---
echo "==========================================================="
echo "GROUND TRUTH (Entity 1, Top 10)"
echo "==========================================================="
cat /content/KGSUMM/Task1/Unsupervised_Learning_IRES_Model/ESBM_v1.2/Tool/ESBM-groundtruth/groundtruth/1/1_gold_top10_0.nt

# --- 2. Print the GATES (Baseline) model's summary for entity 1 ---
echo -e "\n\n==========================================================="
echo "GATES (Baseline) Summary (Entity 1, Top 10)----DBPEDIA"
echo "==========================================================="
cat /content/KGSUMM/Task1/Unsupervised_Learning_IRES_Model/ESBM_v1.2/Dataset/output_summaries/dbpedia/1/1_top10.nt


echo -e "\n\n==========================================================="
echo "GROUND TRUTH (Entity 101, Top 10)"
echo "==========================================================="
cat /content/KGSUMM/Task1/Unsupervised_Learning_IRES_Model/ESBM_v1.2/Tool/ESBM-groundtruth/groundtruth/101/101_gold_top10_0.nt

echo -e "\n==========================================================="
echo "GATES (Baseline) Summary (Entity 101, Top 10)----LMDB"
echo "==========================================================="
cat /content/KGSUMM/Task1/Unsupervised_Learning_IRES_Model/ESBM_v1.2/Dataset/output_summaries/lmdb/101/101_top10.nt