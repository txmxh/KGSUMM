%%bash

# 1.  all paths as shell variables
JAR_FILE_DIR="/content/Unsupervised_Learning_IRES_Model/ESBM_v1.2/Evaluation"
ELIS_FILE_PATH="/content/Unsupervised_Learning_IRES_Model/ESBM_v1.2/Tool/ESBM-groundtruth/elist.txt"

# ---  new, correct path for the list files ---
BENCHMARK_DIR="/content/Unsupervised_Learning_IRES_Model/ESBM_v1.2/Dataset/ESBM_benchmark_v1.2"
SUMMARY_DIR="/content/Unsupervised_Learning_IRES_Model/ESBM_v1.2/Dataset/output_summaries"


# 2. Create the list files (this part worked) and save them in the correct BENCHMARK_DIR
echo "--- Creating 'dbpedia_data' from $ELIS_FILE_PATH ---"
awk -F '\t' '/dbpedia/ {print $1}' "$ELIS_FILE_PATH" > "$BENCHMARK_DIR/dbpedia_data"

echo "--- Creating 'lmdb_data' from $ELIS_FILE_PATH ---"
awk -F '\t' '/lmdb/ {print $1}' "$ELIS_FILE_PATH" > "$BENCHMARK_DIR/lmdb_data"
echo "--- List files created successfully in $BENCHMARK_DIR ---"


# 3. containing the .jar file
cd $JAR_FILE_DIR

# 4. ---Run the evaluation ONE time, as shown in the README.md ---
# 1st Argument: The benchmark folder (now contains lists + data folders)
# 2nd Argument: The parent summary folder (contains /dbpedia and /lmdb)

echo -e "\n--- Running GATES (Baseline) Evaluation for DBPedia and LMDB ---"
java -jar esummeval_v1.2.jar \
$BENCHMARK_DIR \
$SUMMARY_DIR
