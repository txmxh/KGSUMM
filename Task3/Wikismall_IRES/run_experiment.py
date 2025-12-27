import subprocess
import re
import csv
import os
import sys

# --- CONFIGURATION ---
# The command we successfully built in the previous steps
COMMAND = [
    "python3", "main.py",
    "-b", "esbm",
    "-d", "wikies_small",
    "-p", "./data/wikies_small",
    "-f", "json",
    "-tf", "node_freq",
    "-e", "RGCN",
    "-ne", "30",
    "-lr", "0.01"
]

OUTPUT_CSV = "final_results.csv"

def parse_and_save(output_text):
    """
    Looks for the specific output lines from main.py and saves them to CSV.
    Target lines look like:
    Fscore_5: 0.306, Fscore_10: 0.535
    MAP_5: 0.31, MAP_10: 0.505
    """
    # Regex patterns to find the numbers
    fscore_pattern = r"Fscore_5:\s*([\d\.]+),\s*Fscore_10:\s*([\d\.]+)"
    map_pattern = r"MAP_5:\s*([\d\.]+),\s*MAP_10:\s*([\d\.]+)"

    fscore_match = re.search(fscore_pattern, output_text)
    map_match = re.search(map_pattern, output_text)

    results = {
        "Model": "IRES-RGCN",
        "Dataset": "WikiES-SMALL",
        "Epochs": 50,
        "F1_Top5": "N/A",
        "F1_Top10": "N/A",
        "MAP_Top5": "N/A",
        "MAP_Top10": "N/A"
    }

    if fscore_match:
        results["F1_Top5"] = fscore_match.group(1)
        results["F1_Top10"] = fscore_match.group(2)
    
    if map_match:
        results["MAP_Top5"] = map_match.group(1)
        results["MAP_Top10"] = map_match.group(2)

    # Save to CSV
    file_exists = os.path.isfile(OUTPUT_CSV)
    
    with open(OUTPUT_CSV, mode='a', newline='') as f:
        writer = csv.DictWriter(f, fieldnames=results.keys())
        if not file_exists:
            writer.writeheader()
        writer.writerow(results)

    print(f"\nSuccess! Metrics saved to {OUTPUT_CSV}")
    print(f"   F1 (Top 5/10): {results['F1_Top5']} / {results['F1_Top10']}")
    print(f"   MAP (Top 5/10): {results['MAP_Top5']} / {results['MAP_Top10']}")

def run():
    print(f"Starting Experiment...")
    print(f"   Command: {' '.join(COMMAND)}\n")

    # Run the command and capture output while streaming it to console
    process = subprocess.Popen(
        COMMAND, 
        stdout=subprocess.PIPE, 
        stderr=subprocess.STDOUT, 
        text=True, 
        bufsize=1, 
        universal_newlines=True
    )

    full_output = []
    
    # Stream output to console line by line
    for line in process.stdout:
        print(line, end='') # Print to console live
        full_output.append(line)

    process.wait()
    
    if process.returncode == 0:
        # If successful, parse the full captured log
        parse_and_save("".join(full_output))
    else:
        print("\nExperiment failed. Check errors above.")

if __name__ == "__main__":
    run()