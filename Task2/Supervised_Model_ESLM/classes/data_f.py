import os
import re
import numpy as np
import datetime
from urllib.parse import urlparse

# Functions for data loading and preprocessing
def build_dictionary(input_dict):
    """Build dictionary from TSV index file (e.g., entities.tsv)"""
    
    try:
        f = open(input_dict, "r", encoding="utf-8")
        content = f.readlines()
        f.close()
    except FileNotFoundError as e:
        # Re-raise the error with the correct path context for debugging
        raise FileNotFoundError(f"Error loading dictionary: {input_dict}. Please ensure index files are present.") from e
        
    idx2dict = dict()
    for items in content:
        items = items.replace("\n", "")
        items = items.split("\t")
        if len(items) >= 2: # Guard against malformed lines in index files
            idx = int(items[0])
            dict_value = items[1]
            idx2dict[dict_value]=idx
    
    return idx2dict

# Load DGLKE (where the file path error occurs)
def load_dglke(ds_name):
    """Load pre-trained graph embeddings, prioritizing local FACES structure."""
    
    project_root = os.getcwd() # Should be the /FACES/ folder
    
    # Construct paths to the required KGE index files (entities.tsv, relations.tsv)
    # Assumes index files are under 'data_kge/{ds_name}-esbm'
    index_base_path = os.path.join(project_root, 'data_kge', f'{ds_name}-esbm')

    # Construct paths to the required KGE vector files (.npy)
    # Assumes vectors are under 'faces-kge-model/ComplEx_{ds_name}'
    vector_base_path = os.path.join(project_root, 'faces-kge-model', f'ComplEx_{ds_name}')
    
    # 1. Load Dictionaries (Index Files)
    input_entity_dict = os.path.join(index_base_path, "entities.tsv")
    input_relation_dict = os.path.join(index_base_path, "relations.tsv")

    entity2ix = build_dictionary(input_entity_dict)
    pred2ix = build_dictionary(input_relation_dict)
    
    # 2. Load Vectors
    entity_vec_file = os.path.join(vector_base_path, f'{ds_name}_ComplEx_entity.npy')
    pred_vec_file = os.path.join(vector_base_path, f'{ds_name}_ComplEx_relation.npy')
    
    try:
        entity2vec = np.load(entity_vec_file, mmap_mode='r')
        pred2vec = np.load(pred_vec_file, mmap_mode='r')
    except FileNotFoundError as e:
        print(f"CRITICAL ERROR: KGE Vector file not found for {ds_name}. Looked at: {entity_vec_file}")
        raise FileNotFoundError(f"Missing KGE vector file for {ds_name}. Please ensure {os.path.basename(entity_vec_file)} is correctly placed.") from e

    return entity2vec, pred2vec, entity2ix, pred2ix

# --- Rest of your functions (format_triples, writer, etc.) follow ---
def format_triples(triples): 
    formatted_triples=[]
    for triple in triples:
        h, r, t = triple
        # head
        head = h.split('/')[-1]
        # relation
        try: 
            clean_relation= r.split('/')[-1]
        except Exception as e:
            print (triple)
            print (e)
            break 
        clean_relation= re.sub(r'.*#', '', clean_relation)        
        relation = clean_relation
        #tail
        t = str(t)
        if t.startswith('http://'): # check if the tail is not literal
            tail= t.split('/')[-1]
        else:
            clean_literal= re.sub(r'\^\^<http.*', '', t)
            clean_literal=clean_literal.replace('"','')
            clean_literal=clean_literal.replace('@e','')
            tail= clean_literal
        if len(tail)>0:
            input_formatted = f"{head}[SEP]{relation}[SEP]{tail}"
            formatted_triples.append(input_formatted)
        else:
            input_formatted = f"{head}[SEP]{relation}"
            formatted_triples.append(input_formatted)
    return formatted_triples

def writer(db_dir, directory, eid, top_or_rank, topk, rank_list):
    "Write triples to file"
    with open(os.path.join(db_dir, f"{eid}", f"{eid}_desc.nt"), encoding="utf8") as fin:
        with open(os.path.join(directory, f"{eid}_{top_or_rank}{topk}.nt"), "w", encoding="utf8") as fout:
            triples = [triple for _, triple in enumerate(fin)]
            for rank in rank_list:
                fout.write(triples[rank])

def get_rank_triples(db_path, num, top_n, triples_dict):
    triples=[]
    encoded_triples = []
    filename = os.path.join(db_path, "{}".format(num), "{}_rank.nt".format(num))
    if os.path.exists(os.path.join(db_path, "{}".format(num), "{}_rank_top{}.nt".format(num, top_n))):
        filename = os.path.join(db_path, "{}".format(num), "{}_rank_top{}.nt".format(num, top_n))
    with open(filename, encoding="utf8") as reader:
        for i, triple in enumerate(reader):
            triple = triple.replace("\n", "").strip()
            triples.append(triple)

            encoded_triple = triples_dict[triple]
            encoded_triples.append(encoded_triple)
    return triples, encoded_triples

def get_topk_triples(db_path, num, top_n, triples_dict):
    triples=[]
    encoded_triples = []

    with open(os.path.join(db_path, "{}".format(num), "{}_top{}.nt".format(num, top_n)), encoding="utf8") as reader:
        for i, triple in enumerate(reader):
            triple = triple.replace("\n", "").strip()
            triples.append(triple)

            encoded_triple = triples_dict[triple]
            encoded_triples.append(encoded_triple)
    return triples, encoded_triples


def get_all_data(db_path, num, top_n, file_n):
    import glob
    triples_dict = {}
    triple_tuples = []
    ### Retrieve all triples of an entity based on eid
    with open(os.path.join(db_path, "{}".format(num), "{}_desc.nt".format(num)), encoding="utf8") as reader:
        for i, triple in enumerate(reader):
            if len(triple)==1:
                continue
            triple_tuple = triple.replace("\n", "").strip()#parserline(triple)
            triple_tuples.append(triple_tuple)
            if triple_tuple not in triples_dict:
                triples_dict[triple_tuple] = len(triples_dict)
    gold_list = []
    ds_name = db_path.split("/")[-1].split("_")[0]

    ### Get file_n/ n files of ground truth summaries for faces dataset
    if ds_name=="faces":
        gold_files = glob.glob(os.path.join(db_path, "{}".format(num), "{}_gold_top{}_*".format(num, top_n).format(num)))
        #print(len(gold_files))
        if len(gold_files) != file_n:
            file_n = len(gold_files)

    ### Retrieve ground truth summaries of an entity based on eid and total of file_n
    for i in range(file_n):
        with open(os.path.join(db_path,
            "{}".format(num),
            "{}_gold_top{}_{}.nt".format(num, top_n, i).format(num)),
            encoding="utf8") as reader:
            #print(path.join(db_path, "{}".format(num), "{}_gold_top{}_{}.nt".format(num, top_n, i).format(num)))
            n_list = []
            for i, triple in enumerate(reader):
                if len(triple)==1:
                    continue
                triple_tuple = triple.replace("\n", "").strip()#parserline(triple)
                gold_id = triples_dict[triple_tuple]
                n_list.append(gold_id)
            gold_list.append(n_list)

    return gold_list, triples_dict, triple_tuples

def format_time(elapsed):
    """Takes a time in seconds and returns a string hh:mm:ss"""
    # Round to the nearest second.
    elapsed_rounded = int(round((elapsed)))
    # Format as hh:mm:ss
    return str(datetime.timedelta(seconds=elapsed_rounded))