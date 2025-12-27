from mlcroissant import Dataset
from collections import defaultdict
import os

JSON_PATH = "./json/WikiPro-s-test.json"
OUTPUT_DIR = "../datasets/WikiES_benchmark"
ENTITY_BASE = "http://www.wikidata.org/entity/"
PROP_BASE   = "http://www.wikidata.org/prop/direct/"

def _load_entities(dataset):
    entities = {}
    for row in dataset.records("entities"):
        # print("RAW:", row)

        eid = row["entities/id"]
        
        entity_id = row["entities/entity"] # entities/entity -> ID code | entities/wikidata_label -> label
        if isinstance(entity_id, bytes):
            entity_id = entity_id.decode("utf-8")
        
        # entity_label = row["entities/wikidata_label"]
        # if isinstance(entity_label, bytes):
        #     entity_label = entity_label.decode("utf-8")
        
        # if entity_label is None:
        #     entity_label = entity_id
        
        entities[eid] = entity_id # entity_id if want id code | entity_label if want label

    print(f"[INFO] Loaded {len(entities)} entities.")
    return entities

def _load_predicates(dataset):
    predicates = {}
    for row in dataset.records("predicates"):
        # print ("RAW:", row)

        pid = row["predicates/id"]

        predicate_id = row["predicates/predicate"]
        if isinstance(predicate_id, bytes):
            predicate_id = predicate_id.decode("utf-8")

        # predicate_label = row["predicates/predicate_label"] # predicates/predicate -> ID code | predicates/predicate_label -> label
        # if isinstance(predicate_label, bytes):
        #     predicate_label = predicate_label.decode("utf-8")
        
        # if predicate_label is None:
        #     predicate_label = predicate_id

        predicates[pid] = predicate_id # predicate_id if want id code | predicate_label if want label

    print(f"[INFO] Loaded {len(predicates)} predicates.")
    return predicates

def _index_triples(dataset):
    """
    Build: entity_id -> list of triples (s,p,o)
    for all triples where entity appears in subject OR object.
    """
    triples_by_entity = defaultdict(list)
    total = 0

    for row in dataset.records("triples"):
        # print("RAW:", row)
        s, p, o = row["triples/subject"], row["triples/predicate"], row["triples/object"]

        triples_by_entity[s].append((s, p, o))  #  appears as subject
        triples_by_entity[o].append((s, p, o))  #  appears as object

        total += 1

    print(f"[INFO] Loaded {total} triples. Entities participating: {len(triples_by_entity)}")
    return triples_by_entity

def _load_root_entities(dataset):
    roots = set()
    for row in dataset.records("root-entities"):
        # print("RAW:", row)
        roots.add(row["root_entities/entity"])

    print(f"[INFO] Loaded {len(roots)} root entities.")
    return roots

def _load_ground_truths(dataset):
    ground_truths_summary = defaultdict(list)
    total = 0
    for row in dataset.records("ground-truths"):
        # print("RAW:", row)

        root = row["ground_truths/root_entity"]
        s, p, o = row["ground_truths/subject"], row["ground_truths/predicate"], row["ground_truths/object"]
        ground_truths_summary[root].append((s, p, o))
        total += 1

    print(f"[INFO] Loaded {total} gold triples for {len(ground_truths_summary)} roots.")
    return ground_truths_summary

def _write_nt(f, entities, predicates, s, p, o):
    subj = entities.get(s)
    pred = predicates.get(p)
    obj = entities.get(o)

    if subj is None or pred is None or obj is None:
        print(f"[WARN] Missing entity or predicate for triple ({s}, {p}, {o}). Skipping...")
        
        if subj is None:
            print(f"[DEBUG] Missing SUBJECT id={s} (p={p}, o={o})")

        if pred is None:
            print(f"[DEBUG] Missing PREDICATE id={p} (s={s}, o={o})")

        if obj is None:
            print(f"[DEBUG] Missing OBJECT id={o} (s={s}, p={p})")
        
        return False
    
    subj_uri = f"{ENTITY_BASE}{subj}"
    pred_uri = f"{PROP_BASE}{pred}"
    obj_uri  = f"{ENTITY_BASE}{obj}"
    
    f.write(f"<{subj_uri}> <{pred_uri}> <{obj_uri}> .\n")
    return True


def main():
    print(f"[INFO] Loading dataset from: {JSON_PATH}")
    dataset = Dataset(jsonld=JSON_PATH)

    print("[INFO] Record sets:", dataset.metadata.record_sets)
    
    entities = _load_entities(dataset)
    predicates = _load_predicates(dataset)
    triples_by_ent = _index_triples(dataset)
    roots = _load_root_entities(dataset)
    golds = _load_ground_truths(dataset)
    
    roots |= set(golds.keys()) # inplace or add all keys from golds to roots
    print(f"[INFO] Total root entities to process: {len(roots)}")

    dataset_file = os.path.basename(JSON_PATH)
    dataset_name, _ = os.path.splitext(dataset_file)
    dataset_dir = dataset_name + "_data"

    data_dir = os.path.join(OUTPUT_DIR, dataset_dir)
    os.makedirs(data_dir, exist_ok=True)

    for root in sorted(roots):
        if root not in entities:
            print(f"[WARN] Root entity id {root} not found in entities. Skipping...")
            continue
        
        root_dir = os.path.join(data_dir, str(root))
        os.makedirs(root_dir, exist_ok=True)

        # Description file
        desc_path = os.path.join(root_dir, f"{root}_desc.nt")
        triples = triples_by_ent.get(root, [])

        with open(desc_path, "w", encoding="utf-8") as f:
            count_written = 0
            for s, p, o in triples:
                if s == root:
                    if _write_nt(f, entities, predicates, s, p, o):
                        count_written += 1
                
                elif o == root:
                    if _write_nt(f, entities, predicates, root, p, s):
                        count_written += 1
                
                else:
                    continue
        
        # Gold Summary file
        gold_path = os.path.join(root_dir, f"{root}_gold.nt")
        gold_triples = golds.get(root, [])

        with open(gold_path, "w", encoding="utf-8") as f:
            gold_written = 0
            for s, p, o in gold_triples:
                if s == root:
                    if _write_nt(f, entities, predicates, s, p, o):
                        gold_written += 1
                    
                elif o == root:
                    if _write_nt(f, entities, predicates, root, p, s):
                        gold_written += 1
                
                else:
                    continue
        
        print(f"[INFO] Root {root}: "
              f"{len(triples)} description triples → {count_written} written; "
              f"{len(gold_triples)} gold → {gold_written} written."
            )
    
    print(f"[DONE] Dataset converted. Output inside: {data_dir}")

if __name__ == "__main__":
    main()