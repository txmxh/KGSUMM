#!/usr/bin/env python3
"""
make_predicate_embeddings.py

Usage:
    python make_predicate_embeddings.py \
        --relation_map data/mydb_transE/relation2id.txt \
        --pred_csv data/WikiCinema-s-predicates.csv \
        --out_npz data/predicate_embs.npz \
        [--nt_dir datasets/WikiES_training_data/mydb/]  # optional: augment with contexts
"""
import os
import argparse
import numpy as np
import pandas as pd
from gensim.models import Word2Vec
from gensim.utils import simple_preprocess
from pathlib import Path
import glob

def read_relation2id(path):
    # supports optional count first line
    m = {}
    with open(path, "r", encoding="utf8") as f:
        lines = [ln.strip() for ln in f if ln.strip()]
    start = 1 if lines and lines[0].isdigit() else 0
    for ln in lines[start:]:
        parts = ln.split()
        if len(parts) >= 2:
            rel = parts[0]
            rid = int(parts[1])
            m[rel] = rid
    return m

def build_predicate_texts(rel_map, pred_csv_path):
    # pred_csv expected cols: id predicate predicate_label predicate_desc
    df = pd.read_csv(pred_csv_path, sep=None, engine="python")  # auto-detect sep
    # unify column names (lower)
    df.columns = [c.strip() for c in df.columns]
    # choose columns by name heuristics
    label_col = None
    desc_col = None
    for c in df.columns:
        lc = c.lower()
        if 'label' in lc and label_col is None:
            label_col = c
        if 'desc' in lc or 'description' in lc:
            desc_col = c
    if label_col is None:
        raise ValueError("Could not detect a predicate label column in CSV.")
    # Build mapping: predicate string (Pxxx or whatever) -> "label + description"
    pred_text = {}
    for _, row in df.iterrows():
        pred_key = str(row.get('predicate', row.get('id', None)))
        label = str(row.get(label_col, "")).strip()
        desc = str(row.get(desc_col, "")).strip() if desc_col else ""
        text = label
        if desc:
            text += " . " + desc
        pred_text[pred_key] = text
    # For any relations in rel_map missing from CSV, add placeholder
    for rel in rel_map.keys():
        if rel not in pred_text:
            pred_text[rel] = rel  # fallback: use rel token itself
    return pred_text

def tokenize_corpus(pred_texts, nt_dir=None):
    # pred_texts: dict rel -> text
    sentences = []
    # tokenize predicate label+desc
    for rel, text in pred_texts.items():
        toks = simple_preprocess(text)  # lower, split, remove punctuation
        if toks:
            sentences.append(toks)
    # optionally augment with contexts extracted from .nt files (subject/object tokens)
    if nt_dir:
        p = Path(nt_dir)
        if p.exists():
            for ntfile in p.rglob("*.nt"):
                with open(ntfile, "r", encoding="utf8", errors="ignore") as fr:
                    for ln in fr:
                        ln = ln.strip()
                        if not ln:
                            continue
                        parts = ln.split()
                        if len(parts) < 3:
                            continue
                        # naive: take angle-bracket content or fallback tokens
                        subj = parts[0].strip("<>")
                        pred = parts[1].strip("<>")
                        obj = parts[2].strip("<>").strip('\"')
                        # create small sentence from subject/pred/object labels (split on / and #)
                        for piece in (subj, obj):
                            tokens = [p for p in piece.replace("http://", "").replace("https://","").replace("/", " ").replace("#", " ").split() if p]
                            if tokens:
                                sentences.append([t.lower() for t in tokens])
    return sentences

def train_w2v(sentences, dim=100, window=5, min_count=1, epochs=50):
    model = Word2Vec(sentences=sentences, vector_size=dim, window=window,
                     min_count=min_count, sg=1, negative=10, epochs=epochs, workers=4)
    return model

def build_pred_embeddings(rel_map, pred_texts, w2v_model, dim):
    # produce numpy array of shape (n_rel, dim) where index = relation id
    n_rel = max(rel_map.values()) + 1
    emb_mat = np.zeros((n_rel, dim), dtype=np.float32)
    for rel, rid in rel_map.items():
        text = pred_texts.get(rel, rel)
        toks = simple_preprocess(text)
        tok_vecs = []
        for t in toks:
            if t in w2v_model.wv:
                tok_vecs.append(w2v_model.wv[t])
        if tok_vecs:
            emb = np.mean(tok_vecs, axis=0)
        else:
            emb = np.random.normal(scale=0.01, size=(dim,)).astype(np.float32)  # fallback vector
        emb_mat[rid] = emb
    return emb_mat

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--relation_map", default="/Users/praveen/Documents/Academics/Winter Sem 2025/KG SUMM/Week 07/ESA for WikiES/datasets/WikiES_benchmark/WikiCinema-SMALL-transE/relation2id.txt", help="path to relation2id.txt")
    parser.add_argument("--pred_csv", default="/Users/praveen/Documents/Academics/Winter Sem 2025/KG SUMM/Week 07/ESA for WikiES/datasets/WikiES_CSV/WikiCinema-s/WikiCinema-s-predicates.csv", help="path to WikiCinema-s-predicates.csv")
    parser.add_argument("--nt_dir", default=None, help="optional: path to folder with .nt files to augment corpus")
    parser.add_argument("--out_npz", default="/Users/praveen/Documents/Academics/Winter Sem 2025/KG SUMM/Week 07/ESA for WikiES/datasets/WikiES_benchmark/WikiCinema-SMALL-transE/predicateW2VEmbeddings")
    parser.add_argument("--dim", type=int, default=100)
    parser.add_argument("--epochs", type=int, default=50)
    args = parser.parse_args()

    rel_map = read_relation2id(args.relation_map)
    pred_texts = build_predicate_texts(rel_map, args.pred_csv)
    sentences = tokenize_corpus(pred_texts, nt_dir=args.nt_dir)
    print(f"Built corpus with {len(sentences)} sentences (predicates + optional contexts).")
    w2v = train_w2v(sentences, dim=args.dim, epochs=args.epochs)
    pred_emb = build_pred_embeddings(rel_map, pred_texts, w2v, args.dim)

    out_dir = os.path.dirname(args.out_npz)
    if out_dir and not os.path.exists(out_dir):
        os.makedirs(out_dir, exist_ok=True)
    # save embeddings + mapping (relation->id) and gensim model
    np.savez(args.out_npz, embeddings=pred_emb)
    w2v.save(args.out_npz + ".w2v.model")
    print("Saved predicate embeddings to", args.out_npz)
    print("Saved gensim model to", args.out_npz + ".w2v.model")

if __name__ == "__main__":
    main()
