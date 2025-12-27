#!/usr/bin/env python3
"""
Re-generate transE input files from raw .nt description files.

Usage (run from ESA/ root):
    python regen_transE_files.py <db-name>

Produces:
  ESA/data/<db-name>_transE/entity2id.txt
  ESA/data/<db-name>_transE/relation2id.txt
  ESA/data/<db-name>_transE/train2id.txt
  ESA/data/<db-name>_transE/regen_report.txt
"""
import os, sys, glob, re
from collections import OrderedDict, Counter
from urllib.parse import unquote, urlparse

# ------------------ Normalizer ------------------
_q_re = re.compile(r'\?oldid=.*$', re.IGNORECASE)
_query_re = re.compile(r'\?.*$')
_lang_dt_re = re.compile(r'(@[a-zA-Z\-]{2,10}$|\^\^<.*>$)')  # strip @en or ^^<...>
_trailing_slash_re = re.compile(r'/$')

def normalize_entity_key(raw: str) -> str:
    """
    Normalize a token from an .nt triple into a stable key.
    Strips angle brackets, query params (?...), fragments (#...), percent-decodes,
    removes language/datatype tags, trailing slashes, surrounding quotes,
    and collapses spaces into underscores.
    """
    if raw is None:
        return ""
    s = str(raw).strip()
    if not s:
        return ""
    # remove angle brackets if present
    if s.startswith("<") and s.endswith(">"):
        s = s[1:-1]
    # quick strip of language/datatype suffix when quoted literal
    s = s.strip()
    # Use urlparse to try to split URL-like tokens
    try:
        p = urlparse(s)
        # If there's a path with last segment, prefer it
        if p.path:
            last = p.path.split("/")[-1]
            if last:
                s = last
            else:
                # fallback to netloc or the path
                s = p.path or p.netloc or s
        else:
            # not a url with path; keep as is but drop query/fragments below
            s = s
    except Exception:
        s = s
    # remove common oldid/query patterns
    s = _q_re.sub("", s)
    s = _query_re.sub("", s)
    # percent-decode
    try:
        s = unquote(s)
    except Exception:
        pass
    # remove anything before a '#' leaving fragment tail
    if "#" in s:
        s = s.split("#")[-1]
    # strip language or datatype suffixes such as "foo"@en or "123"^^<...>
    s = _lang_dt_re.sub("", s)
    # remove trailing slash
    s = _trailing_slash_re.sub("", s)
    # remove surrounding quotes if any
    if s.startswith('"') and s.endswith('"'):
        s = s[1:-1]
    # collapse whitespace to underscore and remove extra underscores
    s = s.replace(" ", "_")
    s = re.sub(r'__+', '_', s)
    s = s.strip("_")
    # Ensure not empty
    if s == "":
        return "UNK"
    return s

# ------------------ Helpers ------------------
def resolve_desc_path(db_path: str, num: str):
    d = os.path.join(db_path, num)
    cand1 = os.path.join(d, f"{num}_desc.nt")
    if os.path.exists(cand1):
        return cand1
    cand2 = os.path.join(d, "desc.nt")
    if os.path.exists(cand2):
        return cand2
    # fallback: first .nt not containing _gold_
    hits = sorted(glob.glob(os.path.join(d, "*.nt")))
    for h in hits:
        if "_gold_" not in os.path.basename(h):
            return h
    return None

def parse_nt_simple(line: str):
    """
    Return (sub_raw, pred_raw, obj_raw) or (None,None,None) if unparsable.
    Accepts lines of the form: <s> <p> <o> .  (with possible literals).
    """
    if not line:
        return (None, None, None)
    ln = line.strip()
    if not ln or not ln.endswith('.'):
        return (None, None, None)
    # split into three main parts using '>' boundary for URIs where possible
    # but also allow space-splitting for simple cases
    try:
        # handle common '<...>' tokens reliably
        parts = []
        i = 0
        n = len(ln)
        while i < n and len(parts) < 3:
            if ln[i] == '<':
                # capture until next '>'
                j = ln.find('>', i+1)
                if j == -1:
                    break
                parts.append(ln[i:j+1])
                i = j+1
            elif ln[i] == '"':
                # capture quoted literal
                j = ln.find('"', i+1)
                # find closing quote (naive)
                if j == -1:
                    break
                # try to include language/datatype suffix if present
                k = j+1
                while k < n and ln[k] not in [' ', '.']:
                    k += 1
                parts.append(ln[i:k])
                i = k
            else:
                # other token (e.g., bareword)
                # take until next space
                j = ln.find(' ', i)
                if j == -1:
                    parts.append(ln[i:])
                    i = n
                else:
                    parts.append(ln[i:j])
                    i = j+1
            # skip whitespace
            while i < n and ln[i] == ' ':
                i += 1
        if len(parts) < 3:
            # fallback naive split
            sp = ln.split()
            if len(sp) < 3:
                return (None, None, None)
            s = sp[0]; p = sp[1]; o = " ".join(sp[2:])
            return (s, p, o)
        sub_raw, pred_raw, obj_raw = parts[0], parts[1], parts[2]
        return (sub_raw, pred_raw, obj_raw)
    except Exception:
        return (None, None, None)

# ------------------ Main regeneration ------------------
def build_transE(db_name: str):
    root = "/Users/praveen/Documents/Academics/Winter Sem 2025/KG SUMM/Week 07/ESA for WikiES/datasets/WikiES_benchmark/WikiCinema-SMALL"
    db_path = root
    if not os.path.exists(db_path):
        raise FileNotFoundError(f"datasets/WikiES_testing_data/{db_name} not found. Run from ESA/ root.")

    out_dir = "/Users/praveen/Documents/Academics/Winter Sem 2025/KG SUMM/Week 07/ESA for WikiES/datasets/WikiES_benchmark/WikiCinema-SMALL-transE"
    os.makedirs(out_dir, exist_ok=True)

    # collect numeric entity subdirs
    subdirs = sorted([d for d in os.listdir(db_path) if os.path.isdir(os.path.join(db_path, d)) and d.isdigit()],
                     key=lambda x: int(x))
    if not subdirs:
        print("[WARN] No numeric entity dirs found under", db_path)
        return

    ent2id = OrderedDict()
    rel2id = OrderedDict()
    triples = []
    malformed_lines = []
    counts_by_entity = {}
    # iterate in entity order for determinism
    for ent in subdirs:
        desc = resolve_desc_path(db_path, ent)
        if desc is None:
            counts_by_entity[ent] = 0
            continue
        good = 0
        with open(desc, "r", encoding="utf8") as f:
            for ln_no, ln in enumerate(f, start=1):
                s_raw, p_raw, o_raw = parse_nt_simple(ln)
                if s_raw is None:
                    malformed_lines.append((ent, desc, ln_no, ln.strip()))
                    continue
                s_key = normalize_entity_key(s_raw)
                p_key = normalize_entity_key(p_raw)
                o_key = normalize_entity_key(o_raw)
                # add to vocabs
                if s_key not in ent2id:
                    ent2id[s_key] = len(ent2id)
                if o_key not in ent2id:
                    ent2id[o_key] = len(ent2id)
                if p_key not in rel2id:
                    rel2id[p_key] = len(rel2id)
                triples.append((ent2id[s_key], ent2id[o_key], rel2id[p_key]))
                good += 1
        counts_by_entity[ent] = good

    # write entity2id.txt
    ent_file = os.path.join(out_dir, "entity2id.txt")
    with open(ent_file, "w", encoding="utf8") as fe:
        fe.write(str(len(ent2id)) + "\n")
        for k, v in ent2id.items():
            fe.write(f"{k}\t{v}\n")

    # write relation2id.txt
    rel_file = os.path.join(out_dir, "relation2id.txt")
    with open(rel_file, "w", encoding="utf8") as fr:
        fr.write(str(len(rel2id)) + "\n")
        for k, v in rel2id.items():
            fr.write(f"{k}\t{v}\n")

    # write train2id.txt with header count
    train_file = os.path.join(out_dir, "train2id.txt")
    with open(train_file, "w", encoding="utf8") as ft:
        ft.write(str(len(triples)) + "\n")
        for s, o, r in triples:
            ft.write(f"{s}\t{o}\t{r}\n")

    # write report
    report_file = os.path.join(out_dir, "regen_report.txt")
    with open(report_file, "w", encoding="utf8") as rep:
        rep.write(f"db: {db_name}\n")
        rep.write(f"entity_dirs_processed: {len(subdirs)}\n")
        rep.write(f"entities_in_vocab: {len(ent2id)}\n")
        rep.write(f"relations_in_vocab: {len(rel2id)}\n")
        rep.write(f"triples_written: {len(triples)}\n\n")
        rep.write("sample entity counts per entity-dir (first 50):\n")
        for e in list(counts_by_entity.items())[:50]:
            rep.write(f"{e[0]}: {e[1]}\n")
        rep.write("\nmalformed/skipped lines (first 50):\n")
        for entry in malformed_lines[:50]:
            rep.write(f"{entry}\n")
    print("[OK] Wrote transE files to", out_dir)
    print("[OK] entity_count:", len(ent2id), "rel_count:", len(rel2id), "triples:", len(triples))
    print("[INFO] Report at", report_file)

# ------------------ CLI ------------------
if __name__ == "__main__":
    if len(sys.argv) < 2:
        print("Usage: python regen_transE_files.py <db-name>")
        sys.exit(1)
    dbname = sys.argv[1]
    build_transE(dbname)
