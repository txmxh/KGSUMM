#!/usr/bin/env python3
# train_transe_pytorch.py
import os, sys, numpy as np, argparse, random, math
import torch
import torch.nn as nn
from tqdm import trange

# Simple TransE model
class TransE(nn.Module):
    def __init__(self, n_ent, n_rel, dim, p_norm=1, margin=1.0):
        super().__init__()
        self.emb_e = nn.Embedding(n_ent, dim)
        self.emb_r = nn.Embedding(n_rel, dim)
        self.dim = dim
        self.p_norm = p_norm
        self.margin = margin
        nn.init.xavier_uniform_(self.emb_e.weight.data)
        nn.init.xavier_uniform_(self.emb_r.weight.data)

    def forward_score(self, s, o, r):
        # s,o: (batch,), r: (batch,)
        s_e = self.emb_e(s)
        o_e = self.emb_e(o)
        r_e = self.emb_r(r)
        return torch.norm(s_e + r_e - o_e, p=self.p_norm, dim=1)

    def normalize_embeddings(self):
        # L2 normalize entities & relations
        with torch.no_grad():
            e = self.emb_e.weight.data
            e_norm = e / (e.norm(p=2, dim=1, keepdim=True) + 1e-12)
            self.emb_e.weight.copy_(e_norm)
            r = self.emb_r.weight.data
            r_norm = r / (r.norm(p=2, dim=1, keepdim=True) + 1e-12)
            self.emb_r.weight.copy_(r_norm)

def read_map(path):
    with open(path, "r", encoding="utf8") as f:
        lines = f.read().splitlines()
    # allow optional first-line count
    start = 1 if lines and lines[0].strip().isdigit() and "\t" not in lines[0] else 0
    m = {}
    for ln in lines[start:]:
        parts = ln.strip().split()
        if len(parts) >= 2:
            m[parts[0]] = int(parts[1])
    return m

def read_train(path):
    with open(path, "r", encoding="utf8") as f:
        lines = [ln.strip() for ln in f if ln.strip()]
    start = 1 if lines and lines[0].isdigit() else 0
    triples=[]
    for ln in lines[start:]:
        a = ln.split()
        if len(a) >= 3:
            triples.append((int(a[0]), int(a[1]), int(a[2])))
    return triples

def neg_sample(triples, n_ent):
    # return a corrupted triple by corrupting head or tail
    s,o,r = triples
    if random.random() < 0.5:
        s = random.randrange(n_ent)
    else:
        o = random.randrange(n_ent)
    return (s,o,r)

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--data_dir", default="/Users/praveen/Documents/Academics/Winter Sem 2025/KG SUMM/Week 07/ESA for WikiES/datasets/WikiES_benchmark/WikiCinema-SMALL-transE")
    parser.add_argument("--dim", type=int, default=100)
    parser.add_argument("--margin", type=float, default=1.0)
    parser.add_argument("--lr", type=float, default=0.001)
    parser.add_argument("--batch_size", type=int, default=512)
    parser.add_argument("--epochs", type=int, default=50)
    parser.add_argument("--device", default="cpu")
    parser.add_argument("--save_npz", default="/Users/praveen/Documents/Academics/Winter Sem 2025/KG SUMM/Week 07/ESA for WikiES/datasets/WikiES_benchmark/WikiCinema-SMALL-transE")
    parser.add_argument("--normalize", action="store_true")
    parser.add_argument("--seed", type=int, default=42)
    args = parser.parse_args()

    random.seed(args.seed)
    np.random.seed(args.seed)
    torch.manual_seed(args.seed)

    ent_map = read_map(os.path.join(args.data_dir, "entity2id.txt"))
    rel_map = read_map(os.path.join(args.data_dir, "relation2id.txt"))
    triples = read_train(os.path.join(args.data_dir, "train2id.txt"))
    n_ent = len(ent_map)
    n_rel = len(rel_map)
    print("n_ent", n_ent, "n_rel", n_rel, "triples", len(triples))

    device = torch.device(args.device)
    model = TransE(n_ent, n_rel, args.dim, p_norm=1, margin=args.margin).to(device)
    opt = torch.optim.Adam(model.parameters(), lr=args.lr)

    # convert triples to numpy for sampling
    triple_arr = np.array(triples, dtype=np.int64)
    N = len(triple_arr)

    for epoch in range(args.epochs):
        model.train()
        total_loss = 0.0
        idxs = np.random.permutation(N)
        for i in range(0, N, args.batch_size):
            batch_idx = idxs[i:i+args.batch_size]
            batch = triple_arr[batch_idx]
            s = torch.tensor(batch[:,0], dtype=torch.long, device=device)
            o = torch.tensor(batch[:,1], dtype=torch.long, device=device)
            r = torch.tensor(batch[:,2], dtype=torch.long, device=device)
            # negative samples
            s_neg = s.clone().cpu().numpy()
            o_neg = o.clone().cpu().numpy()
            for bi in range(len(batch)):
                if random.random() < 0.5:
                    s_neg[bi] = random.randrange(n_ent)
                else:
                    o_neg[bi] = random.randrange(n_ent)
            s_neg = torch.tensor(s_neg, dtype=torch.long, device=device)
            o_neg = torch.tensor(o_neg, dtype=torch.long, device=device)

            pos_score = model.forward_score(s,o,r)
            neg_score = model.forward_score(s_neg,o_neg,r)
            # margin ranking loss: want pos < neg (distance)
            target = torch.ones_like(pos_score, device=device)
            loss = torch.mean(torch.clamp(pos_score - neg_score + args.margin, min=0.0))
            opt.zero_grad()
            loss.backward()
            opt.step()
            total_loss += loss.item() * len(batch)
        avg_loss = total_loss / N
        if args.normalize:
            model.normalize_embeddings()
        print(f"Epoch {epoch+1}/{args.epochs} avg_loss={avg_loss:.6f}")
    # final normalize
    model.normalize_embeddings()
    ent_emb = model.emb_e.weight.data.cpu().numpy()
    rel_emb = model.emb_r.weight.data.cpu().numpy()
    np.savez(args.save_npz, ent_embedding=ent_emb, rel_embedding=rel_emb)
    # also copy entity2id/relation2id alongside saved npz directory for utils.load_transE
    out_dir = os.path.dirname(args.save_npz) or "."
    with open(os.path.join(args.data_dir, "entity2id.txt"), "r", encoding="utf8") as fr:
        open(os.path.join(out_dir, "entity2id.txt"), "w", encoding="utf8").write(fr.read())
    with open(os.path.join(args.data_dir, "relation2id.txt"), "r", encoding="utf8") as fr:
        open(os.path.join(out_dir, "relation2id.txt"), "w", encoding="utf8").write(fr.read())
    print("Saved embeddings to", args.save_npz)

if __name__ == "__main__":
    main()
