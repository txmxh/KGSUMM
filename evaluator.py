#!/usr/bin/env python3
"""
computeMetrics.py

Compute F1 (average across annotators), NDCG and MAP for predicted triples
compared to gold triples (dynamic / top5 / top10 modes).

Directory assumptions (per-entity directories numbered 1..N):

Gold root (example):
<goldRoot>/1/1_gold_dynamic.nt
<goldRoot>/1/1_gold_top5_0.nt
<goldRoot>/1/1_gold_top10_0.nt
...

Predictions root (example):
<predRoot>/1/1_rank.nt
<predRoot>/1/1_top5.nt
<predRoot>/1/1_top10.nt
...
"""

from __future__ import annotations
import os
import argparse
import math
import numpy as np
from typing import List, Dict, Iterable


# ---------------------------
# Metric classes (CamelCase)
# ---------------------------
class FMeasure:
    """Average F1 across multiple gold summaries.

    summTriples: ordered list of predicted triple identifiers (strings)
    goldSummaries: list of gold summaries (each a list of triple identifiers)
    """

    @staticmethod
    def GetScore(summTriples: List[str], goldSummaries: Iterable[Iterable[str]]) -> float:
        k = len(summTriples)
        if k == 0:
            return 0.0
        f_list = []
        for gold in goldSummaries:
            goldList = list(gold)
            # compute how many predicted triples are present in this gold
            corr = sum(1 for t in summTriples if t in goldList)
            precision = corr / k if k > 0 else 0.0
            recall = corr / len(goldList) if len(goldList) > 0 else 0.0
            if (precision + recall) == 0:
                f_score = 0.0
            else:
                f_score = 2 * (precision * recall) / (precision + recall)
            f_list.append(f_score)
        return float(np.mean(f_list)) if f_list else 0.0

    def __repr__(self):
        return self.__class__.__name__


class NDCG:
    """Normalized Discounted Cumulative Gain.

    Input:
      goldSummaries: list of gold summary lists; each element is a triple identifier (string)
      predictedRankedTriples: ordered list of triple identifiers (strings)
    """

    def GetScore(self, goldSummaries: Iterable[Iterable[str]], predictedRankedTriples: List[str]) -> float:
        gradeList, tripleGrade = self.SetGoldTriplesDict(goldSummaries)

        # DCG
        dcg = 0.0
        for pos, triple in enumerate(predictedRankedTriples, start=1):
            rel = tripleGrade.get(triple, 0)
            dcg += rel / math.log2(pos + 1)

        # IDCG (ideal order)
        idcg = 0.0
        for pos, idealRel in enumerate(gradeList, start=1):
            idcg += idealRel / math.log2(pos + 1)

        if idcg == 0.0:
            return 0.0
        return dcg / idcg

    @staticmethod
    def SetGoldTriplesDict(goldSummaries: Iterable[Iterable[str]]):
        tripleGrade: Dict[str, int] = {}
        for gold in goldSummaries:
            for triple in gold:
                tripleGrade[triple] = tripleGrade.get(triple, 0) + 1
        gradeList = sorted(tripleGrade.values(), reverse=True)
        return gradeList, tripleGrade

    def __repr__(self):
        return self.__class__.__name__


class MAP:
    """Mean Average Precision across multiple gold summaries."""

    def GetMap(self, summTriples: List[str], goldSummList: Iterable[Iterable[str]]) -> float:
        goldSumms = list(goldSummList)
        if len(goldSumms) == 0:
            return 0.0
        sum_ap = 0.0
        for gold in goldSumms:
            sum_ap += self.GetAveragePrecision(summTriples, list(gold))
        return sum_ap / len(goldSumms)

    def GetAveragePrecision(self, summTriples: List[str], goldSumm: List[str]) -> float:
        if len(goldSumm) == 0:
            return 0.0
        avg_p = 0.0
        hits = 0
        for i in range(1, len(summTriples) + 1):
            if summTriples[i - 1] in goldSumm:
                hits += 1
                avg_p += self.GetPrecisionAtK(summTriples[:i], goldSumm)
        if len(goldSumm) == 0:
            return 0.0
        return avg_p / len(goldSumm)

    @staticmethod
    def GetPrecisionAtK(prefixSumm: List[str], goldSumm: List[str]) -> float:
        k = len(prefixSumm)
        if k == 0:
            return 0.0
        corr = sum(1 for t in prefixSumm if t in goldSumm)
        return corr / k

    def __repr__(self):
        return self.__class__.__name__


# ------------------------------------
# Utility functions 
# ------------------------------------
def ReadTriples(filePath: str) -> List[str]:
    """Read triple lines from a .nt file; normalize by stripping whitespace."""
    triples = []
    if not os.path.exists(filePath):
        return triples
    with open(filePath, "r", encoding="utf8") as fh:
        for line in fh:
            s = line.strip()
            if s:
                triples.append(s)
    return triples


def GatherGoldSummaries(goldDir: str, entityId: str, mode: str) -> List[List[str]]:
    """
    Collect all gold summaries for a given entity and mode.
    mode in {'dynamic','top5','top10'}.
    Returns list of gold lists (each list is ordered list of triple strings).
    """
    golds = []
    if mode == "dynamic":
        pattern = f"{entityId}_gold_dynamic"
    elif mode == "top5":
        pattern = f"{entityId}_gold_top5"
    elif mode == "top10":
        pattern = f"{entityId}_gold_top10"
    else:
        raise ValueError("mode must be one of 'dynamic','top5','top10'")

    # include any file that starts with that pattern (supports suffixes like _0, _1)
    for fname in sorted(os.listdir(goldDir)):
        if fname.startswith(pattern) and fname.endswith(".nt"):
            golds.append(ReadTriples(os.path.join(goldDir, fname)))

    return golds


def GetPredictedTriples(predDir: str, entityId: str, mode: str, dynamicNum: int = None) -> List[str]:
    """
    Read predicted triples for the entity depending on mode.
    - dynamic: read <id>_rank.nt and take top dynamicNum triples (dynamicNum required)
    - top5: read <id>_top5.nt
    - top10: read <id>_top10.nt
    """
    if mode == "dynamic":
        rankFile = os.path.join(predDir, f"{entityId}", f"{entityId}_rank.nt")
        ranked = ReadTriples(rankFile)
        if dynamicNum is None:
            return ranked
        return ranked[:dynamicNum]
    elif mode == "top5":
        return ReadTriples(os.path.join(predDir, f"{entityId}", f"{entityId}_top5.nt"))
    elif mode == "top10":
        return ReadTriples(os.path.join(predDir, f"{entityId}", f"{entityId}_top10.nt"))
    else:
        raise ValueError("mode must be one of 'dynamic','top5','top10'")


def GetEntityIdsFromGoldRoot(goldRoot: str) -> List[str]:
    """Return sorted list of entity directory names (numbers) present in goldRoot."""
    ids = []
    if not os.path.isdir(goldRoot):
        return ids
    for entry in os.listdir(goldRoot):
        p = os.path.join(goldRoot, entry)
        if os.path.isdir(p) and entry.isdigit():
            ids.append(entry)
    ids = sorted(ids, key=lambda x: int(x))
    return ids


# ------------------------------------
# Core: Compute metrics over all entities
# ------------------------------------
def ComputeMetricsForMode(goldRoot: str, predRoot: str, mode: str, verbose: bool = False):
    """
    goldRoot: path containing per-entity directories with gold files
    predRoot: path containing per-entity prediction directories
    mode: 'dynamic' | 'top5' | 'top10'
    """

    entityIds = GetEntityIdsFromGoldRoot(goldRoot)
    if len(entityIds) == 0:
        raise RuntimeError(f"No entity directories found in goldRoot: {goldRoot}")

    fmeasure = FMeasure()
    ndcg = NDCG()
    mapMetric = MAP()

    perEntityResults = []
    skipped = 0

    for eid in entityIds:
        goldDir = os.path.join(goldRoot, eid)
        predDir = os.path.join(predRoot, eid)
        if not os.path.isdir(predDir):
            # skip if no predictions for this entity
            skipped += 1
            if verbose:
                print(f"Skipping entity {eid}: prediction dir not found at {predDir}")
            continue

        # gather gold summaries (supports multiple annotators e.g. *_gold_top5_0.nt)
        goldSummaries = GatherGoldSummaries(goldDir, eid, mode)
        if len(goldSummaries) == 0:
            skipped += 1
            if verbose:
                print(f"Skipping entity {eid}: no gold summaries for mode '{mode}' in {goldDir}")
            continue

        # if mode dynamic, determine dynamicNum from gold_dynamic files:
        dynamicNum = None
        if mode == "dynamic":
            # Using the first gold_dynamic file's length as the target number
            # If multiple gold_dynamic files exist, we use the length of the first one
            # (you can change to max/median if desired)
            dynamicNum = len(goldSummaries[0])

        # get predicted triples for this entity (predicted triples are triple strings)
        predTriples = GetPredictedTriples(predRoot, eid, mode, dynamicNum)

        # If predicted is empty -> skip
        if len(predTriples) == 0:
            skipped += 1
            if verbose:
                print(f"Skipping entity {eid}: no predicted triples found for mode '{mode}'")
            continue

        # For FMeasure & MAP: they expect an ordered predicted list and list of gold lists
        fmScore = FMeasure.GetScore(predTriples, goldSummaries)
        mapScore = mapMetric.GetMap(predTriples, goldSummaries)

        # For NDCG: the predicted list must be triple identifiers (strings) that match gold keys
        ndcgScore = ndcg.GetScore(goldSummaries, predTriples)

        perEntityResults.append({
            "entity": eid,
            "numGold": len(goldSummaries[0]),
            "predCount": len(predTriples),
            "F1": fmScore,
            "NDCG": ndcgScore,
            "MAP": mapScore
        })

        if verbose:
            print(f"Entity {eid}: F1={fmScore:.4f} NDCG={ndcgScore:.4f} MAP={mapScore:.4f} (pred {len(predTriples)}, gold {len(goldSummaries[0])})")

    # aggregate
    if len(perEntityResults) == 0:
        raise RuntimeError("No entities processed successfully. Check directories and files.")

    avgF1 = float(np.mean([r["F1"] for r in perEntityResults]))
    avgNDCG = float(np.mean([r["NDCG"] for r in perEntityResults]))
    avgMAP = float(np.mean([r["MAP"] for r in perEntityResults]))

    # print summary
    print("--------------------------------------------------")
    print(f"Mode: {mode}")
    print(f"Entities processed: {len(perEntityResults)}  Skipped: {skipped}")
    print(f"Average F1 : {avgF1:.6f}")
    print(f"Average NDCG : {avgNDCG:.6f}")
    print(f"Average MAP : {avgMAP:.6f}")
    print("--------------------------------------------------")

    return perEntityResults, {"avgF1": avgF1, "avgNDCG": avgNDCG, "avgMAP": avgMAP}


# ------------------------------------
# CLI
# ------------------------------------
def ParseArguments():
    parser = argparse.ArgumentParser(description="Compute F1, NDCG, MAP for ESA outputs.")
    parser.add_argument("--goldRoot", default="/Users/praveen/Documents/Academics/Winter Sem 2025/KG SUMM/Week 07/ESA for WikiES/datasets/WikiES_benchmark/WikiCinema-SMALL-TEST", help="Root directory containing gold entity dirs (1,2,3..)")
    parser.add_argument("--predRoot", default="/Users/praveen/Documents/Academics/Winter Sem 2025/KG SUMM/Week 07/ESA for WikiES/model/Generated Output/WikiCinema", help="Root directory containing predictions (Generated Output/WikiCinema)")
    parser.add_argument("--mode", required=True, choices=["dynamic", "top5", "top10"], help="Which mode to evaluate")
    parser.add_argument("--verbose", action="store_true", help="Verbose per-entity printing")
    return parser.parse_args()


def Main():
    args = ParseArguments()
    perEntityResults, summary = ComputeMetricsForMode(args.goldRoot, args.predRoot, args.mode, args.verbose)


if __name__ == "__main__":
    Main()
