# Entity Summarization: Comparative Analysis & Ontology-Guided Improvement

[![License: MIT](https://img.shields.io/badge/License-MIT-yellow.svg)](https://opensource.org/licenses/MIT)
[![Python](https://img.shields.io/badge/Python-3.8%2B-blue)](https://www.python.org/)
[![PyTorch](https://img.shields.io/badge/PyTorch-GNN-red)](https://pytorch.org/)

This repository contains the complete codebase and reports for a multi-stage research project on **Entity Summarization (ES)** in Knowledge Graphs. The project spans from comparative benchmarking of state-of-the-art models to the development of **H-IRES**, a novel ontology-aware unsupervised summarizer.

---

## 📚 Project Overview

The research is divided into three key phases:

| Phase | Task | Focus | Key Outcome |
| :--- | :--- | :--- | :--- |
| **I** | **Comparative Analysis** | IRES (Unsupervised) vs. ESLM (Supervised) | ESLM outperforms in quality; IRES offers flexibility but suffers from "generic" centrality bias. |
| **II** | **WikiES Evaluation** | Benchmarking IRES on WikiES-SMALL (JSON) | Established structural baselines for modern JSON-based Knowledge Graphs. |
| **III** | **Novel Contribution** | **H-IRES (Hierarchical IRES)** | Developed an **Ontology-Guided GNN** that solves the "generic bias" of IRES, achieving state-of-the-art ranking (NDCG > 0.93). |

---

## 🧠 Phase I & II: Analysis & Benchmarking

### Task 1 & 2: Supervised vs. Unsupervised
I conducted a rigorous comparison between **IRES** (Iterative Relationship Scoring) and **ESLM** (Entity Summarization with Language Models) on standard datasets (DBpedia, FACES, LinkedMDB).

* **Findings:**
    * **ESLM** excels by leveraging contextual language models (T5), mimicking human preference for descriptive facts (e.g., professions, key works).
    * **IRES** effectively identifies structural hubs but struggles with "semantic noise"—often selecting administrative metadata (e.g., *image_size*) simply because it is central in the graph.
    * **Conclusion:** Unsupervised models need explicit semantic signals to compete with supervised approaches.

### Task 3: WikiES-SMALL Evaluation
I adapted the IRES model to process the **WikiES** dataset, moving away from legacy XML to modern JSON graph structures.
* **Result:** Achieved **F-Measure: 0.375** and **MAP: 0.277**, establishing a robust baseline for unsupervised summarization on this dataset.

---

## 🚀 Phase III: H-IRES (The Novel Contribution)

To address the limitations identified in Phase I, I developed **H-IRES (Hierarchical IRES)**. This model injects **Ontological Semantics** into the unsupervised learning process, forcing the GNN to prioritize information-rich entities.

### Key Innovations
1.  **Ontology Fusion:** Integrates **Information Content (IC)** and **Hierarchical Depth** into node embeddings.
    * *Effect:* The model learns that rare classes (e.g., *Astronaut*) are more important than common ones (e.g., *Person*).
2.  **Adaptive Scalability:** A dual-mode GNN architecture that handles massive graphs like **WIKIES** (33k+ nodes) by switching to lightweight Basis Decomposition when relation types > 1,000.
3.  **Triple-Aware Scoring:** A custom metric pipeline that handles URI normalization and ID-to-Label translation automatically.

### Performance Results (H-IRES)
H-IRES achieved exceptional ranking performance, validating the ontology-guided approach.

<img width="542" height="317" alt="image" src="https://github.com/user-attachments/assets/d221631f-7251-4a88-b9f9-6a0368187338" />


> **Highlight:** The near-perfect NDCG (>0.93) on FACES and DBpedia confirms that Ontology-Guided summarization successfully mimics human ranking preferences without requiring training labels.

---
