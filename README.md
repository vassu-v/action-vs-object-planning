# Object Commitment as a Diagnostic Pressure Point in Grounded Planning

> **Paper Title:** Object Commitment as a Diagnostic Pressure Point in Grounded Planning  
> **Author:** Shoryavardhaan Gupta  
> **Date:** February 2026

[![DOI](https://zenodo.org/badge/DOI/10.5281/zenodo.18196407.svg)](https://doi.org/10.5281/zenodo.18196407)

## Abstract

Contemporary grounded planning agents commonly delegate object selection to external resolvers - APIs, heuristics, privileged interfaces, a pattern that masks representational failures. We isolate object commitment as a diagnostic variable: two agents with identical perception, world models, and training differ only in whether object arguments are resolved externally (**Variant A**) or internally (**Variant B**). This exposes when mean-pooled representations fail at categorical object grounding.

We demonstrate four non-intuitive dissociations:
1.  **The Archive Dichotomy:** Planning and grounding are orthogonal. The same model achieves **100%** multi-step planning success while failing completely (**0%**) at object selection on identical tasks when entropy increases from 4 to 30+ objects.
2.  **The Data Scale Paradox:** Homogeneous training data causes performance collapse from 100% (300 trajectories) to 0% (500+ trajectories) - more data actively harms performance through **Statistical Gravity**.
3.  **Capacity Scaling Dissociation:** Width-only scaling (8M parameters) destroys intelligence, while balanced width+depth scaling (75M parameters) recovers planning but *not* grounding.
4.  **Architectural Ceiling:** Grounding bottlenecks are architectural, not parametric. Even 220x scaling cannot overcome categorical failures under high entropy.

---

## Design Philosophy

This repository is an autopsy, not an optimization effort. The code is intentionally minimal to isolate representational effects from architectural confounding variables.

**Principles:**
*   **Deterministic Environment:** We remove stochasticity and partial observability to prove that failures are *representational*, not perceptual.
*   **Object-Agnostic Abstractions:** We use mean-pooling (standard in Model-Based RL) to specifically test where this abstraction breaks.
*   **Identical Controls:** Variant A and B share exact models, data, and seeds; only the *commitment timing* differs.

## Non-Goals

This work avoids common agent optimization targets to maintain diagnostic purity:
*   **NOT** about beating benchmarks (SOTA chasing).
*   **NOT** about learning object-centric representations (we intentionally use object-agnostic ones to measure the failure).
*   **NOT** about competing with LLMs (though the findings predict LLM hallucinations).

---

## Key Findings

### 1. The Archive Dichotomy (Planning vs. Grounding)
The same agent model, trained on "Mixed-300" data, shows completely orthogonal capabilities depending on environment entropy:

| Task | Environment | Variant B Success | Diagnosis |
|:-----|:------------|:------------------|:----------|
| **Archive Log** | Standard (4 objects) | **100%** | Perfect multi-step planning (RUN → MOVE). |
| **Archive Log** | Dense (30+ objects) | **0%** | **Grounding Wall.** Fails to distinguish specific logs from other files. |

**Implication:** Agents can be perfect planners but blind grounders. Benchmark success in low-entropy environments does not generalize to real-world density.

### 2. Statistical Gravity (The Data Scale Paradox)
Training on *more* homogeneous data actively degrades performance.
*   **100-300 Trajectories:** 100% Success (Viable Planning Window)
*   **500+ Trajectories:** 0% Success (Collapse)

**Mechanism:** As data volume increases, the agent overfits to statistically dominant patterns (e.g., "MOVE docs to tmp") rather than learning task-conditional logic.

### 3. Width-to-Depth Ratio (Topology Matters)
Parameter count alone does not determine capability.
*   **Base (343K, 2 layers):** 100% Planning Success
*   **Width-Only (8M, 2 layers):** 0% Planning Success (Complete Collapse)
*   **Width+Depth (75M, 6 layers):** 100% Planning Success (Recovery)

**Implication:** Intelligence emerges from the *ratio* of compositional depth to representational width.

---

## Repository Structure

This repository contains the complete experimental code to reproduce these findings.

```
expv2pushpublish/
├── Core/
│   ├── env.py              # Environment & filesystem primitives
│   ├── planning_utils.py   # Action encoding & object indexing
│   ├── task.py             # Task definitions & Variant A resolver (Delegated Grounding)
│   └── ...
├── Models/
│   ├── VariantA.py         # Privileged/Heuristic Grounding (Control)
│   ├── VariantB.py         # Internal/Learned Grounding (Diagnostic)
│    └── ...
├── run_train.py            # Training script
├── run_benchmark.py        # Full benchmark suite
├── run_eval.py             # Single task evaluation trace
└── ...
```

### Components
*   **Variant A (Delegated Grounding):** Selects Action Type internally; Object Arguments resolved via `Core/task.py` heuristics. Represents current tool-using agents.
*   **Variant B (Internal Commitment):** Must select Action Type AND Object Arguments via `Models/VariantB.py` neural networks. Exposes the mean-pooling bottleneck.

---

## Installation

Requirements: Python 3.8+, PyTorch 1.12+, NumPy.

```bash
# Clone repository
git clone https://github.com/vassu-v/action-vs-object-planning.git
cd action-vs-object-planning/expv2pushpublish

# Install dependencies
pip install torch numpy
```

---

## Reproduction

### 1. Train the "Mixed-300" Model (Primary Configuration)
This model is trained on 100 standard trajectories and 200 dense trajectories.

```bash
python run_train.py --n_standard 100 --n_dense 200 --epochs 700 --seed 42
```

### 2. Reproduce the "Archive Dichotomy"
Verify that the SAME model succeeds in Standard but fails in Dense.

**Standard Environment (Success):**
```bash
python run_eval.py --variant B --task archive --env standard --runs 5
# Expected: Success Rate: 100.0%
```

**Dense Environment (Failure):**
```bash
python run_eval.py --variant B --task archive --env dense --runs 5
# Expected: Success Rate: 0.0%
```

### 3. Reproduce "Statistical Gravity" (Data Scale Paradox)
Train on homogeneous data to see the performance collapse.

**Viable Window (300 Trajectories):**
```bash
python run_train.py --n_standard 300 --n_dense 0 --epochs 700 --seed 42
python run_eval.py --variant B --task log --env standard --runs 10
# Expected: ~100% Success
```

**Collapse (500 Trajectories):**
```bash
python run_train.py --n_standard 500 --n_dense 0 --epochs 700 --seed 42
python run_eval.py --variant B --task log --env standard --runs 10
# Expected: ~0% Success
```

### 4. Reproduce Capacity Scaling (Width vs. Depth)
To reproduce the capacity scaling results, modification of `Core/training.py` is required.

**Width-Only Scaling (8M Parameters):**
Manually edit `Core/training.py` to increase the hidden layer width from 256 to 1600 in all model definitions:
```python
# Change all instances of 256 -> 1600:
nn.Linear(input_dim, 1600), nn.ReLU(),
nn.Linear(1600, 1600), nn.ReLU(),
nn.Linear(1600, output_dim)
```
Then train and evaluate:
```bash
python run_train.py --epochs 700 --n_standard 300 --n_dense 0 --seed 42
python run_eval.py --variant B --task archive --env standard --runs 5
# Expected: 0% Success (Width-only collapse)
```

**Balanced Scaling (75M Parameters):**
In addition to the width change, increase depth by adding 4 more hidden layers to each model class in `Core/training.py`.
```python
# Code structure should look like this (6 layers x 1600 units):
nn.Linear(input_dim, 1600), nn.ReLU(),
nn.Linear(1600, 1600), nn.ReLU(),
nn.Linear(1600, 1600), nn.ReLU(),
...,
nn.Linear(1600, output_dim)
```
Expected result: **100% Success** (Planning recovered).

**Restore Original Code:**
```bash
git checkout Core/training.py
```

---

## Citation

```bibtex
@article{gupta2026object,
  title={Object Commitment as a Diagnostic Pressure Point in Grounded Planning},
  author={Gupta, Shoryavardhaan},
  journal={Zenodo},
  year={2026}
}
```
