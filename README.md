# Experiment C: Post-Hoc Online Adaptation

> **Extension of:** [expv2pushpublish](../expv2pushpublish/) — the base experiment  
> **Paper:** Object Commitment as a Diagnostic Pressure Point in Grounded Planning  
> **Author:** Shoryavardhaan Gupta

---

## Purpose

This directory extends the [base experiment](../expv2pushpublish/) with a post-hoc verification: **Can online learning rescue the Grounding Wall?**

The base experiment (`expv2pushpublish`) established that mean-pooled representations fail at categorical object grounding in dense environments (0% success), even while maintaining perfect planning in standard environments (100% success). This extension tests whether that finding holds when the agent is given every advantage — Selective Plasticity, Curiosity-Driven Exploration, and Hindsight Experience Replay.

**Result:** The finding holds. The Grounding Wall is architectural, not distributional.

---

## What This Extension Adds

All shared infrastructure (`Core/env.py`, `Core/task.py`, `Models/`, `run_train.py`, `run_eval.py`, `run_benchmark.py`) is inherited from the base experiment. This extension adds only:

| File | Purpose |
|:-----|:--------|
| `Core/online.py` | Selective Plasticity (frozen hidden layers, trainable I/O) |
| `Core/curiosity.py` | Intrinsic reward to break statistical gravity |
| `Core/hindsight.py` | Hindsight Experience Replay — relabel failed trajectories |
| `Core/replay.py` | Snapshot-aware replay buffer |
| `run_full_adaptation.py` | Online adaptation harness (ε-greedy + HER + Curiosity) |

---

## Key Pathologies Discovered

| Pathology | Trigger | Confirms Base Finding? |
|:---|:---|:---|
| **Brain Rot** | Full plasticity on failed traces | ✅ Destroys even Variant A (oracle) performance |
| **Statistical Gravity** | Homogeneous replay data | ✅ Agent locks into dominant patterns |
| **The Recursive Trap** | Low curiosity | ✅ Agent only learns from what it already knows |

---

## Reproduction

**Prerequisite:** The base experiment models must be trained first. See [expv2pushpublish/README.md](../expv2pushpublish/README.md) for training instructions.

```bash
# Verify Grounding Wall exists (from base experiment)
python run_eval.py --variant B --task archive --env dense --runs 5
# Expected: 0%

# Attempt online rescue with full machinery
python run_full_adaptation.py --variant B --env dense --online --curiosity --steps 60 --epsilon 0.6 --her_ratio 0.5
# Expected: Still 0% — Grounding Wall holds

# Observe curiosity singularity  
python run_full_adaptation.py --variant B --env dense --online --curiosity --steps 60
# Watch: nov scores climb but success remains 0%
```

See [ONLINE_LEARNING_GUIDE.md](ONLINE_LEARNING_GUIDE.md) for the full protocol.

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
