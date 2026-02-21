# Online Adaptation Protocol

> Extension of the [base experiment (main branch)](https://github.com/vassu-v/action-vs-object-planning/tree/main). Refer to the [main branch README](https://github.com/vassu-v/action-vs-object-planning/tree/main#readme) for core concepts (Variant A/B, Archive Dichotomy, Statistical Gravity).

---

## Overview

This guide documents the post-hoc adaptation protocol used to verify that the Grounding Wall finding from the base experiment is architectural and cannot be overcome through online learning.

**Base Finding (expv2pushpublish):** Variant B achieves 0% success in dense environments due to mean-pooling's inability to preserve categorical object identity.

**This Extension's Question:** Can we fix this at inference time with online adaptation?

**Answer:** No.

---

## Protocol

### Phase 1: Selective Plasticity
- **Mechanism**: Freeze hidden layers (`Net[2]`), update only Input (`Net[0]`) and Output (`Net[4]`) projections
- **Why**: Prevents Brain Rot — full plasticity destroys the base model's physics knowledge
- **Implementation**: `Core/online.py`

### Phase 2: Curiosity-Driven Exploration
- **Mechanism**: Novelty bonus = `||state - buffer_mean||`, blended with task score
- **Why**: Forces the agent to explore beyond statistically dominant patterns
- **Implementation**: `Core/curiosity.py`

### Phase 3: Hindsight Relabeling
- **Mechanism**: Failed trajectories are checked against all task goals; accidental successes are relabeled as training signal
- **Why**: Manufactures gold-signal samples even when the agent cannot succeed at its target task
- **Implementation**: `Core/hindsight.py`

---

## Reproduction Commands

```bash
# Step 1: Confirm base finding still holds
python run_eval.py --variant B --task archive --env dense --runs 5

# Step 2: Run full adaptation with all machinery enabled
python run_full_adaptation.py --variant B --env dense --online --curiosity --steps 60 --epsilon 0.6 --her_ratio 0.5

# Step 3: Observe curiosity singularity (novelty climbs, success stays 0%)
python run_full_adaptation.py --variant B --env dense --online --curiosity --steps 60
```

---

## Pathology Reference

| Pathology | Trigger | Effect |
|:---|:---|:---|
| **Brain Rot** | Full plasticity | Variant A drops to 0% — base model corrupted |
| **Statistical Gravity** | Homogeneous data | Agent locked into high-frequency patterns |
| **Recursive Trap** | Low exploration | Agent only learns from its own errors |
