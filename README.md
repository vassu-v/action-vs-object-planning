# Object Commitment as a Diagnostic Pressure Point in Grounded Planning

Code accompanying the paper:  
*"Object Commitment as a Diagnostic Pressure Point in Grounded Planning"*  
Shoryavardhaan Gupta (2026)

[![Paper](https://img.shields.io/badge/paper-PDF-blue)](link_to_your_pdf_or_arXiv_later)
[![Preprints.org](https://img.shields.io/badge/Preprints.org-Paper-red)](link_to_preprints)
(Under Review)

## Overview

This repository accompanies the paper *Object Commitment as a Diagnostic Pressure Point in Grounded Planning*.

We show that:
- Action-only agents achieve high success by externalizing object binding
- This externalization masks representational failures in global-latent world models
- Requiring internal object commitment exposes structured, repeatable failure modes
- Increasing training data exacerbates these failures rather than resolving them

## Purpose
This repository accompanies the paper "Object Commitment as a Diagnostic Pressure Point in Grounded Planning".

The code is intentionally minimal. It is not designed to produce a strong agent. Its purpose is diagnostic: to demonstrate that many modern agent benchmarks achieve high success by externalizing object binding, thereby masking representational failures in global-latent world models.

If you are looking for a performant agent, this is not the right repository.

## Design Philosophy
This repository is an autopsy, not an optimization effort. The design is intentionally simple to isolate representational effects.

**Design Principles:**
- Deterministic, non-linguistic environment (to isolate representation, not perception)
- Object-agnostic world and effect models (to test representational limits)
- Identical models and data across variants (to isolate object commitment)
- No architectural tricks or relational priors

## Variants
**Variant A (Delegated Grounding):**
- Plans over action types only
- Object arguments resolved externally by a fixed interface
- Mirrors common benchmark and tool-use agent designs

**Variant B (Internalized Commitment):**
- Must select both action and object internally
- Shares identical perception, models, and data with Variant A
- Exposes object-level failures invisible under delegated grounding

## Usage

### Minimal Repro Commands
```bash
# Train models
python run_train.py --epochs 600 --steps 100

# Evaluate Variant B
python run_eval.py --variant B --task log --runs 10
python run_eval.py --variant B --task summary --runs 10
```

### Automated Pipeline
Run the entire training and evaluation suite in one command. Results are automatically timestamped and saved to the `exp/` directory.
```bash
python run_pipeline.py --epochs 20 --runs 10 --seed 42
```
*   `--epochs`: Number of training epochs (default: 20).
*   `--runs`: Number of evaluation runs per variant/task (default: 10).
*   `--seed`: Random seed for reproducibility (default: 42).

The results in the paper (including the data scale paradox and optimization stability) can be reproduced using the logs in `exp/` or by running the pipeline with corresponding flags.

## Non-Goals
This is where we explicitly state what this work is NOT about.

**Non-goals:**
- Beating benchmarks
- Learning object-centric representations
- Demonstrating optimal planning
- Competing with Transformer-based agents

Recent additions include capacity- and data-scale experiments showing that increasing model depth and training data does not resolve identity collapse under object-agnostic abstractions.

## Citation

If you use this code or find this work useful, please cite:

```bibtex
@article{gupta2026object,
  title={Object Commitment as a Diagnostic Pressure Point in Grounded Planning},
  author={Gupta, Shoryavardhaan},
  journal={...},  <!-- Update when accepted -->
  year={2026}
}
```
