# Explicit Object Commitment in Non-Linguistic Agents:
## Object-Centric Planning in Deterministic Environments

Official code for the paper:  
*"Evaluating the Impact of Explicit Object Commitment in Neural Planning"*  
(Under Writing)

[![Paper](https://img.shields.io/badge/paper-Under%20Review-red)]()

## Abstract
This repository contains the official implementation for "Evaluating the Impact of Explicit Object Commitment in Neural Planning". The system implements a deterministic "Creature World" environment to isolate and compare two distinct cognitive architectures: a standard action-code planner (Variant A) and a strictly object-centric planner (Variant B).

## Scientific Goals
The primary objective is to validate whether forcing an agent to explicitly "commit" to an object *before* imagining consequences leads to more robust planning than implicit action generation. We control for model capacity by sharing the exact same predictive models (World Model, Effect Model) between both variants.

## Architectures

### Shared Foundation
Both variants utilize a frozen set of models trained on a heuristic dataset:
*   **World Model**: Predicts global state transitions $S_{t+1} = f(S_t, A)$.
*   **Effect Model**: Predicts specific attributes of objects affected by actions.
*   **Argument Networks**: Learns syntactic validity (e.g., "logs/sum.log" is a file).

### Experimental Variants
*   **Variant A (Baseline)**: Standard neural planning. It generates (Action, Object) pairs jointly and scores them based on global heuristics. Validates using a "soft" resolver.
*   **Variant B (Object-Centric - *Proposed*)**: Implements a strict cognitive pipeline:
    1.  **Commit**: Irreversibly select an object $O$.
    2.  **Imagine**: Predict consequences $S_{t+1}$ given $O$.
    3.  **Score**: evaluate using specific object-effect predictors.
    *   *Note*: This variant includes "loud failure" modes—it does not auto-correct invalid bindings, ensuring errors are diagnostic.

## Reproducibility
*   **Deterministic Execution**: All scripts accept a `--seed` argument.
*   **Configuration Saving**: Training artifacts (`models.pt`) include exact architecture configs and object indices.
*   **Statistical Evaluation**: Evaluation scripts run $N$ independent trials to report Mean $\pm$ Std Dev.

## Usage

### Prerequisites
*   Python 3.8+
*   PyTorch
*   **Windows Users**: Must install [VC++ Redistributable](https://aka.ms/vs/17/release/vc_redist.x64.exe).

### 1. Training
Train the foundational models. Dataset generation is deterministic and on-the-fly.
```bash
python run_train.py --epochs 100 --steps 100 --seed 42
```
*   `--epochs`: Recommended 75-150 for convergence on the "Generate Log" task.
*   `--steps`: Number of trajectories (recommended 100).

### 2. Evaluation
Compare the variants on specific research goals.
```bash
python run_eval.py --variant A --task log --runs 10
```

#### Evaluation Tasks
*   **Generate Log (`--task log`)**: 
    *   **Goal**: Ensure `logs/sum.log` is created.
    *   **Path**: Requires executing `RUN scripts/sum.py`.
    *   **Focus**: Baseline affordance detection and action selection.
*   **Write Summary (`--task summary`)**:
    *   **Goal**: Ensure `docs/summary_day1.txt` exists.
    *   **Path**: Requires `CREATE` (directory) followed by `WRITE` (specific file).
    *   **Focus**: Probing "Object Commitment" depth. Variant B must commit to the correct object/path across steps, whereas Variant A uses a privileged resolver.

#### Parameters
*   `--variant`: `A` (Guided Goal) or `B` (Object Centric).
*   `--runs`: Number of statistical samples (recommended 10+).
*   `--task`: Selection of the experimental goal.
*   `--steps`: Max steps allowed per run (default 10).
```bash
python run_eval.py --variant B --task summary --runs 10 --seed 100
```
Output will report success rates and average steps over 10 independent runs.

### 3. Automated Pipeline
Run the entire training and evaluation suite in one command. Results are automatically timestamped and saved to the `exp/` directory.
```bash
python run_pipeline.py --epochs 20 --runs 10 --seed 42
```
*   `--epochs`: Number of training epochs (default: 20).
*   `--runs`: Number of evaluation runs per variant/task (default: 10).
*   `--seed`: Random seed for reproducibility (default: 42).

## Repository Structure
*   `Core/`: Environment, Dataset, and Shared Model Definitions.
*   `Models/`: Planner implementations (VariantA.py, VariantB.py).
*   `run_train.py`: Main entry point for model training.
*   `run_eval.py`: Main entry point for statistical evaluation.
*   `run_pipeline.py`: Automation script for full train-eval cycles.

---

## License
MIT License