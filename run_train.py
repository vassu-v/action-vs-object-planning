import os
import argparse
import random
import torch
import numpy as np
from Core import env, dataset, training

MODEL_PATH = "models.pt"
DEVICE = "cuda" if torch.cuda.is_available() else "cpu"

def set_seed(seed):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)

def main():
    parser = argparse.ArgumentParser(description="Train Models for Experiment C")
    parser.add_argument("--steps", type=int, default=None, help="Total trajectories (split 50/50)")
    parser.add_argument("--n_standard", type=int, default=250, help="Standard trajectories")
    parser.add_argument("--n_dense", type=int, default=250, help="Dense trajectories")
    parser.add_argument("--epochs", type=int, default=30, help="Training epochs")
    parser.add_argument("--seed", type=int, default=42, help="Random seed")
    args = parser.parse_args()

    # Handle legacy --steps if provided
    if args.steps is not None:
        args.n_standard = args.steps // 2
        args.n_dense = args.steps - args.n_standard

    set_seed(args.seed)

    print(f"Generating Dataset (Standard: {args.n_standard}, Dense: {args.n_dense})...")
    data = dataset.generate_dataset(n_standard=args.n_standard, n_dense=args.n_dense)
    
    # Capture object index from the generated data's latest world snapshot
    # This ensures models match the environment vocabulary
    latest_snapshot = dataset.snapshot_world()
    object_index = {obj: i for i, obj in enumerate(latest_snapshot["objects"].keys())}

    print(f"Training Models ({args.epochs} epochs) on {DEVICE}...")
    models = training.train_models(data, epochs=args.epochs)

    print("Saving Models and Config...")
    bundle = {
        "models": {k: v.state_dict() for k, v in models.items() if hasattr(v, "state_dict")},
        "config": models["config"],
        "object_index": models["object_index"]
    }
    torch.save(bundle, MODEL_PATH)
    print(f"Done. Saved to {MODEL_PATH}")

if __name__ == "__main__":
    main()
