
import argparse
import torch
import os
from Core import env, dataset, training

MODEL_PATH = "models.pt"
import random
import numpy as np

def set_seed(seed):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)

def main():
    parser = argparse.ArgumentParser(description="Train Organism Models")
    parser.add_argument("--epochs", type=int, default=20, help="Training epochs (default: 20)")
    parser.add_argument("--steps", type=int, default=100, help="Trajectories")
    parser.add_argument("--seed", type=int, default=42, help="Random seed")
    args = parser.parse_args()

    set_seed(args.seed)

    print("Initializing World...")
    env.create_world()

    print(f"Generating Dataset ({args.steps} trajectories)...")
    data = dataset.generate_dataset(n_trajectories=args.steps)
    
    print(f"Training Models ({args.epochs} epochs)...")
    results = training.train_models(data, epochs=args.epochs, verbose=True)
    
    print("Saving Models and Config...")
    
    # helper to get state dict if possible
    def get_state(v):
        return v.state_dict() if hasattr(v, "state_dict") else v

    # Separate config
    config = results.pop("config")
    
    save_data = {
        "models": {k: get_state(v) for k, v in results.items()},
        "object_index": results["object_index"],
        "config": config
    }
    
    torch.save(save_data, MODEL_PATH)
    print(f"Done. Saved to {MODEL_PATH}")

if __name__ == "__main__":
    main()
