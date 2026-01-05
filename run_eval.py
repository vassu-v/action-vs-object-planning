
import argparse
import torch
import os
import sys
from Core import env, task, training
from Models import VariantA, VariantB

import random
import numpy as np

MODEL_PATH = "models.pt"
DEVICE = "cuda" if torch.cuda.is_available() else "cpu"

def set_seed(seed):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)

def load_models_from_bundle(path, device):
    if not os.path.exists(path):
        raise FileNotFoundError(f"Model file {path} not found. Run run_train.py first.")
        
    bundle = torch.load(path, map_location=device)
    config = bundle["config"]
    states = bundle["models"]
    
    # Reconstruct Object Index
    object_index = bundle["object_index"]
    
    # Reconstruct Models using Config
    world_model = training.WorldModel(config["input_dim"], config["output_dim"]).to(device)
    world_model.load_state_dict(states["world_model"])
    
    policy = training.PolicyNet().to(device)
    policy.load_state_dict(states["policy"])
    
    file_net = training.FileArgNet(config["max_files"]).to(device)
    file_net.load_state_dict(states["file_net"])
    
    dir_net = training.DirArgNet(config["max_dirs"]).to(device)
    dir_net.load_state_dict(states["dir_net"])
    
    # Infer effect model dims from config/index
    eff_in = config["input_dim"] # (global + action) same as WM
    eff_out = len(object_index)
    
    effect_model = training.ObjectEffectModel(eff_in, eff_out).to(device)
    effect_model.load_state_dict(states["effect_model"])

    return {
        "world_model": world_model,
        "policy": policy,
        "file_net": file_net,
        "dir_net": dir_net,
        "effect_model": effect_model,
        "object_index": object_index,
        "device": device
    }

def main():
    parser = argparse.ArgumentParser(description="Run Organism Evaluation")
    parser.add_argument("--variant", choices=["A", "B"], required=True)
    parser.add_argument("--task", choices=["log", "summary"], default="log", help="Task to evaluate")
    parser.add_argument("--runs", type=int, default=1, help="Number of runs to average")
    parser.add_argument("--steps", type=int, default=10, help="Max task steps")
    parser.add_argument("--seed", type=int, default=100, help="Base seed")
    args = parser.parse_args()

    set_seed(args.seed)

    print(f"Loading Models from {MODEL_PATH}...")
    models = load_models_from_bundle(MODEL_PATH, DEVICE)
    
    assert models["world_model"] is not None
    assert models["effect_model"] is not None
    # Both variants share identical predictive models.
    
    target_task = task.TASK_LOG if args.task == "log" else task.TASK_SUMMARY
    
    print(f"\nRunning Task: {target_task['name']}")
    print(f"Variant: {args.variant}")
    print(f"Base Seed: {args.seed} | Runs: {args.runs}")
    print("-" * 30)

    planner_fn = None
    binder_fn = None
    
    if args.variant == "A":
        planner_fn = VariantA.plan_one_step
        binder_fn = None 
    elif args.variant == "B":
        planner_fn = VariantB.plan_one_step
        binder_fn = VariantB.bind_object_to_action

    success_count = 0
    total_steps = 0
    
    for i in range(args.runs):
        # vary seed slightly per run to test robustness
        run_seed = args.seed + i
        set_seed(run_seed)
        
        print(f"\n--- Run {i+1}/{args.runs} (Seed {run_seed}) ---")
        
        # Reset environment for each run
        # Reset environment for each run
        # task.reset_task_env() # Removed invalid call 
        # (Assuming task.py has reset_task_env or we rely on create_world per run? 
        # The task execution usually resets world relative to task. 
        # But 'run_task_with_trace' starts with snapshot_world. 
        # Ideally we'd reset the world state to clean slate.
        # But env.create_world() is destructive.
        # For pure eval, we might just run forward. 
        # But 'run_task_with_trace' calls 'reset_world' implicitly via task setup?
        # Let's check Core/task.py. It doesn't reset world.
        # We should reset world here.)
        # Reset environment for each run
        env.create_world() 
        # (create_world is now destructive, so no reset_world needed)

        result = task.run_task_with_trace(
            target_task, 
            models=models,
            planner_fn=planner_fn,
            binder_fn=binder_fn,
            max_steps=args.steps, 
            verbose=True # Debug: Enable verbosity
        )
        
        is_success = result["success"]
        steps = result["steps"]
        print(f"Result: {'SUCCESS' if is_success else 'FAIL'} in {steps} steps")
        
        if is_success:
            success_count += 1
            total_steps += steps

    success_rate = (success_count / args.runs) * 100
    avg_steps = total_steps / success_count if success_count > 0 else 0
    
    print("=" * 30)
    print(f"EVALUATION COMPLETE ({args.runs} runs)")
    print(f"Variant: {args.variant}")
    print(f"Success Rate: {success_rate:.1f}%")
    print(f"Avg Steps (Successes): {avg_steps:.2f}")
    if args.runs > 1:
        # Simple Binomial Std Dev approximation
        std_dev = np.sqrt((success_rate/100 * (1 - success_rate/100)) / args.runs) * 100
        print(f"Std Dev: +/- {std_dev:.1f}%")
    print("=" * 30)

if __name__ == "__main__":
    main()
