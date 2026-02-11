
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
    
    policy = training.PolicyNet(config["output_dim"]).to(device)
    policy.load_state_dict(states["policy"])
    
    file_net = training.FileArgNet(config["output_dim"], config["max_files"]).to(device)
    file_net.load_state_dict(states["file_net"])
    
    dir_net = training.DirArgNet(config["output_dim"], config["max_dirs"]).to(device)
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
    # Map CLI task names to Task Definitions
    TASK_MAP = {
        "log": task.TASK_LOG,
        "summary": task.TASK_SUMMARY,
        "refactor": task.TASK_REFACTOR,
        "archive": task.TASK_ARCHIVE_LOGS,
        "deploy": task.TASK_DEPLOY,
        "config": task.TASK_BACKUP_CONFIG,
        "test": task.TASK_RUN_TESTS
    }
    
    parser = argparse.ArgumentParser(description="Run Organism Evaluation")
    parser.add_argument("--variant", choices=["A", "B"], required=True)
    parser.add_argument("--task", choices=list(TASK_MAP.keys()), default="log", help="Task to evaluate")
    parser.add_argument("--runs", type=int, default=1, help="Number of runs to average")
    parser.add_argument("--steps", type=int, default=10, help="Max task steps")
    parser.add_argument("--seed", type=int, default=100, help="Base seed")
    parser.add_argument("--env", choices=["standard", "dense", "mixed"], default="mixed", help="Environment config")

    args = parser.parse_args()

    set_seed(args.seed)

    print(f"Loading Models from {MODEL_PATH}...")
    models = load_models_from_bundle(MODEL_PATH, DEVICE)
    
    assert models["world_model"] is not None
    assert models["effect_model"] is not None
    
    target_task = TASK_MAP[args.task]
    
    print(f"\nRunning Task: {target_task['name']}")
    print(f"Variant: {args.variant}")
    print(f"Base Seed: {args.seed} | Runs: {args.runs} | Env: {args.env}")
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
        
        # Determine Env Config
        if args.env == "mixed":
            cfg = "dense" if random.random() > 0.5 else "standard"
        else:
            cfg = args.env

        env.create_world(config=cfg) 

        result = task.run_task_with_trace(
            target_task, 
            models=models,
            planner_fn=planner_fn,
            binder_fn=binder_fn,
            max_steps=args.steps, 
            verbose=True 
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
