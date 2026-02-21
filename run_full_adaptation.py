"""
Exploratory Online Adaptation Harness
---------------------------------------
Extension of the base experiment (expv2pushpublish). Tests whether 
Curiosity-Driven Exploration, Hindsight Experience Replay (HER), 
and Selective Plasticity can rescue the Grounding Wall finding.

See: ../expv2pushpublish/ for the base experiment and core findings.
"""

import argparse
import torch
import os
import random
import numpy as np
from Core import env, task, training
from Core.replay import ReplayBuffer
from Core.online import OnlineLearner
from Models import VariantA, VariantB
from Core.hindsight import extract_hindsight_samples
from Core.curiosity import CuriosityModule

MODEL_PATH = "models.pt"
DEVICE = "cuda" if torch.cuda.is_available() else "cpu"

def set_seed(seed):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)

def load_models_from_bundle(path, device):
    if not os.path.exists(path):
        raise FileNotFoundError(f"Model file {path} not found.")
    bundle = torch.load(path, map_location=device)
    config = bundle["config"]
    states = bundle["models"]
    object_index = bundle["object_index"]
    
    world_model = training.WorldModel(config["input_dim"], config["output_dim"]).to(device)
    world_model.load_state_dict(states["world_model"])
    policy = training.PolicyNet(config["output_dim"]).to(device)
    policy.load_state_dict(states["policy"])
    file_net = training.FileArgNet(config["output_dim"], config["max_files"]).to(device)
    file_net.load_state_dict(states["file_net"])
    dir_net = training.DirArgNet(config["output_dim"], config["max_dirs"]).to(device)
    dir_net.load_state_dict(states["dir_net"])
    eff_in = config["input_dim"]
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
        "device": device,
        "config": config # Persist for saving
    }

def save_models_to_bundle(models, path):
    bundle = {
        "models": {k: v.state_dict() for k, v in models.items() if hasattr(v, "state_dict")},
        "config": models["config"],
        "object_index": models["object_index"]
    }
    torch.save(bundle, path)
    print(f">> SUCCESS: Model weights persisted to {path}")

def explore_or_exploit(models, snapshot, goal, epsilon):
    """
    ε-greedy exploration logic: with probability ε, explores.
    - 50% of exploration: Keep ActionType, pick random Object (Grounding exploration)
    - 50% of exploration: Pick random ActionType + random Object (Policy exploration)
    """
    action_type, planned_obj, score = VariantB.plan_one_step(models, snapshot, goal)
    
    if random.random() < epsilon:
        # We are in exploration mode
        if random.random() < 0.5:
            # POLICY EXPLORATION: Try a new action entirely
            from Core.planning_utils import ACTION_TYPES
            action_type = random.choice(ACTION_TYPES)
        
        # GROUNDING EXPLORATION: Pick a totally random object
        valid_objects = [k for k in snapshot["objects"].keys()]
        if valid_objects:
            random_key = random.choice(valid_objects)
            if ":" in random_key:
                obj_type, obj_name = random_key.split(":", 1)
                return action_type, (obj_type, obj_name), score, True
    
    return action_type, planned_obj, score, False


def main():
    TASK_MAP = {
        "log": task.TASK_LOG,
        "summary": task.TASK_SUMMARY,
        "refactor": task.TASK_REFACTOR,
        "archive": task.TASK_ARCHIVE_LOGS,
        "deploy": task.TASK_DEPLOY,
        "config": task.TASK_BACKUP_CONFIG,
        "test": task.TASK_RUN_TESTS
    }
    
    parser = argparse.ArgumentParser(description="Exploratory Online Adaptation (Curiosity + HER)")
    parser.add_argument("--variant", choices=["A", "B"], default="B")
    parser.add_argument("--runs_per_task", type=int, default=5)
    parser.add_argument("--steps", type=int, default=10)
    parser.add_argument("--seed", type=int, default=100)
    parser.add_argument("--env", choices=["standard", "dense", "mixed"], default="dense")
    parser.add_argument("--online", action="store_true", help="Enable Online Learning")
    parser.add_argument("--save", action="store_true", help="Save updated weights to models.pt")
    
    # Hybrid Hyperparameters
    parser.add_argument("--epsilon", type=float, default=0.6, help="Initial exploration prob")
    parser.add_argument("--epsilon_decay", type=float, default=0.995, help="Decay per batch update")
    parser.add_argument("--min_epsilon", type=float, default=0.05)
    parser.add_argument("--her_ratio", type=float, default=0.5, help="HER sampling mix")
    
    parser.add_argument("--curiosity", action="store_true", help="Enable curiosity-driven exploration")
    parser.add_argument("--novelty_weight", type=float, default=0.3, help="Weight for novelty bonus")
    
    args = parser.parse_args()

    set_seed(args.seed)
    models = load_models_from_bundle(MODEL_PATH, DEVICE)
    
    # Initialize Persistent Learner & Curiosity
    learner = None
    replay_buffer = None
    curiosity = None
    
    if args.online:
        print(">> ONLINE LEARNING ENABLED: Selective Plasticity Active (Frozen Core)")
        learner = OnlineLearner(models, learning_rate=1e-4)
        replay_buffer = ReplayBuffer(capacity=5000)
        
        if args.curiosity:
            print(f">> CURIOSITY ENABLED: Novelty Weight = {args.novelty_weight}")
            curiosity = CuriosityModule(buffer_size=200, novelty_weight=args.novelty_weight)
    
    current_epsilon = args.epsilon
    
    # Logic for Variant A is slightly different but we focus on B for the grounding study
    binder_fn = None if args.variant == "A" else VariantB.bind_object_to_action

    global_stats = {t: {"success": 0, "total": 0} for t in TASK_MAP}
    
    # Curriculum Filtering (Still present for real signal)
    current_threshold = 0.2
    max_threshold = 0.7
    threshold_step = 0.02

    for task_key, target_task in TASK_MAP.items():
        print(f"\n>>> ADAPTING ON TASK: {target_task['name']}")
        print("-" * 30)
        
        for i in range(args.runs_per_task):
            run_seed = args.seed + i
            set_seed(run_seed)
            print(f"Run {i+1}/{args.runs_per_task} (Seed {run_seed}) [ε={current_epsilon:.2f}]")
            
            cfg = args.env
            if cfg == "mixed":
                cfg = "dense" if random.random() > 0.5 else "standard"
            
            env.create_world(config=cfg)
            
            # CUSTOM WRAPPER for Epsilon-Greedy Exploration
            def exploratory_planner(models, snapshot, goal, **kwargs):
                a, o, s, _ = explore_or_exploit(models, snapshot, goal, current_epsilon)
                return a, o, s
            
            # NOTE: VariantA/B usage normally happens in task.run_task_with_trace
            # but we need to pass a planner that respects epsilon
            result = task.run_task_with_trace(
                target_task, models=models, 
                planner_fn=exploratory_planner, binder_fn=binder_fn,
                max_steps=args.steps, verbose=True,
                curiosity=curiosity
            )
            
            is_success = result["success"]
            if is_success:
                global_stats[task_key]["success"] += 1

            # --- Online Learning Step with Hybrid Logic ---
            if args.online and replay_buffer is not None:
                if is_success:
                    print(f"Result: SUCCESS. Adding trace ({len(result['trace'])} steps)")
                    replay_buffer.add_trajectory(result["trace"])
                else:
                    # IRONY FIX: Capture partial logic signal
                    effective_threshold = min(current_threshold, 0.15) 
                    useful_steps = [t for t in result["trace"] if t["score"] > effective_threshold]
                    if useful_steps:
                        replay_buffer.add_trajectory(useful_steps)
                    
                    # HINDSIGHT: Capture accidental successes for other tasks
                    hindsight_samples = extract_hindsight_samples(result, TASK_MAP, task_key)
                    for relabeled_trace, m_task, m_goal in hindsight_samples:
                        print(f">> HINDSIGHT HIT: Trajectory achieved {m_task} (Relabeling)")
                        replay_buffer.add_trajectory(relabeled_trace, goal=m_goal, is_hindsight=True)

                
                # Training Step
                MIN_BATCH = 16
                if len(replay_buffer) >= MIN_BATCH:
                    loss = learner.train_step(replay_buffer, batch_size=MIN_BATCH)
                    print(f"[Learner] Update Complete. Loss: {loss:.4f} | Buffer: {replay_buffer.hindsight_stats()}")
                    
                    # Anneal Epsilon and Threshold
                    current_epsilon = max(args.min_epsilon, current_epsilon * args.epsilon_decay)
                    if current_threshold < max_threshold:
                        current_threshold += threshold_step
                else:
                    print(f"[Learner] Accumulating... ({len(replay_buffer)}/{MIN_BATCH})")
            # ----------------------------
            
            global_stats[task_key]["total"] += 1

    print("\n" + "=" * 50)
    print("HYBRID ADAPTATION SUMMARY")
    print("=" * 50)
    for t, stat in global_stats.items():
        rate = (stat["success"] / stat["total"]) * 100 if stat["total"] > 0 else 0
        print(f"{t:10s}: {rate:5.1f}% ({stat['success']}/{stat['total']})")
    print("=" * 50)

    if args.save:
        save_models_to_bundle(models, MODEL_PATH)

if __name__ == "__main__":
    main()
