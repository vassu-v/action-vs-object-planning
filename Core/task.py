import numpy as np
from .env import Action, apply_action, create_world
from .dataset import snapshot_world

GOAL_LOGS = {
    "logs": 1.0,
    "docs": 0.0,
}

GOAL_DOCS = {
    "logs": 0.0,
    "docs": 1.0,
}

GOAL_QUIET = {
    "logs": -1.0,
    "docs": 0.0,
}


def resolve_action_args(action_type, snapshot, goal=None):
    """
    Minimal argument resolver for task execution.
    Uses simple heuristics (not learning yet).
    This implies 'Variant A' style explicit action selection.

    # NOTE: This resolver is intentionally privileged.
    # It represents environments where object binding
    # is handled externally or deterministically.
    """
    if goal is None:
        goal = {}

    if action_type == "RUN":
        return ("scripts/sum.py",)

    elif action_type == "READ":
        return ("docs/notes.txt",)

    elif action_type == "WRITE":
        if goal.get("docs", 0.0) > 0.5:
             return ("docs/summary_day1.txt", "\nSummary ready.")
        return ("docs/notes.txt", "\nupdate")

    elif action_type == "REPLACE":
        return ("docs/notes.txt", "replaced content\n")

    elif action_type == "CREATE":
        if goal.get("docs", 0.0) > 0.5:
             # Privileged target for Write Summary task
             return ("docs", "summary_day1.txt", "SUMMARY START\n")
        return ("docs", f"task_{np.random.randint(1000)}.txt", "task output\n")

    elif action_type == "MOVE":
        return ("docs/notes.txt", "tmp")

    else:
        return ()


def run_task_with_trace(task, models, planner_fn, binder_fn=None, max_steps=5, verbose=True):
    """
    Executes a task using a pluggable planner.
    
    Args:
        task: Task definition dict.
        models: Dictionary of trained models.
        planner_fn: Function(models, snapshot, goal) -> (action_type, ... result)
        binder_fn: Optional Function(action_type, result, snapshot) -> args
                   Used by VariantB to bind object decisions to arguments.
    """
    # The environment is expected to be initialized by the caller (destructive create_world)
    trace = []

    snapshot = snapshot_world()

    for step in range(max_steps):
        # Check success BEFORE acting
        if task["success"](snapshot):
            return {
                "success": True,
                "steps": step,
                "trace": trace,
                "final_snapshot": snapshot
            }

        # 1. Plan
        # Planner returns whatever specific structure it uses
        plan_output = planner_fn(models, snapshot, task["goal"])
        
        # 2. Resolve Arguments
        if binder_fn:
            # Planner output expected: (action_type, obj_pair, score)
            action_type, obj_pair, score = plan_output
            args = binder_fn(action_type, obj_pair, snapshot)
        else:
            # Planner output expected: (action_type, score)
            action_type, score = plan_output
            args = resolve_action_args(action_type, snapshot, goal=task["goal"])
        
        action = Action(action_type, args)

        # 3. Act
        before = snapshot
        success = apply_action(action)
        snapshot = snapshot_world()

        # Record trace entry
        trace.append({
            "step": step,
            "action": action_type,
            "args": args,
            "success": success,
            "score": score,
            "objects_before": set(before["objects"].keys()),
            "objects_after": set(snapshot["objects"].keys())
        })

        if verbose:
            status = "OK" if success else "FAIL"
            print(f"[{step}] ACTION = {action_type:6s} {str(args):20s} | {status} | score = {score:.3f}")

    # Final success check
    return {
        "success": task["success"](snapshot),
        "steps": max_steps,
        "trace": trace,
        "final_snapshot": snapshot
    }

TASK_LOG = {
    "name": "Generate Log",
    "goal": GOAL_LOGS,
    "success": lambda snap: "file:logs/sum.log" in snap["objects"]
}

TASK_SUMMARY = {
    "name": "Write Summary",
    "goal": GOAL_DOCS,
    "success": lambda snap: "file:docs/summary_day1.txt" in snap["objects"]
}

