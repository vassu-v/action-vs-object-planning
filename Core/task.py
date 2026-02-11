import numpy as np
from .env import Action, apply_action, create_world
from .dataset import snapshot_world

GOAL_LOGS = {
    "logs": 1.0,
    "docs": 0.0,
    "maint": 0.0,
    "deploy": 0.0,
    "config": 0.0
}

GOAL_DOCS = {
    "logs": 0.0,
    "docs": 1.0,
    "maint": 0.0, 
    "deploy": 0.0,
    "config": 0.0
}

GOAL_MAINT = {
    "logs": 0.0,
    "docs": 0.0,
    "maint": 1.0,
    "deploy": 0.0,
    "config": 0.0,
    "refactor": 0.0,
    "archive": 0.0,
    "test": 0.0
}

GOAL_TEST = {
    "logs": 0.0,
    "docs": 0.0,
    "maint": 1.0,
    "deploy": 0.0,
    "config": 0.0,
    "refactor": 0.0,
    "archive": 0.0,
    "test": 1.0
}

GOAL_REFACTOR = {
    "logs": 0.0,
    "docs": 0.0,
    "maint": 1.0,
    "deploy": 0.0,
    "config": 0.0,
    "refactor": 1.0,
    "archive": 0.0
}

GOAL_ARCHIVE = {
    "logs": 0.0,
    "docs": 0.0,
    "maint": 1.0,
    "deploy": 0.0,
    "config": 0.0,
    "refactor": 0.0,
    "archive": 1.0
}

GOAL_DEPLOY = {
    "logs": 0.0,
    "docs": 0.0,
    "maint": 0.0,
    "deploy": 1.0,
    "config": 0.0
}

GOAL_CONFIG = {
    "logs": 0.0,
    "docs": 0.0,
    "maint": 0.0,
    "deploy": 1.0,
    "config": 1.0
}

GOAL_QUIET = {
    "logs": -1.0,
    "docs": 0.0,
    "maint": 0.0,
    "deploy": 0.0
}


def resolve_action_args(action_type, snapshot, goal=None):
    """
    Minimal argument resolver for task execution.
    Uses simple heuristics (not learning yet).
    This implies 'Variant A' style explicit action selection.

    # NOTE: This resolver is intentionally privileged.
    # It implements the "Delegated Grounding" condition described in the paper.
    # By resolving object arguments externally, Variant A isolates the planning
    # capability from the grounding capability.
    """
    if goal is None:
        goal = {}

    if action_type == "RUN":
        if goal.get("maint", 0.0) > 0.5:
            # Explicit: RUN_TESTS task
            return ("tests/test_main.py",)
        return ("scripts/sum.py",)

    elif action_type == "READ":
        # Add explicit handling for config/archive tasks
        if goal.get("deploy", 0.0) > 0.5:
            # Config task might need to read settings first
            return ("config/settings.yaml",)
        if goal.get("maint", 0.0) > 0.5:
            # Archive task might read logs
            return ("logs/error.log",)
        return ("docs/notes.txt",)

    elif action_type == "WRITE":
        if goal.get("docs", 0.0) > 0.5:
             return ("docs/summary_day1.txt", "\nSummary ready.")
        return ("docs/notes.txt", "\nupdate")

    elif action_type == "REPLACE":
        if goal.get("deploy", 0.0) > 0.5:
            return ("config/settings.yaml", "version: 2.0\n")
        return ("docs/notes.txt", "replaced content\n")

    elif action_type == "CREATE":
        if goal.get("docs", 0.0) > 0.5:
             # Privileged target for Write Summary task
             return ("docs", "summary_day1.txt", "SUMMARY START\n")
        return ("docs", f"task_{np.random.randint(1000)}.txt", "task output\n")

    elif action_type == "MOVE":
        # Maintenance: Refactor, Archive, or Test
        if goal.get("maint", 0.0) > 0.5:
            # Check specific sub-task
            if goal.get("refactor", 0.0) > 0.5:
                # Refactor: move source files to tests
                for src in ["src/main.py", "src/utils.py", "scripts/sum.py"]:
                    if f"file:{src}" in snapshot["objects"]:
                        return (src, "tests")
            elif goal.get("archive", 0.0) > 0.5:
                # Archive: move logs to tmp
                for log in ["logs/error.log", "logs/server.log", "logs/access.log", "logs/sum.log"]:
                    if f"file:{log}" in snapshot["objects"]:
                        return (log, "tmp")
            # Fallback for generic maint tasks
            return ("docs/notes.txt", "tmp")
        
        # Deployment tasks
        if goal.get("deploy", 0.0) > 0.5:
            # Config task (Specific sub-goal)
            if goal.get("config", 0.0) > 0.5:
                if "file:config/settings.yaml" in snapshot["objects"]:
                    return ("config/settings.yaml", "tmp")
            
            # Deploy task: move bin to web
            for b in ["bin/app.exe", "bin/script.sh"]:
                if f"file:{b}" in snapshot["objects"]:
                    return (b, "web")
            
            # Fallback for config if bin is gone
            if "file:config/settings.yaml" in snapshot["objects"]:
                return ("config/settings.yaml", "tmp")
            
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
            args = binder_fn(action_type, obj_pair, snapshot, models, task["goal"])
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
            act_str = str(action_type) if action_type else "None"
            print(f"[{step}] ACTION = {act_str:6s} {str(args):20s} | {status} | score = {score:.3f}")

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

TASK_REFACTOR = {
    "name": "Refactor Code",
    "goal": GOAL_REFACTOR,
    # Move src/main.py -> tests/main.py
    "success": lambda snap: "file:tests/main.py" in snap["objects"]
}

TASK_DEPLOY = {
    "name": "Deploy App",
    "goal": GOAL_DEPLOY,
    # Move bin/app.exe -> web/app.exe
    "success": lambda snap: "file:web/app.exe" in snap["objects"]
}


TASK_BACKUP_CONFIG = {
    "name": "Backup Config", 
    "goal": GOAL_CONFIG,
    # Move config to tmp (achievable)
    "success": lambda snap: "file:tmp/settings.yaml" in snap["objects"]
}



TASK_RUN_TESTS = {
    "name": "Run Tests",
    "goal": GOAL_TEST,
    "success": lambda snap: "file:logs/test_main.log" in snap["objects"]
}

TASK_ARCHIVE_LOGS = {
    "name": "Archive Logs",
    "goal": GOAL_ARCHIVE,
    # Success if any .log file is moved to tmp
    "success": lambda snap: any(k.startswith("file:tmp/") and k.endswith(".log") for k in snap["objects"])
}

