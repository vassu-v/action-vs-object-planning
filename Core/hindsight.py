"""
Hindsight Experience Replay (HER) for Grounding Recovery.

Core Insight: A trajectory that fails one task may succeed at another.
By relabeling failed trajectories with achieved goals, we generate
gold-signal samples without requiring actual task success.
"""

def get_achieved_goal(final_snapshot, task_map):
    """
    Check final state against all task success conditions.
    Returns list of (task_key, task_goal) that were achieved.
    """
    achieved = []
    for task_key, task_def in task_map.items():
        # Check if the task's success condition is met in the final snapshot
        if task_def["success"](final_snapshot):
            achieved.append((task_key, task_def["goal"]))
    return achieved


def relabel_trajectory(trace, achieved_goal):
    """
    Relabels a trajectory with a new goal.
    
    Args:
        trace: Original execution trace from run_task_with_trace
        achieved_goal: Goal dict of the task that was actually achieved
    
    Returns:
        Relabeled trace with new goal injected into each step.
    """
    relabeled = []
    for step in trace:
        new_step = step.copy()
        new_step["hindsight_goal"] = achieved_goal
        new_step["is_hindsight"] = True
        relabeled.append(new_step)
    return relabeled


def extract_hindsight_samples(result, task_map, original_task_key):
    """
    Master function: given a failed result, find all achievable relabelings.
    
    Args:
        result: Output of run_task_with_trace
        task_map: All task definitions
        original_task_key: The task that was attempted (to exclude self)
    
    Returns:
        List of (relabeled_trace, matched_task_key, matched_goal)
    """
    if result["success"]:
        return []  # No hindsight needed for successes
    
    final_snapshot = result["final_snapshot"]
    trace = result["trace"]
    
    if not trace:
        return []
    
    # Find what was actually achieved
    achieved = get_achieved_goal(final_snapshot, task_map)
    
    hindsight_samples = []
    for task_key, goal in achieved:
        if task_key == original_task_key:
            continue  # Skip self
        
        relabeled = relabel_trajectory(trace, goal)
        hindsight_samples.append((relabeled, task_key, goal))
    
    # PARTIAL HINDSIGHT: Even if no task fully succeeded,
    # check intermediate states for task achievements.
    if not hindsight_samples:
        hindsight_samples = _extract_partial_hindsight(trace, task_map, original_task_key)
    
    return hindsight_samples


def _extract_partial_hindsight(trace, task_map, original_task_key):
    """
    Checks each intermediate state (not just final) for task achievements.
    """
    partial = []
    
    for i, step in enumerate(trace):
        snap_after = step.get("state_after")
        if not snap_after:
            continue
            
        for task_key, task_def in task_map.items():
            if task_key == original_task_key:
                continue
            try:
                if task_def["success"](snap_after):
                    print(f"   [Hindsight] MATCH: Step {i} achieved {task_key}!")
                    partial_trace = trace[:i+1]
                    relabeled = relabel_trajectory(partial_trace, task_def["goal"])
                    partial.append((relabeled, task_key, task_def["goal"]))
                    break
            except Exception:
                continue
    
    return partial
