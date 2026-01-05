import numpy as np
import torch
from Core.planning_utils import index_world_objects
from Core.inference import imagine_next, policy_priors, score_transition
from Models.VariantA import predict_object_effects, action_object_mask

def plan_one_step(models, snapshot, goal, top_actions=3, top_objects=3):
    """
    Variant B Planner: Object Centric.
    Selects (ActionType, Object) pair.

    # NOTE: Variant B uses an object-conditioned scoring function that 
    # decomposes global change and object-specific effects, while 
    # Variant A uses a unified transition score.
    """
    DEVICE = models["device"]
    priors = policy_priors(models, snapshot, top_k=top_actions)

    best = (-1e9, None, None)

    for action_type, prior in priors:
        # Step 1: COMMIT to an object (Irreversible)
        # We find top candidates using argument nets + syntactic mask
        objects = candidate_objects_for_action(
            models, action_type, snapshot, top_k=top_objects
        )

        if not objects:
            continue

        for obj_type, obj_name in objects:
            # Committed Pair
            committed_object = (obj_type, obj_name)
            
            # Step 2: IMAGINE consequence
            # NOTE: Imagination is intentionally object-agnostic.
            pred_global = imagine_next(models, snapshot["global"], action_type)
            
            # Step 3: SCORE (Shared Function)
            def effect_fn_wrapper(snap, act_type):
                 return predict_object_effects(models, snap, act_type)
            
            # We pass the committed object here. 
            score = score_transition(
                snapshot, pred_global, committed_object, action_type, goal, effect_fn_wrapper
            )
            
            # Add prior bonus
            score += 0.5 * np.log(prior + 1e-6)

            if score > best[0]:
                best = (score, action_type, committed_object)

    # Return structure: (action_type, args_obj, score)
    return best[1], best[2], best[0]


def candidate_objects_for_action(models, action_type, snapshot, top_k=3):
    """
    Returns candidate object IDs relevant for this action.
    Uses learned argument networks explicitly passed in models.
    """
    DEVICE = models["device"]
    file_net = models["file_net"]
    dir_net = models["dir_net"]
    
    file_to_id, dir_to_id = index_world_objects(snapshot)

    state = torch.tensor(
        snapshot["global"], dtype=torch.float32
    ).unsqueeze(0).to(DEVICE)

    candidates = []

    # helper to check syntactic validity (light mask)
    object_index = models["object_index"]
    mask = action_object_mask(action_type, object_index, snapshot)
    
    with torch.no_grad():
        if action_type in ["READ", "WRITE", "REPLACE", "RUN", "MOVE"] and file_to_id:
            logits = file_net(state)[0].cpu().numpy()
            top_ids = logits.argsort()[::-1][:top_k]
            for i in top_ids:
                for f, fid in file_to_id.items():
                    if fid == i:
                        # Mask check: Only allow syntactically valid objects
                        # The mask is by index in global OBJECT_INDEX.
                        global_idx = object_index.get(f"file:{f}")
                        if global_idx is not None and mask[global_idx] > 0:
                            candidates.append(("file", f))

        if action_type in ["CREATE", "MOVE"] and dir_to_id:
            logits = dir_net(state)[0].cpu().numpy()
            top_ids = logits.argsort()[::-1][:top_k]
            for i in top_ids:
                for d, did in dir_to_id.items():
                    if did == i:
                        # Similar mask check for dirs
                        global_idx = object_index.get(f"dir:{d}")
                        # Dirs are less strictly masked currently in VariantA (often just 0 or 1 dependent on specific logic)
                        # But action_object_mask logic for 'MOVE' etc. looks at prefixes 'file:'.
                        # So for now, we only mask if explicit dir logic exists, otherwise strict reliance.
                        # VariantA mask doesn't explicitly handle 'dir:' prefixes well yet.
                        # Let's trust the network for dirs but still allow valid files.
                        # Actually, looking at VariantA.action_object_mask:
                        # MOVE -> 'file:' 
                        # CREATE -> 'file:docs/' (wait, create takes dir arg?)
                        # Variant A resolver creates files.
                        # Variant B selects DIR to create IN.
                        # Conflict: VariantA masked targets of action. 
                        # VariantB selects arguments.
                        # If Action=CREATE, VariantB wants a DIR. VariantA mask allows 'file:docs'.
                        # This is a bit disjoint. We will skip mask for dirs to avoid blocking valid ops.
                        candidates.append(("dir", d))

    return candidates


def score_object_effects(action_type, obj, predicted_effects, goal):
    score = 0.0

    for e in predicted_effects:
        if obj in e:
            if e.startswith("file:logs"):
                score += 5.0 * goal.get("logs", 0.0)
            if e.startswith("file:docs"):
                score += 3.0 * goal.get("docs", 0.0)

    return score


def affordance_bonus(before_snap, after_pred_global):
    """
    Reward actions that significantly change the world.
    """
    delta = np.linalg.norm(after_pred_global - before_snap["global"])
    return 0.2 * delta

def bind_object_to_action(action_type, obj_pair, snapshot):
    """
    Explicitly binds a selected object to actual action arguments.
    Syntactic fallback for non-primary arguments.
    
    # NOTE: The executor performs a fixed syntactic mapping from selected 
    # objects to action arguments. It is intentionally simple/rigid.
    # NOTE: Invalid bindings are not corrected to preserve diagnostic failure.
    """
    if obj_pair is None: 
        return ()
        
    obj_type, obj_name = obj_pair
    
    # Strict Binding Logic
    
    if action_type == "RUN":
        # Can only run files
        if obj_type == "file":
             return (obj_name,)
        # If obj_type is dir, we do NOT fix it. We return invalid args.
        return (obj_name,) 

    elif action_type == "READ":
        if obj_type == "file":
            return (obj_name,)
        return (obj_name,)

    elif action_type == "WRITE":
        # WRITE(path, content)
        if obj_type == "file":
            return (obj_name, "\n[LOG ENTRY]\n")
        return (obj_name, "Invalid")

    elif action_type == "REPLACE":
        if obj_type == "file":
            return (obj_name, "REPLACED CONTENT")
        return (obj_name, "Invalid")
            
    elif action_type == "CREATE":
        # CREATE(dir, filename, content)
        if obj_type == "dir":
             return (obj_name, f"task_{np.random.randint(1000)}.txt", "task output\n")
        return (obj_name, "fail.txt", "fail")

    elif action_type == "MOVE":
        # NOTE: Intentional asymmetry in argument mapping
        # MOVE (src, dest)
        if obj_type == "dir":
            return ("docs/notes.txt", obj_name) 
        elif obj_type == "file":
            return (obj_name, "tmp") 
        
    return ()




