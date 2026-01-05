import numpy as np
import torch
from Core.env import Action
from Core.planning_utils import encode_action, index_world_objects
from Core.inference import imagine_next, policy_priors, score_transition

def action_object_mask(action_type, object_index, snapshot):
    """
    Returns a binary mask over objects indicating which objects
    this action is allowed to affect, conditioned on existence.
    """
    mask = np.zeros(len(object_index), dtype=np.float32)
    existing = set(snapshot["objects"].keys())

    for obj, i in object_index.items():
        # General Rule: Objects must exist (except for CREATE target dirs)
        is_existent = obj in existing
        
        if action_type == "RUN":
            if is_existent and obj.startswith("file:scripts/"):
                mask[i] = 1.0

        elif action_type == "CREATE":
            # CREATE targets DIRECTORIES that must exist
            if is_existent and obj.startswith("dir:"):
                mask[i] = 1.0

        elif action_type == "WRITE":
            if is_existent and (obj.startswith("file:docs/") or obj.startswith("file:data/")):
                mask[i] = 1.0

        elif action_type == "MOVE":
            if is_existent and obj.startswith("file:"):
                mask[i] = 1.0

        elif action_type == "REPLACE":
            if is_existent and obj.startswith("file:"):
                mask[i] = 1.0

        elif action_type == "READ":
            pass

    return mask

def predict_object_effects(models, snapshot, action_type, top_k=5):
    """
    Predicts which objects will be affected by an action.
    """
    DEVICE = models["device"]
    effect_model = models["effect_model"]
    OBJECT_INDEX = models["object_index"]
    
    dummy = Action(action_type, ())
    a_enc = encode_action(dummy)
    x = np.concatenate([snapshot["global"], a_enc])

    x_t = torch.tensor(x, dtype=torch.float32).unsqueeze(0).to(DEVICE)

    with torch.no_grad():
        logits = effect_model(x_t)[0]
        probs = torch.sigmoid(logits).cpu().numpy()

    # apply existence-aware action mask
    mask = action_object_mask(action_type, OBJECT_INDEX, snapshot)
    probs = probs * mask

    # 🔑 TOP-K instead of threshold
    top_ids = probs.argsort()[::-1][:top_k]
    return {
        obj for obj, i in OBJECT_INDEX.items()
        if i in top_ids and probs[i] > 0
    }

def plan_one_step(models, snapshot, goal, top_k=6):
    """
    Variant A Planner: Guided Goal Planning.
    Directly selects action type, then relies on heuristic resolution.
    """
    # 1. Get candidate actions from Policy
    priors = policy_priors(models, snapshot, top_k=top_k)

    best_score = -1e9
    best_action = None

    # 2. Evaluate candidates using World Model + Effect Predictor
    for action_type, prior in priors:
        pred_global = imagine_next(models, snapshot["global"], action_type)

        def effect_fn_wrapper(snap, act_type):
             return predict_object_effects(models, snap, act_type)

        # Shared scoring function (Variant A passes None for committed_object)
        s = score_transition(
            snapshot, pred_global, None, action_type, goal, effect_fn_wrapper
        )

        # Policy prior biases but does not dominate
        s += 0.5 * np.log(prior + 1e-6)

        if s > best_score:
            best_score = s
            best_action = action_type

    return best_action, best_score

