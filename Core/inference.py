import numpy as np
import torch
from .env import Action
from .planning_utils import encode_action, ACTION_TYPES

# Stateless inference functions.
# All functions require 'models' dict (output of training.train_models) context.

def imagine_next(models, global_state, action_type):
    """
    Predict next global state embedding using the world model.
    """
    DEVICE = models["device"]
    world_model = models["world_model"]
    
    dummy_action = Action(action_type, ()) # args don't impact dummy encoding size, but we need consistency
    a_enc = encode_action(dummy_action)

    x = np.concatenate([global_state, a_enc])
    x_t = torch.tensor(x, dtype=torch.float32).unsqueeze(0).to(DEVICE)

    with torch.no_grad():
        pred_next = world_model(x_t)[0].cpu().numpy()

    return pred_next

def policy_priors(models, snapshot, top_k=3):
    DEVICE = models["device"]
    policy = models["policy"]
    
    state = torch.tensor(
        snapshot["global"], dtype=torch.float32
    ).unsqueeze(0).to(DEVICE)

    with torch.no_grad():
        logits = policy(state)[0]
        probs = torch.softmax(logits, dim=0).cpu().numpy()

    # top-k action indices
    top_ids = probs.argsort()[::-1][:top_k]
    return [(ACTION_TYPES[i], probs[i]) for i in top_ids]

def score_transition(snapshot, imagined_global, committed_object, action_type, goal, effect_fn):
    """
    Unified scoring function for both variants.
    
    Args:
        snapshot: Current state
        imagined_global: Predicted next global embeddings
        committed_object: (obj_type, obj_name) or None. 
                          If None (Variant A), object-specific effects are ignored.
                          If set (Variant B), we score effects specific to that object.
        action_type: The action being evaluated
        goal: Goal dictionary
        effect_fn: Injected function (snapshot, action_type) -> list[object_keys]
    """
    score = 0.0

    # 1. Latent Change (Universal)
    # Measures "did the world change in a meaningful way?"
    latent_delta = np.linalg.norm(imagined_global - snapshot["global"])
    score += 0.5 * latent_delta
    
    # 2. Object Specific Effects
    # We predict which objects are affected by this action type
    effects = effect_fn(snapshot, action_type)
    
    if committed_object is not None:
        # Variant B case: We have committed to an object.
        # We only reward if the committed object is actually affected.
        target_key = f"{committed_object[0]}:{committed_object[1]}"
        
        if target_key in effects:
             # Narrow check: only files satisfy scientific goals
             if target_key.startswith("file:"):
                 if "logs" in target_key:
                     score += 25.0 * goal.get("logs", 0.0)
                 if "docs" in target_key:
                     score += 15.0 * goal.get("docs", 0.0)
    else:
        # Variant A case: No commitment.
        # We reward ANY useful effect predicted by the model on FILES.
        for e in effects:
             if e.startswith("file:"):
                 if "logs" in e:
                     score += 25.0 * goal.get("logs", 0.0)
                 if "docs" in e:
                     score += 15.0 * goal.get("docs", 0.0)

    # 3. Variant A Goal Prior
    # Variant A has privileged resolution (hardcoded targets in task.py).
    # We add a strong bootstrap signal for RUN to prioritize it over noise.
    if committed_object is None:
        if action_type == "RUN":
            score += 10.0 * goal.get("logs", 0.0)

    # 4. Action Penalties (Shared)
    if action_type == "READ":
        score -= 1.0
    elif action_type == "CREATE":
        # Discourage infinite creation
        score -= 0.5

    return score
