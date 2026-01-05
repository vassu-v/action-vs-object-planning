import numpy as np
from .env import Action

# ---- Action vocabulary (LOCKED) ----
ACTION_TYPES = ["READ", "WRITE", "REPLACE", "CREATE", "MOVE", "RUN"]
ACTION_TO_ID = {a: i for i, a in enumerate(ACTION_TYPES)}

def encode_action(action: Action):
    """
    Symbolic action encoding:
    - one-hot for action type
    - simple numeric features for args length
    """
    one_hot = np.zeros(len(ACTION_TYPES), dtype=np.float32)
    one_hot[ACTION_TO_ID[action.type]] = 1.0

    # minimal, cheap arg features
    arg_feats = np.array([
        len(action.args),
        sum(len(str(a)) for a in action.args)
    ], dtype=np.float32)

    return np.concatenate([one_hot, arg_feats])

def index_world_objects(snapshot):
    """
    Assign stable indices to files and directories in a snapshot.
    
    # NOTE: Object indices are local to the snapshot.
    # No cross-time object identity is preserved by design.
    # Argument selection operates over snapshot-local object slots,
    # approximating salience rather than persistent identity.
    """
    files = []
    dirs = []

    for k in snapshot["objects"].keys():
        if k.startswith("file:"):
            files.append(k.replace("file:", ""))
        elif k.startswith("dir:"):
            dirs.append(k.replace("dir:", ""))

    files = sorted(files)
    dirs = sorted(dirs)

    file_to_id = {f: i for i, f in enumerate(files)}
    dir_to_id = {d: i for i, d in enumerate(dirs)}

    return file_to_id, dir_to_id

def compute_object_deltas(before, after, eps=1e-3):
    """
    Returns a dict: object_key -> 0/1 indicating change.
    """
    deltas = {}

    before_objs = before["objects"]
    after_objs = after["objects"]

    all_keys = set(before_objs) | set(after_objs)

    for k in all_keys:
        if k not in before_objs:
            deltas[k] = 1  # appeared
        elif k not in after_objs:
            deltas[k] = 1  # disappeared
        else:
            diff = np.linalg.norm(after_objs[k] - before_objs[k])
            deltas[k] = int(diff > eps)

    return deltas

def build_object_index(dataset):
    keys = set()
    for sample in dataset:
        keys |= set(sample["state_before"]["objects"].keys())
        keys |= set(sample["state_after"]["objects"].keys())

    keys = sorted(keys)
    return {k: i for i, k in enumerate(keys)}
