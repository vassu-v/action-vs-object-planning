import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
from .env import Action
from .planning_utils import encode_action, index_world_objects, compute_object_deltas, build_object_index, ACTION_TYPES, ACTION_TO_ID

DEVICE = "cuda" if torch.cuda.is_available() else "cpu"

# ---- Model Definitions ----

class WorldModel(nn.Module):
    def __init__(self, input_dim, output_dim):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(input_dim, 256),
            nn.ReLU(),
            nn.Linear(256, 256),
            nn.ReLU(),
            nn.Linear(256, output_dim)
        )
    def forward(self, x):
        return self.net(x)

class PolicyNet(nn.Module):
    def __init__(self):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(128, 128),
            nn.ReLU(),
            nn.Linear(128, len(ACTION_TYPES))
        )
    def forward(self, x):
        return self.net(x)

class FileArgNet(nn.Module):
    def __init__(self, max_files):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(128, 128),
            nn.ReLU(),
            nn.Linear(128, max_files)
        )
    def forward(self, x):
        return self.net(x)

class DirArgNet(nn.Module):
    def __init__(self, max_dirs):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(128, 64),
            nn.ReLU(),
            nn.Linear(64, max_dirs)
        )
    def forward(self, x):
        return self.net(x)

class ObjectEffectModel(nn.Module):
    # NOTE: This model predicts effects over the global object index space
    # (built from all dataset keys). In contrast, argument selection
    # operates over local snapshot indices. This asymmetry contributes
    # to the misalignment we study between planning and execution.
    def __init__(self, in_dim, out_dim):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(in_dim, 256),
            nn.ReLU(),
            nn.Linear(256, out_dim)
        )
    def forward(self, x):
        return self.net(x)


# ---- Data Builders (Private) ----

def _build_training_pairs(dataset):
    X, Y = [], []
    for sample in dataset:
        s = sample["state_before"]["global"]
        a = encode_action(sample["action"])
        s_next = sample["state_after"]["global"]
        X.append(np.concatenate([s, a]))
        Y.append(s_next)
    return np.stack(X), np.stack(Y)

def _build_policy_data(dataset):
    X, Y = [], []
    for sample in dataset:
        state = sample["state_before"]["global"]
        action = sample["action"]
        X.append(state)
        Y.append(ACTION_TO_ID[action.type])
    return np.stack(X), np.array(Y)

def _build_argument_data(dataset):
    file_X, file_Y = [], []
    dir_X, dir_Y = [], []

    for sample in dataset:
        state = sample["state_before"]["global"]
        action = sample["action"]
        snapshot = sample["state_before"]
        file_to_id, dir_to_id = index_world_objects(snapshot)

        if action.type in ["READ", "WRITE", "REPLACE", "RUN", "MOVE"]:
            if len(action.args) > 0:
                file_path = action.args[0]
                if file_path in file_to_id:
                    file_X.append(state)
                    file_Y.append(file_to_id[file_path])

        if action.type in ["CREATE", "MOVE"]:
            dir_path = None
            if action.type == "CREATE" and len(action.args) > 0:
                dir_path = action.args[0]
            elif action.type == "MOVE" and len(action.args) > 1:
                dir_path = action.args[1]
            if dir_path and dir_path in dir_to_id:
                dir_X.append(state)
                dir_Y.append(dir_to_id[dir_path])

    if not file_X:
        file_X = np.zeros((0, 128))
        file_Y = np.zeros((0,))
    else:
        file_X = np.stack(file_X)
        file_Y = np.array(file_Y)

    if not dir_X:
        dir_X = np.zeros((0, 128))
        dir_Y = np.zeros((0,))
    else:
        dir_X = np.stack(dir_X)
        dir_Y = np.array(dir_Y)

    return file_X, file_Y, dir_X, dir_Y

def _build_object_effect_data(dataset, object_index):
    X, Y = [], []
    for sample in dataset:
        s = sample["state_before"]["global"]
        a = encode_action(sample["action"])
        deltas = compute_object_deltas(sample["state_before"], sample["state_after"])
        y = np.zeros(len(object_index), dtype=np.float32)
        for k, v in deltas.items():
            if k in object_index:
                y[object_index[k]] = v
        X.append(np.concatenate([s, a]))
        Y.append(y)
    return np.stack(X), np.stack(Y)


# ---- Main Training Function ----

def train_models(dataset, epochs=20, verbose=True):
    """
    Trains all sub-models and returns them in a dictionary.
    No global state mutation.
    """
    
    # 1. World Model
    X, Y = _build_training_pairs(dataset)
    input_dim = X.shape[1]
    output_dim = Y.shape[1]
    
    world_model = WorldModel(input_dim, output_dim).to(DEVICE)
    optimizer = optim.Adam(world_model.parameters(), lr=1e-3)
    loss_fn = nn.MSELoss()
    
    Xt = torch.tensor(X, dtype=torch.float32).to(DEVICE)
    Yt = torch.tensor(Y, dtype=torch.float32).to(DEVICE)
    
    if verbose: print(f"Training WorldModel ({epochs} epochs)...")
    for _ in range(epochs):
        optimizer.zero_grad()
        pred = world_model(Xt)
        loss = loss_fn(pred, Yt)
        loss.backward()
        optimizer.step()
        
    # 2. Policy
    Xp, Yp = _build_policy_data(dataset)
    policy = PolicyNet().to(DEVICE)
    opt_p = optim.Adam(policy.parameters(), lr=1e-3)
    loss_p = nn.CrossEntropyLoss()
    
    Xp_t = torch.tensor(Xp, dtype=torch.float32).to(DEVICE)
    Yp_t = torch.tensor(Yp, dtype=torch.long).to(DEVICE)
    
    if verbose: print("Training PolicyNet...")
    for _ in range(epochs):
        opt_p.zero_grad()
        logits = policy(Xp_t)
        loss = loss_p(logits, Yp_t)
        loss.backward()
        opt_p.step()
        
    # 3. Arguments
    file_X, file_Y, dir_X, dir_Y = _build_argument_data(dataset)
    MAX_FILES = int(file_Y.max()) + 1 if len(file_Y) > 0 else 1
    MAX_DIRS = int(dir_Y.max()) + 1 if len(dir_Y) > 0 else 1
    
    file_net = FileArgNet(MAX_FILES).to(DEVICE)
    dir_net = DirArgNet(MAX_DIRS).to(DEVICE)
    
    opt_file = optim.Adam(file_net.parameters(), lr=1e-3)
    opt_dir = optim.Adam(dir_net.parameters(), lr=1e-3)
    
    if len(file_X) > 0:
        file_Xt = torch.tensor(file_X, dtype=torch.float32).to(DEVICE)
        file_Yt = torch.tensor(file_Y, dtype=torch.long).to(DEVICE)
        if verbose: print("Training FileArgNet...")
        for _ in range(epochs): # Train args fully
            opt_file.zero_grad()
            logits = file_net(file_Xt)
            loss = loss_p(logits, file_Yt)
            loss.backward()
            opt_file.step()
            
    if len(dir_X) > 0:
        dir_Xt = torch.tensor(dir_X, dtype=torch.float32).to(DEVICE)
        dir_Yt = torch.tensor(dir_Y, dtype=torch.long).to(DEVICE)
        if verbose: print("Training DirArgNet...")
        for _ in range(epochs):
            opt_dir.zero_grad()
            logits = dir_net(dir_Xt)
            loss = loss_p(logits, dir_Yt)
            loss.backward()
            opt_dir.step()

    # 4. Object Effects
    object_index = build_object_index(dataset)
    Xe, Ye = _build_object_effect_data(dataset, object_index)
    
    effect_model = ObjectEffectModel(Xe.shape[1], Ye.shape[1]).to(DEVICE)
    opt_e = optim.Adam(effect_model.parameters(), lr=1e-3)
    loss_e = nn.BCEWithLogitsLoss()
    
    Xe_t = torch.tensor(Xe, dtype=torch.float32).to(DEVICE)
    Ye_t = torch.tensor(Ye, dtype=torch.float32).to(DEVICE)
    
    if verbose: print("Training ObjectEffectModel...")
    for _ in range(epochs):
        opt_e.zero_grad()
        logits = effect_model(Xe_t)
        loss = loss_e(logits, Ye_t)
        loss.backward()
        opt_e.step()

    if verbose: print("All models trained.") # Fixed: don't start lines with space in python, though in string it is fine.
    
    return {
        "world_model": world_model,
        "policy": policy,
        "file_net": file_net,
        "dir_net": dir_net,
        "effect_model": effect_model,
        "object_index": object_index,
        "device": DEVICE,
        "config": {
            "input_dim": input_dim,
            "output_dim": output_dim,
            "max_files": MAX_FILES,
            "max_dirs": MAX_DIRS,
            "object_keys": list(object_index.keys()),
            # "seed": ... (seed is handled efficiently enough by torch/numpy global state if set, but explicit is better. 
            # We'll rely on the runner setting the seed)
        }
    }
