"""
Selective Plasticity Adaptation Engine
---------------------------------------
Restricts gradient updates to input/output layers while freezing the core 
hidden layers. Prevents Catastrophic Forgetting (Brain Rot) during online 
adaptation. Part of the post-hoc verification of expv2pushpublish findings.
"""

import torch
import torch.nn as nn
import torch.optim as optim
import numpy as np
from .training import _build_training_pairs, _build_policy_data, _build_argument_data, _build_object_effect_data, build_object_index

class OnlineLearner:
    def __init__(self, models, learning_rate=1e-5):
        """
        Initializes the Online Learner with Selective Plasticity.
        
        Args:
            models: Dictionary of trained models (shared with actor).
            learning_rate: LR for the fine-tuning updates.
        """
        self.models = models
        self.device = models["device"]
        self.lr = learning_rate
        
        # We need to reconstruct the optimizers, but strictly for the parameters we want to train.
        # "Selective Plasticity": Freeze the core (hidden layers), train I/O interfaces.
        
        self.optimizers = {}
        self.loss_fns = {
            "world_model": nn.MSELoss(),
            "policy": nn.CrossEntropyLoss(),
            "file_net": nn.CrossEntropyLoss(),
            "dir_net": nn.CrossEntropyLoss(),
            "effect_model": nn.BCEWithLogitsLoss()
        }
        
        self._setup_optimizers()

    def _setup_optimizers(self):
        """
        Configures optimizers for each model, enforcing the freezing mask.
        Assumes standard 3-layer MLP structure: [Linear, ReLU, Linear, ReLU, Linear]
        Index 0: Input Projection (Train)
        Index 2: Core Reasoning (Freeze)
        Index 4: Output Projection (Train)
        """
        
        # Helper to get trainable params
        def get_io_params(model):
            trainable = []
            # Freeze everything first
            for p in model.parameters():
                p.requires_grad = False
            
            # Unfreeze Input Layer (Net[0])
            for p in model.net[0].parameters():
                p.requires_grad = True
                trainable.append(p)
                
            # Unfreeze Output Layer (Net[4])
            if len(model.net) > 4:
                for p in model.net[4].parameters():
                    p.requires_grad = True
                    trainable.append(p)
            
            return trainable

        # 1. World Model
        wm_params = get_io_params(self.models["world_model"])
        self.optimizers["world_model"] = optim.Adam(wm_params, lr=self.lr)
        
        # 2. Policy
        pol_params = get_io_params(self.models["policy"])
        self.optimizers["policy"] = optim.Adam(pol_params, lr=self.lr)
        
        # 3. Argument Nets
        file_params = get_io_params(self.models["file_net"])
        self.optimizers["file_net"] = optim.Adam(file_params, lr=self.lr)
        
        dir_params = get_io_params(self.models["dir_net"])
        self.optimizers["dir_net"] = optim.Adam(dir_params, lr=self.lr)
        
        # 4. Effect Model
        eff_params = get_io_params(self.models["effect_model"])
        self.optimizers["effect_model"] = optim.Adam(eff_params, lr=self.lr)


    def train_step(self, replay_buffer, batch_size=32):
        """
        Performs a single gradient update on all models using a batch from the buffer.
        """
        if len(replay_buffer) < batch_size:
            return 0.0 # Not enough data
            
        # 1. Sample Batch
        batch = replay_buffer.sample(batch_size)
        
        # 2. Reconstruct mini-dataset for helpers
        mini_dataset = []
        for (snap_before, action, snap_after) in batch:
            mini_dataset.append({
                "state_before": snap_before,
                "action": action,
                "state_after": snap_after
            })

        total_loss = 0.0

        # --- Train World Model ---
        X, Y = _build_training_pairs(mini_dataset)
        Xt = torch.tensor(X, dtype=torch.float32).to(self.device)
        Yt = torch.tensor(Y, dtype=torch.float32).to(self.device)
        
        self.optimizers["world_model"].zero_grad()
        pred = self.models["world_model"](Xt)
        loss_wm = self.loss_fns["world_model"](pred, Yt)
        loss_wm.backward()
        self.optimizers["world_model"].step()
        total_loss += loss_wm.item()
        
        # --- Train Policy ---
        Xp, Yp = _build_policy_data(mini_dataset)
        Xp_t = torch.tensor(Xp, dtype=torch.float32).to(self.device)
        Yp_t = torch.tensor(Yp, dtype=torch.long).to(self.device)
        
        self.optimizers["policy"].zero_grad()
        logits = self.models["policy"](Xp_t)
        loss_p = self.loss_fns["policy"](logits, Yp_t)
        loss_p.backward()
        self.optimizers["policy"].step()
        total_loss += loss_p.item()

        # --- Train Argument Nets (Grounding Recovery) ---
        file_X, file_Y, dir_X, dir_Y = _build_argument_data(mini_dataset)
        
        if len(file_X) > 0:
            file_Xt = torch.tensor(file_X, dtype=torch.float32).to(self.device)
            file_Yt = torch.tensor(file_Y, dtype=torch.long).to(self.device)
            # Clip targets to match model's MAX_FILES to avoid out-of-bounds
            # (In theory, online learning might see 101st file, but model only has 100 slots)
            max_out = self.models["file_net"].net[4].out_features
            file_Yt = torch.clamp(file_Yt, 0, max_out - 1)
            
            self.optimizers["file_net"].zero_grad()
            logits = self.models["file_net"](file_Xt)
            loss_file = self.loss_fns["file_net"](logits, file_Yt)
            loss_file.backward()
            self.optimizers["file_net"].step()
            total_loss += loss_file.item()

        if len(dir_X) > 0:
            dir_Xt = torch.tensor(dir_X, dtype=torch.float32).to(self.device)
            dir_Yt = torch.tensor(dir_Y, dtype=torch.long).to(self.device)
            max_out = self.models["dir_net"].net[4].out_features
            dir_Yt = torch.clamp(dir_Yt, 0, max_out - 1)
            
            self.optimizers["dir_net"].zero_grad()
            logits = self.models["dir_net"](dir_Xt)
            loss_dir = self.loss_fns["dir_net"](logits, dir_Yt)
            loss_dir.backward()
            self.optimizers["dir_net"].step()
            total_loss += loss_dir.item()

        # --- Train Effect Model ---
        object_index = self.models["object_index"]
        Xe, Ye = _build_object_effect_data(mini_dataset, object_index)
        Xe_t = torch.tensor(Xe, dtype=torch.float32).to(self.device)
        Ye_t = torch.tensor(Ye, dtype=torch.float32).to(self.device)
        
        self.optimizers["effect_model"].zero_grad()
        logits = self.models["effect_model"](Xe_t)
        loss_e = self.loss_fns["effect_model"](logits, Ye_t)
        loss_e.backward()
        self.optimizers["effect_model"].step()
        total_loss += loss_e.item()
        
        return total_loss

