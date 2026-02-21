import random
import numpy as np
import torch
from .env import Action

class ReplayBuffer:
    def __init__(self, capacity=10000):
        self.capacity = capacity
        self.buffer = []
        self.position = 0
        
        # Separate hindsight buffer for cleaner analysis
        self.hindsight_buffer = []
        self.hindsight_position = 0
        self.hindsight_capacity = capacity // 2

    def add_trajectory(self, trace, goal=None, is_hindsight=False):
        """
        Adds a full execution trace to the buffer.
        
        Args:
            trace: List of transition steps.
            goal: Optional override goal (used for hindsight relabeling).
            is_hindsight: Boolean flag to route to hindsight buffer.
        """
        for step in trace:
            snap_before = step["state_before"]
            snap_after = step["state_after"]
            raw_action = step["action"]
            
            if isinstance(raw_action, Action):
                action = raw_action
            else:
                # Backward compatibility for strings
                action = Action(raw_action, step.get("args", ()))
            
            # Use provided goal, otherwise look for hindsight_goal in step
            effective_goal = goal if goal is not None else step.get("hindsight_goal", {})
            hindsight_flag = is_hindsight or step.get("is_hindsight", False)
            
            if hindsight_flag:
                self._push_hindsight(snap_before, action, snap_after, effective_goal)
            else:
                self._push(snap_before, action, snap_after, effective_goal)

    def _push(self, snap_before, action, snap_after, goal=None):
        """Saves a real transition."""
        if len(self.buffer) < self.capacity:
            self.buffer.append(None)
        
        self.buffer[self.position] = (snap_before, action, snap_after, goal)
        self.position = (self.position + 1) % self.capacity

    def _push_hindsight(self, snap_before, action, snap_after, goal=None):
        """Saves a relabeled hindsight transition."""
        if len(self.hindsight_buffer) < self.hindsight_capacity:
            self.hindsight_buffer.append(None)
        
        self.hindsight_buffer[self.hindsight_position] = (snap_before, action, snap_after, goal)
        self.hindsight_position = (self.hindsight_position + 1) % self.hindsight_capacity

    def sample(self, batch_size, hindsight_ratio=0.5):
        """
        Mixed sampling: blend real experience with hindsight.
        """
        n_hindsight = int(batch_size * hindsight_ratio) if self.hindsight_buffer else 0
        n_real = batch_size - n_hindsight
        
        batch = []
        
        if n_real > 0 and self.buffer:
            # Handle cases where buffer is smaller than requested real batch
            real_batch = random.sample(self.buffer, min(n_real, len(self.buffer)))
            batch.extend(real_batch)
        
        if n_hindsight > 0 and self.hindsight_buffer:
            hindsight_batch = random.sample(self.hindsight_buffer, min(n_hindsight, len(self.hindsight_buffer)))
            batch.extend(hindsight_batch)
            
        # For backward compatibility with existing OnlineLearner.train_step
        # it expects [(s_b, a, s_a), ...]
        return [(s_b, a, s_a) for (s_b, a, s_a, *_) in batch]

    def hindsight_stats(self):
        return {
            "real": len(self.buffer),
            "hindsight": len(self.hindsight_buffer),
            "total": len(self.buffer) + len(self.hindsight_buffer)
        }

    def __len__(self):
        return len(self.buffer) + len(self.hindsight_buffer)
