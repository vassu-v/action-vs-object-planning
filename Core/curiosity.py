"""
Curiosity-Driven Exploration Engine
------------------------------------
Intrinsic reward module that breaks statistical gravity by rewarding 
transitions to novel states. Used in the post-hoc adaptation experiments 
to test whether exploration can overcome the Grounding Wall.
"""

import numpy as np

class CuriosityModule:
    def __init__(self, buffer_size=200, novelty_weight=0.3):
        """
        buffer_size: How many recent states to compute novelty against
        novelty_weight: 0.3 recommended from math analysis
        """
        self.state_buffer = []
        self.buffer_size = buffer_size
        self.novelty_weight = novelty_weight
        self.task_weight = 1.0 - novelty_weight  # 0.7
        
        # Running mean for efficiency
        self._buffer_mean = None

    def update(self, snapshot):
        """
        Call AFTER each step to update state buffer.
        """
        global_state = snapshot["global"]
        
        self.state_buffer.append(global_state.copy())
        if len(self.state_buffer) > self.buffer_size:
            self.state_buffer.pop(0)
        
        # Recompute mean
        self._buffer_mean = np.mean(
            np.stack(self.state_buffer), axis=0
        )

    def novelty(self, snapshot):
        """
        Novelty = distance of current state from buffer mean.
        
        High novelty = rarely visited state = reward exploration
        Low novelty  = frequently visited = suppress (already known)
        """
        if self._buffer_mean is None or len(self.state_buffer) < 5:
            return 1.0  # High novelty at start (explore freely)
        
        global_state = snapshot["global"]
        raw_novelty = np.linalg.norm(global_state - self._buffer_mean)
        
        # Normalize to [0, 1] range using running stats
        return float(np.tanh(raw_novelty))  # tanh keeps it bounded

    def blend_score(self, task_score, snapshot):
        """
        Core function: blend task signal with novelty bonus.
        
        score = task_score × 0.7 + novelty × 0.3
        """
        nov = self.novelty(snapshot)
        blended = (self.task_weight * task_score) + (self.novelty_weight * nov)
        return blended, nov
