"""
CircuitSense — Signal Augmentation
Physically meaningful augmentations for circuit waveforms.
"""
import torch
import numpy as np

class SignalAugmentor:
    def __init__(self, cfg: dict = None):
        cfg = cfg or {}
        self.noise_prob    = cfg.get("aug_noise_prob",    0.4)
        self.scale_prob    = cfg.get("aug_scale_prob",    0.3)
        self.timewarp_prob = cfg.get("aug_timewarp_prob", 0.2)
        self.dropout_prob  = cfg.get("aug_dropout_prob",  0.2)

    def __call__(self, x: torch.Tensor) -> torch.Tensor:
        # x shape: (1, L)
        if np.random.rand() < self.noise_prob:
            x = x + torch.randn_like(x) * 0.03
        if np.random.rand() < self.scale_prob:
            scale = np.random.uniform(0.8, 1.2)
            x = x * scale
        if np.random.rand() < self.dropout_prob:
            mask = torch.ones_like(x)
            drop_len = np.random.randint(10, 60)
            start = np.random.randint(0, x.shape[-1] - drop_len)
            mask[0, start:start + drop_len] = 0
            x = x * mask
        return x
