"""
CircuitSense — 1D-CNN Feature Extractor
Learns local spectral + morphological patterns from raw waveforms.
"""
import torch
import torch.nn as nn

class CNNExtractor(nn.Module):
    def __init__(self, in_channels=1, channels=[64, 128, 256], kernel_size=7):
        super().__init__()
        layers = []
        prev = in_channels
        for ch in channels:
            layers += [
                nn.Conv1d(prev, ch, kernel_size, padding=kernel_size//2),
                nn.BatchNorm1d(ch),
                nn.GELU(),
                nn.MaxPool1d(2),
            ]
            prev = ch
        self.net = nn.Sequential(*layers)
        self.out_channels = channels[-1]

    def forward(self, x):
        # x: (B, C, L)
        return self.net(x)  # (B, 256, L//8)
