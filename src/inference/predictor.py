"""
CircuitSense — Inference Engine
Loads a trained model and predicts fault class from a waveform.
"""
import torch
import numpy as np
import yaml
from pathlib import Path
from ..models import CircuitSenseModel
from ..data.simulator import FAULT_CLASSES

class CircuitSensePredictor:
    def __init__(self, checkpoint_path="checkpoints/best.pt", device="auto"):
        if device == "auto":
            self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        else:
            self.device = torch.device(device)

        ckpt = torch.load(checkpoint_path, map_location=self.device)
        self.cfg = ckpt["cfg"]
        self.model = CircuitSenseModel(self.cfg["model"]).to(self.device)
        self.model.load_state_dict(ckpt["model_state"])
        self.model.eval()
        self.threshold = self.cfg["inference"].get("threshold", 0.6)

    def preprocess(self, waveform: np.ndarray) -> torch.Tensor:
        """Normalize and convert waveform to model input tensor."""
        w = waveform.astype(np.float32)
        w = (w - w.mean()) / (w.std() + 1e-8)
        return torch.tensor(w, dtype=torch.float32).unsqueeze(0).unsqueeze(0).to(self.device)

    @torch.no_grad()
    def predict(self, waveform: np.ndarray) -> dict:
        x = self.preprocess(waveform)
        probs, attn_w = self.model.predict_proba(x)
        probs_np = probs.cpu().numpy()[0]
        pred_class = int(probs_np.argmax())
        confidence = float(probs_np.max())

        return {
            "predicted_class": pred_class,
            "fault_name": FAULT_CLASSES[pred_class],
            "confidence": confidence,
            "all_probabilities": {FAULT_CLASSES[i]: float(p) for i, p in enumerate(probs_np)},
            "attention_weights": attn_w.cpu().numpy()[0].mean(axis=0).tolist(),
            "is_confident": confidence >= self.threshold,
        }

    @torch.no_grad()
    def predict_batch(self, waveforms: list) -> list:
        return [self.predict(w) for w in waveforms]
