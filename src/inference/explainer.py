"""
CircuitSense — Grad-CAM Explainability
Highlights which time regions triggered the fault prediction.
"""
import torch
import numpy as np

class GradCAMExplainer:
    def __init__(self, model, target_layer_name="cnn.net.8"):
        self.model = model
        self.gradients = None
        self.activations = None

        # Register hooks
        target_layer = dict(model.named_modules())[target_layer_name]
        target_layer.register_forward_hook(self._save_activation)
        target_layer.register_backward_hook(self._save_gradient)

    def _save_activation(self, module, input, output):
        self.activations = output.detach()

    def _save_gradient(self, module, grad_input, grad_output):
        self.gradients = grad_output[0].detach()

    def explain(self, x: torch.Tensor, target_class: int = None) -> np.ndarray:
        self.model.eval()
        x.requires_grad_(True)
        logits, _ = self.model(x)

        if target_class is None:
            target_class = logits.argmax(dim=1).item()

        self.model.zero_grad()
        logits[0, target_class].backward()

        # Global average pooling of gradients over time
        weights = self.gradients.mean(dim=-1, keepdim=True)  # (B, C, 1)
        cam = (weights * self.activations).sum(dim=1)          # (B, T)
        cam = torch.relu(cam)
        cam = cam / (cam.max() + 1e-8)
        return cam.squeeze().cpu().numpy()
