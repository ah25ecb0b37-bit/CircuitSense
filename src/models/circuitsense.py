"""
CircuitSense — Full Model: CNN → BiLSTM → Attention → Classifier
"""
import torch
import torch.nn as nn
from .cnn_extractor import CNNExtractor
from .attention import TemporalAttention

class CircuitSenseModel(nn.Module):
    def __init__(self, cfg: dict):
        super().__init__()
        in_ch   = cfg.get("input_channels", 1)
        cnn_ch  = cfg.get("cnn_channels", [64, 128, 256])
        ksize   = cfg.get("cnn_kernel_size", 7)
        lstm_h  = cfg.get("lstm_hidden", 256)
        lstm_l  = cfg.get("lstm_layers", 2)
        lstm_d  = cfg.get("lstm_dropout", 0.3)
        attn_h  = cfg.get("attention_heads", 8)
        n_cls   = cfg.get("num_classes", 8)
        clf_d   = cfg.get("classifier_dropout", 0.4)

        self.cnn  = CNNExtractor(in_ch, cnn_ch, ksize)
        self.lstm = nn.LSTM(
            input_size=cnn_ch[-1],
            hidden_size=lstm_h,
            num_layers=lstm_l,
            batch_first=True,
            bidirectional=True,
            dropout=lstm_d if lstm_l > 1 else 0.0,
        )
        lstm_out_dim = lstm_h * 2  # bidirectional
        self.attn = TemporalAttention(lstm_out_dim, attn_h)
        self.classifier = nn.Sequential(
            nn.Linear(lstm_out_dim, 256),
            nn.GELU(),
            nn.Dropout(clf_d),
            nn.Linear(256, 128),
            nn.GELU(),
            nn.Dropout(clf_d),
            nn.Linear(128, n_cls),
        )

    def forward(self, x):
        # x: (B, 1, L)
        feat = self.cnn(x)                        # (B, 256, L')
        feat = feat.permute(0, 2, 1)              # (B, L', 256)
        lstm_out, _ = self.lstm(feat)             # (B, L', 512)
        context, attn_w = self.attn(lstm_out)     # (B, 512)
        logits = self.classifier(context)         # (B, n_cls)
        return logits, attn_w

    def predict_proba(self, x):
        logits, attn_w = self.forward(x)
        return torch.softmax(logits, dim=-1), attn_w
