from pydantic import BaseModel
from typing import List, Dict, Optional

class WaveformRequest(BaseModel):
    waveform: List[float]          # Raw 1024-sample waveform
    sample_rate: Optional[int] = 10000

class FaultPrediction(BaseModel):
    predicted_class: int
    fault_name: str
    confidence: float
    all_probabilities: Dict[str, float]
    attention_weights: List[float]
    is_confident: bool

class BatchWaveformRequest(BaseModel):
    waveforms: List[List[float]]

class HealthResponse(BaseModel):
    status: str
    model_loaded: bool
    device: str
    version: str = "1.0.0"
