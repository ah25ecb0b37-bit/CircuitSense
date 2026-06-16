"""
CircuitSense — FastAPI Server
"""
from fastapi import FastAPI, HTTPException, UploadFile, File
from fastapi.middleware.cors import CORSMiddleware
import numpy as np
import io, csv, time
from pathlib import Path

from .schemas import WaveformRequest, FaultPrediction, BatchWaveformRequest, HealthResponse

app = FastAPI(
    title="CircuitSense API",
    description="Neural fault diagnosis from circuit waveforms",
    version="1.0.0",
)

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_methods=["*"],
    allow_headers=["*"],
)

predictor = None
start_time = time.time()

@app.on_event("startup")
async def load_model():
    global predictor
    try:
        from src.inference import CircuitSensePredictor
        base_dir = Path(__file__).resolve().parent.parent
        checkpoint_path = base_dir / "checkpoints" / "best.pt"
        print(f"Loading model from: {checkpoint_path}")
        print(f"Checkpoint exists: {checkpoint_path.exists()}")
        predictor = CircuitSensePredictor(str(checkpoint_path))
        print("Model loaded successfully.")
    except Exception as e:
        print(f"Model not loaded: {e} — run training first.")

@app.get("/health", response_model=HealthResponse)
def health():
    import torch
    device = "cuda" if torch.cuda.is_available() else "cpu"
    return HealthResponse(status="ok", model_loaded=predictor is not None, device=device)

@app.post("/predict", response_model=FaultPrediction)
def predict(req: WaveformRequest):
    if predictor is None:
        raise HTTPException(503, "Model not loaded. Run training first.")
    waveform = np.array(req.waveform, dtype=np.float32)
    if len(waveform) != 1024:
        raise HTTPException(400, f"Expected 1024 samples, got {len(waveform)}")
    result = predictor.predict(waveform)
    return FaultPrediction(**result)

@app.post("/predict/batch")
def predict_batch(req: BatchWaveformRequest):
    if predictor is None:
        raise HTTPException(503, "Model not loaded.")
    results = []
    for w in req.waveforms:
        waveform = np.array(w, dtype=np.float32)
        results.append(predictor.predict(waveform))
    return {"predictions": results, "count": len(results)}

@app.post("/predict/csv")
async def predict_csv(file: UploadFile = File(...)):
    """Upload a CSV file with one waveform per row (1024 columns)."""
    if predictor is None:
        raise HTTPException(503, "Model not loaded.")
    content = await file.read()
    reader = csv.reader(io.StringIO(content.decode()))
    results = []
    for row in reader:
        try:
            waveform = np.array([float(v) for v in row[:1024]], dtype=np.float32)
            results.append(predictor.predict(waveform))
        except Exception as e:
            results.append({"error": str(e)})
    return {"predictions": results, "count": len(results)}