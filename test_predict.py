import numpy as np
from src.data.simulator import generate_waveform, FAULT_CLASSES
from src.inference.predictor import CircuitSensePredictor

predictor = CircuitSensePredictor("checkpoints/best.pt")

print("\n=== CircuitSense Fault Diagnosis Test ===\n")
for fault_id, fault_name in FAULT_CLASSES.items():
    waveform = generate_waveform(fault_id)
    result = predictor.predict(waveform)
    status = "✅" if result["predicted_class"] == fault_id else "❌"
    print(f"{status} True: {fault_name:<30} → Predicted: {result['fault_name']:<30} ({result['confidence']*100:.1f}%)")