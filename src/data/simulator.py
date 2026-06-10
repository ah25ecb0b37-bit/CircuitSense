"""
CircuitSense — Synthetic Waveform Generator
Generates SPICE-inspired RC/RLC circuit waveforms with injected faults.
Each fault class has a physically motivated signal signature.
"""

import numpy as np
import pandas as pd
import os
from pathlib import Path
from tqdm import tqdm
import argparse

# ─── Fault Labels ──────────────────────────────────────────────────────────────
FAULT_CLASSES = {
    0: "NORMAL",
    1: "CAP_DEGRADED",
    2: "CAP_SHORT",
    3: "RES_DRIFT",
    4: "TRANSISTOR_SATURATION",
    5: "DIODE_OPEN",
    6: "INDUCTOR_CORE_SAT",
    7: "POWER_RAIL_NOISE",
}

def add_noise(signal, std=0.02):
    return signal + np.random.normal(0, std, len(signal))

def generate_waveform(fault_class: int, fs=10000, duration=0.1024, noise_std=0.02) -> np.ndarray:
    """Generate a synthetic waveform for a given fault class."""
    n = int(fs * duration)  # 1024 samples
    t = np.linspace(0, duration, n)
    f0 = 50.0  # fundamental frequency Hz

    if fault_class == 0:  # NORMAL — clean sinusoid
        sig = np.sin(2 * np.pi * f0 * t)

    elif fault_class == 1:  # CAP_DEGRADED — increased ripple + phase shift
        sig = np.sin(2 * np.pi * f0 * t + 0.3)
        ripple = 0.15 * np.sin(2 * np.pi * 6 * f0 * t)
        sig = sig + ripple

    elif fault_class == 2:  # CAP_SHORT — sudden amplitude collapse
        sig = np.sin(2 * np.pi * f0 * t)
        fault_start = n // 3
        sig[fault_start:] *= np.linspace(1.0, 0.05, n - fault_start)

    elif fault_class == 3:  # RES_DRIFT — amplitude shift + DC offset
        amplitude = np.random.uniform(1.2, 1.5)
        dc_offset = np.random.uniform(0.1, 0.3)
        sig = amplitude * np.sin(2 * np.pi * f0 * t) + dc_offset

    elif fault_class == 4:  # TRANSISTOR_SATURATION — clipped waveform
        sig = np.sin(2 * np.pi * f0 * t)
        clip = np.random.uniform(0.5, 0.75)
        sig = np.clip(sig, -clip, clip)

    elif fault_class == 5:  # DIODE_OPEN — half-wave rectified appearance
        sig = np.sin(2 * np.pi * f0 * t)
        sig[sig < 0] = 0  # negative half missing

    elif fault_class == 6:  # INDUCTOR_CORE_SAT — distorted with harmonics
        sig = (np.sin(2 * np.pi * f0 * t)
               + 0.3 * np.sin(2 * np.pi * 3 * f0 * t)
               + 0.15 * np.sin(2 * np.pi * 5 * f0 * t))
        # Sudden saturation knee
        knee = n // 2
        sig[knee:] += 0.4 * np.abs(np.sin(2 * np.pi * 2 * f0 * t[knee:]))

    elif fault_class == 7:  # POWER_RAIL_NOISE — high-freq interference on rail
        sig = np.sin(2 * np.pi * f0 * t)
        hf_noise = 0.2 * np.sin(2 * np.pi * 1500 * t + np.random.uniform(0, 2 * np.pi))
        burst = np.zeros(n)
        burst_start = np.random.randint(0, n // 2)
        burst_len = np.random.randint(n // 8, n // 4)
        burst[burst_start:burst_start + burst_len] = 1
        sig = sig + hf_noise * burst

    return add_noise(sig, std=noise_std)


def generate_dataset(samples_per_class=1250, fs=10000, duration=0.1024,
                     noise_std=0.02, output_dir="data/raw", seed=42):
    np.random.seed(seed)
    Path(output_dir).mkdir(parents=True, exist_ok=True)
    n = int(fs * duration)
    col_names = [f"t{i}" for i in range(n)]

    all_rows = []
    print(f"Generating {samples_per_class * len(FAULT_CLASSES)} waveforms...")

    for fault_class, fault_name in tqdm(FAULT_CLASSES.items()):
        for _ in range(samples_per_class):
            waveform = generate_waveform(fault_class, fs, duration, noise_std)
            row = list(waveform) + [fault_class, fault_name]
            all_rows.append(row)

    df = pd.DataFrame(all_rows, columns=col_names + ["label", "fault_name"])
    df = df.sample(frac=1, random_state=seed).reset_index(drop=True)  # shuffle

    output_path = Path(output_dir) / "waveforms.csv"
    df.to_csv(output_path, index=False)
    print(f"Dataset saved: {output_path}  ({len(df)} samples)")

    # Save metadata
    meta = {
        "samples_per_class": samples_per_class,
        "total_samples": len(df),
        "fs": fs,
        "duration": duration,
        "sequence_length": n,
        "fault_classes": FAULT_CLASSES,
    }
    import json
    with open(Path(output_dir) / "metadata.json", "w") as f:
        json.dump(meta, f, indent=2)

    return df


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="CircuitSense waveform generator")
    parser.add_argument("--samples", type=int, default=1250, help="Samples per class")
    parser.add_argument("--output", type=str, default="data/raw")
    parser.add_argument("--seed", type=int, default=42)
    args = parser.parse_args()

    generate_dataset(
        samples_per_class=args.samples,
        output_dir=args.output,
        seed=args.seed
    )
