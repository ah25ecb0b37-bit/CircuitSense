"""
CircuitSense — Visualization utilities
"""
import numpy as np
import matplotlib.pyplot as plt
import matplotlib.cm as cm

FAULT_COLORS = {
    "NORMAL": "#4fa89e",
    "CAP_DEGRADED": "#e8af34",
    "CAP_SHORT": "#dd6974",
    "RES_DRIFT": "#fdab43",
    "TRANSISTOR_SATURATION": "#a86fdf",
    "DIODE_OPEN": "#5591c7",
    "INDUCTOR_CORE_SAT": "#6daa45",
    "POWER_RAIL_NOISE": "#d163a7",
}

def plot_waveform_with_cam(waveform, cam, fault_name, confidence, save_path=None):
    fig, (ax1, ax2) = plt.subplots(2, 1, figsize=(12, 6), facecolor="#111210")
    t = np.arange(len(waveform))

    ax1.set_facecolor("#181917")
    ax1.plot(t, waveform, color=FAULT_COLORS.get(fault_name, "#4fa89e"), linewidth=1.5)
    ax1.set_title(f"Waveform — Predicted: {fault_name} ({confidence*100:.1f}%)",
                  color="#d4d3cf", fontsize=12)
    ax1.tick_params(colors="#7a7975"); ax1.spines[:].set_color("#333431")

    cam_up = np.interp(np.linspace(0, 1, len(waveform)), np.linspace(0, 1, len(cam)), cam)
    ax2.set_facecolor("#181917")
    ax2.fill_between(t, cam_up, alpha=0.8,
                     color="#dd6974" if fault_name != "NORMAL" else "#4fa89e")
    ax2.set_title("Grad-CAM — Important Time Regions", color="#d4d3cf", fontsize=11)
    ax2.tick_params(colors="#7a7975"); ax2.spines[:].set_color("#333431")

    plt.tight_layout(pad=2)
    if save_path:
        plt.savefig(save_path, dpi=150, bbox_inches="tight")
    return fig
