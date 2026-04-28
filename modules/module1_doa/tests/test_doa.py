"""
Unit tests for doa_node.py (synthetic, no hardware needed)

Run:
  python3 -m pytest tests/test_doa.py -q
"""

import numpy as np
import math
import os
import importlib.util

# --- Dynamically load doa_node.py ---
script_path = os.path.join(os.path.dirname(__file__), "..", "doa_node.py")
spec = importlib.util.spec_from_file_location("doa_node", script_path)
doa = importlib.util.module_from_spec(spec)
spec.loader.exec_module(doa)


def test_tau_to_angle_zero():
    """Zero delay should correspond to ~0°."""
    angle = doa.tau_to_angle(0.0, mic_spacing=0.05)
    assert abs(angle) < 1e-6


def test_tau_to_angle_known_delay():
    """Known delay for 30° input should return ~30°."""
    theta_deg = 30.0
    theta_rad = math.radians(theta_deg)
    d = 0.05
    tau = (d / doa.SPEED_OF_SOUND) * math.sin(theta_rad)
    angle = doa.tau_to_angle(tau, mic_spacing=d)
    assert abs(angle - theta_deg) < 0.5


def test_gcc_phat_synthetic_delay():
    """
    Synthetic test: refsig is delayed by N samples.
    The test asserts the estimated delay magnitude matches the known delay magnitude.
    It is tolerant to sign-convention differences between implementations.
    """
    fs = 16000
    t = np.arange(0, 0.03, 1.0 / fs)
    f = 1000.0
    sig = np.sin(2 * np.pi * f * t)

    delay_samples = 4
    refsig = np.concatenate((np.zeros(delay_samples), sig[:-delay_samples]))

    tau, cc = doa.gcc_phat(sig, refsig, fs=fs)

    expected_tau = delay_samples / fs  # magnitude of the delay

    # Check magnitude matches (sign may vary by implementation)
    assert abs(abs(tau) - expected_tau) < 1e-4, f"Expected magnitude {expected_tau}, got {tau}"

    # Basic sanity: cross-correlation should have a clear peak (not flat)
    assert np.max(np.abs(cc)) > 0.0, "Cross-correlation appears flat (no peak found)"
