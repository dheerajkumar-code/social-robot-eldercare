#!/usr/bin/env python3
"""
Module 16 — feature_extractor.py
===================================
Shared MFCC feature extraction used by BOTH training and inference.

Why python_speech_features instead of librosa?
  - No numba dependency (librosa requires numba → fails on Pi5 often)
  - Simpler install: pip install python_speech_features
  - Faster on CPU for short audio clips
  - Proven stable for speaker recognition tasks

Feature vector layout (78 features total):
  [0:13]   mean of 13 MFCC coefficients
  [13:26]  std  of 13 MFCC coefficients
  [26:39]  mean of 13 delta-MFCC
  [39:52]  std  of 13 delta-MFCC
  [52:65]  mean of 13 delta-delta-MFCC
  [65:78]  std  of 13 delta-delta-MFCC

Using mean+std of each coefficient = captures both average voice
characteristics AND variability — critical for speaker discrimination.
"""

import numpy as np
import warnings
warnings.filterwarnings("ignore")

try:
    import python_speech_features as psf
    PSF_AVAILABLE = True
except ImportError:
    PSF_AVAILABLE = False

# ── Constants ──
SAMPLE_RATE  = 16000
N_MFCC       = 13
WIN_LEN      = 0.025    # 25ms window
WIN_STEP     = 0.010    # 10ms step
N_FILTERS    = 26       # mel filterbanks
PREEMPH      = 0.97     # pre-emphasis coefficient
FEATURE_DIM  = N_MFCC * 6   # 78 (mean+std for mfcc, delta, delta-delta)


def extract_mfcc(audio: np.ndarray,
                 sample_rate: int = SAMPLE_RATE) -> np.ndarray:
    """
    Extract 78-dimensional MFCC feature vector from audio.

    Args:
        audio      : float32 numpy array, mono, any length
        sample_rate: audio sample rate (must be 16000)

    Returns:
        np.ndarray of shape (78,) — float32
        Returns zeros on failure (graceful degradation)
    """
    if not PSF_AVAILABLE:
        raise ImportError("Install: pip install python_speech_features")

    if audio is None or len(audio) == 0:
        return np.zeros(FEATURE_DIM, dtype=np.float32)

    # Ensure float32 and proper range
    audio = audio.astype(np.float32)
    if np.max(np.abs(audio)) > 1.0:
        audio = audio / 32768.0      # int16 → float32 normalise

    # Minimum length check: need at least 1 window
    min_samples = int(WIN_LEN * sample_rate)
    if len(audio) < min_samples:
        return np.zeros(FEATURE_DIM, dtype=np.float32)

    try:
        # ── Extract MFCC ──
        mfcc = psf.mfcc(
            signal     = audio,
            samplerate = sample_rate,
            winlen     = WIN_LEN,
            winstep    = WIN_STEP,
            numcep     = N_MFCC,
            nfilt      = N_FILTERS,
            preemph    = PREEMPH,
            appendEnergy = False,     # energy in coeff 0 already
        )
        # mfcc shape: (n_frames, 13)

        if mfcc.shape[0] < 2:
            return np.zeros(FEATURE_DIM, dtype=np.float32)

        # ── Delta and delta-delta ──
        delta       = psf.delta(mfcc, N=2)
        delta_delta = psf.delta(delta, N=2)

        # ── Mean + std for each ──
        feat = np.concatenate([
            np.mean(mfcc,        axis=0),
            np.std(mfcc,         axis=0),
            np.mean(delta,       axis=0),
            np.std(delta,        axis=0),
            np.mean(delta_delta, axis=0),
            np.std(delta_delta,  axis=0),
        ]).astype(np.float32)

        # Safety: replace NaN/Inf
        feat = np.nan_to_num(feat, nan=0.0, posinf=0.0, neginf=0.0)

        return feat

    except Exception as e:
        return np.zeros(FEATURE_DIM, dtype=np.float32)


def extract_from_file(wav_path: str) -> np.ndarray:
    """
    Load a WAV file and extract MFCC features.
    Returns zero vector on error.
    """
    import soundfile as sf

    try:
        audio, sr = sf.read(wav_path, dtype="float32")
        if audio.ndim > 1:
            audio = np.mean(audio, axis=1)   # stereo → mono
        if sr != SAMPLE_RATE:
            # Simple resample using numpy interpolation
            target_len = int(len(audio) * SAMPLE_RATE / sr)
            audio = np.interp(
                np.linspace(0, len(audio), target_len),
                np.arange(len(audio)),
                audio
            ).astype(np.float32)
        return extract_mfcc(audio, SAMPLE_RATE)
    except Exception as e:
        return np.zeros(FEATURE_DIM, dtype=np.float32)


def compute_rms_energy(audio: np.ndarray) -> float:
    """RMS energy of audio frame — used for VAD."""
    return float(np.sqrt(np.mean(audio.astype(np.float32) ** 2)))


def is_speech(audio: np.ndarray, threshold: float = 0.01) -> bool:
    """
    Simple energy-based Voice Activity Detection.
    Returns True if audio likely contains speech.
    threshold: 0.01 works well for indoor environments
    """
    return compute_rms_energy(audio) > threshold
