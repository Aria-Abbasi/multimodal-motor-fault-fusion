import numpy as np
from scipy.stats import kurtosis, skew

def extract_statistical_features(signal: np.ndarray) -> np.ndarray:
    """
    Computes 10 standard statistical features for a 1D signal.
    """
    abs_signal = np.abs(signal)
    rms = np.sqrt(np.mean(signal**2))
    peak = np.max(abs_signal)
    
    features = [
        np.mean(signal),           # Mean
        np.std(signal),            # Standard Deviation
        rms,                       # Root Mean Square
        kurtosis(signal),          # Kurtosis
        skew(signal),              # Skewness
        peak,                      # Peak Value
        peak / rms if rms > 0 else 0, # Crest Factor
        rms / np.mean(abs_signal) if np.mean(abs_signal) > 0 else 0, # Shape Factor
        peak / np.mean(abs_signal) if np.mean(abs_signal) > 0 else 0, # Impulse Factor
        peak / (np.mean(np.sqrt(abs_signal))**2) if np.mean(np.sqrt(abs_signal)) > 0 else 0 # Margin Factor
    ]
    return np.array(features)
