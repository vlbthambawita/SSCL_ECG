import numpy as np
import torch
import random
from scipy.signal import resample
from scipy.ndimage import gaussian_filter1d

#Randomly scale the amplitude of the ECG signal to simulate variations in electrode placement or patient-specific differences.
def amplitude_scaling(ecg, scale_range=(0.8, 1.2)):
    scale_factor = np.random.uniform(*scale_range)
    return ecg * scale_factor


# Stretch or compress the time axis of the ECG signal to simulate variations in heart rate.
def time_stretch(ecg, stretch_factor_range=(0.8, 1.2)):
    stretch_factor = np.random.uniform(*stretch_factor_range)
    stretched_ecg = resample(ecg, int(ecg.shape[0] * stretch_factor), axis=0)
    if stretched_ecg.shape[0] > ecg.shape[0]:
        return stretched_ecg[:ecg.shape[0]]  # Crop to original length
    else:
        # Pad with zeros if shorter
        padding = ecg.shape[0] - stretched_ecg.shape[0]
        return np.pad(stretched_ecg, ((0, padding), (0, 0)), mode='constant')

# Add random Gaussian noise to the ECG signal to simulate sensor noise or environmental interference.
def add_gaussian_noise(ecg, noise_std=0.01):
    noise = np.random.normal(0, noise_std, ecg.shape)
    return ecg + noise

# Randomly mask one or more leads to simulate lead dropout.
def lead_masking(ecg, num_leads_to_mask=1):
    ecg = ecg.copy()
    leads_to_mask = np.random.choice(ecg.shape[1], num_leads_to_mask, replace=False)
    ecg[:, leads_to_mask] = 0  # Set selected leads to zero
    return ecg

#Temporal Cropping
def temporal_cropping(ecg, crop_size=4000):
    if crop_size >= ecg.shape[0]:
        return ecg  # Return the original signal if crop size exceeds length
    start = np.random.randint(0, ecg.shape[0] - crop_size)
    return ecg[start:start + crop_size, :]

# Apply a random bandpass filter to simulate noise filtering.
def bandpass_filter(ecg, low=0.5, high=50, sampling_rate=500):
    from scipy.signal import butter, filtfilt

    nyquist = 0.5 * sampling_rate
    low_cutoff = low / nyquist
    high_cutoff = high / nyquist

    b, a = butter(1, [low_cutoff, high_cutoff], btype='band')
    filtered_ecg = filtfilt(b, a, ecg, axis=0)
    return filtered_ecg