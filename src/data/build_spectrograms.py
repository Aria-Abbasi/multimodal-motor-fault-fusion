"""
src/data/build_spectrograms.py
Generates 128x128 STFT log-spectrogram tensors with strict train-only normalization.
(Finalized Version with Dataset-Agnostic Resilience & MATLAB struct unpacking)
"""
import os
import yaml
import torch
import torch.nn.functional as F
import numpy as np
import pandas as pd
import scipy.signal
import matplotlib.pyplot as plt
from pathlib import Path
from tqdm import tqdm
from .signal_io import load_recording_signals
import argparse

def compute_stft_spectrogram(signal: np.ndarray, target_size=(128, 128)) -> torch.Tensor:
    """Computes STFT, converts to log-magnitude, and resizes to target shape."""
    # Add tiny noise to prevent log(0) issues
    signal = signal + np.random.normal(0, 1e-8, len(signal))
    f, t, Zxx = scipy.signal.stft(signal, window='hann', nperseg=256, noverlap=128)
    log_spec = np.log(np.abs(Zxx) + 1e-8)
    
    # Convert to torch and resize using bilinear interpolation
    tensor_spec = torch.from_numpy(log_spec).unsqueeze(0).unsqueeze(0).float()
    resized_spec = F.interpolate(tensor_spec, size=target_size, mode='bilinear', align_corners=False)
    
    return resized_spec.squeeze(0).squeeze(0)

def main():
    parser = argparse.ArgumentParser(description="Build spectrogram dataset")
    parser.add_argument("--config", type=str, default="configs/base.yaml")
    parser.add_argument("--split_file", type=str, required=True)
    parser.add_argument("--dataset", type=str, required=True)
    args = parser.parse_args()

    # Load configuration for output paths
    with open(args.config, "r") as f:
        config = yaml.safe_load(f)
        
    # Setup directory structure
    base_out_dir = Path(config["paths"]["processed"]) / args.dataset / Path(args.split_file).stem
    tensor_dir = base_out_dir / "tensors"
    qc_dir = base_out_dir / "qc_plots"
    tensor_dir.mkdir(parents=True, exist_ok=True)
    qc_dir.mkdir(parents=True, exist_ok=True)

    # 1. Load the Split File directly
    df = pd.read_csv(args.split_file)
    
    # 2. Resilience Patch: Inject dataset column if missing (Common in CWRU splits)
    if 'dataset' not in df.columns:
        print(f"⚠️ 'dataset' column missing in {args.split_file}. Using argument: {args.dataset}")
        df['dataset'] = args.dataset
    
    # Clean strings
    df['split'] = df['split'].astype(str).str.strip().str.lower()
    df['dataset'] = df['dataset'].astype(str).str.strip().str.lower()
    df['recording_id'] = df['recording_id'].astype(str).str.strip()
    
    # 3. Filter for the requested dataset
    df = df[df['dataset'] == args.dataset.lower()]

    print(f"\n--- Processing {args.dataset} | Protocol: {Path(args.split_file).stem} ---")
    print(f"Total recordings in split file: {len(df)}")
    
    if len(df) == 0:
        raise ValueError(f"No records found for dataset '{args.dataset}' in {args.split_file}")

    print(f"Split distribution: {df['split'].value_counts().to_dict()}")

    # Signal slicing parameters
    window_size = 4096
    overlap = 0.5
    step_size = int(window_size * (1 - overlap))

    # ==========================================
    # PASS 1: Calculate Train Statistics ONLY
    # ==========================================
    print("\nPass 1: Slicing windows & calculating training statistics (Memory-Safe)...")
    train_vib_sum, train_vib_sq_sum, train_samples = 0.0, 0.0, 0
    train_curr_sum, train_curr_sq_sum = 0.0, 0.0
    
    train_df = df[df['split'] == 'train']
    if len(train_df) == 0:
        raise ValueError("No 'train' split rows found! Check your split CSV formatting.")

    for _, row in tqdm(train_df.iterrows(), total=len(train_df), desc="Pass 1: Stats"):
        try:
            vib, curr = load_recording_signals(Path(row['source_path']), args.dataset)
            vib = np.asarray(vib).flatten()
            curr = np.asarray(curr).flatten()
            
            num_windows = (len(vib) - window_size) // step_size + 1
            for i in range(num_windows):
                start, end = i * step_size, i * step_size + window_size
                v_win, c_win = vib[start:end], curr[start:end]
                
                train_samples += window_size
                train_vib_sum += np.sum(v_win)
                train_vib_sq_sum += np.sum(v_win ** 2)
                train_curr_sum += np.sum(c_win)
                train_curr_sq_sum += np.sum(c_win ** 2)
        except Exception:
            continue

    if train_samples == 0:
        raise ValueError("Processed 0 windows. Check signal loading logic and file paths.")

    # Compute Global Mean and Standard Deviation (Train-only)
    vib_mean = train_vib_sum / train_samples
    vib_std = np.sqrt(max(0, (train_vib_sq_sum / train_samples) - (vib_mean ** 2)))
    curr_mean = train_curr_sum / train_samples
    curr_std = np.sqrt(max(0, (train_curr_sq_sum / train_samples) - (curr_mean ** 2)))
    
    vib_std = vib_std if vib_std > 1e-6 else 1.0
    curr_std = curr_std if curr_std > 1e-6 else 1.0
    
    print(f"Train Stats -> Vib: µ={vib_mean:.4f}, σ={vib_std:.4f} | Curr: µ={curr_mean:.4f}, σ={curr_std:.4f}")

    # ==========================================
    # PASS 2: Normalize, STFT, and Save
    # ==========================================
    print("\nPass 2: Generating spectrograms and saving tensors to disk...")
    windows_index_data = []
    global_window_idx = 0
    
    for _, row in tqdm(df.iterrows(), total=len(df), desc="Pass 2: Saving"):
        try:
            vib, curr = load_recording_signals(Path(row['source_path']), args.dataset)
            vib, curr = np.asarray(vib).flatten(), np.asarray(curr).flatten()
            num_windows = (len(vib) - window_size) // step_size + 1
            
            for i in range(num_windows):
                start, end = i * step_size, i * step_size + window_size
                
                # Z-Score Normalization
                v_norm = (vib[start:end] - vib_mean) / vib_std
                c_norm = (curr[start:end] - curr_mean) / curr_std
                
                # Compute individual spectrograms
                v_spec = compute_stft_spectrogram(v_norm)
                c_spec = compute_stft_spectrogram(c_norm)
                
                # Stack as 2-channel multimodal tensor (2, 128, 128)
                multimodal_tensor = torch.stack([v_spec, c_spec])
                tensor_filename = f"{row['recording_id']}_w{i}.pt"
                
                torch.save(multimodal_tensor, tensor_dir / tensor_filename)
                
                windows_index_data.append({
                    'tensor_id': tensor_filename,
                    'recording_id': row['recording_id'],
                    'split': row['split'],
                    'fault_family': row.get('fault_family', 'unknown'),
                    'health_label': row.get('health_label', 'unknown')
                })

                # QC Plot every 10k windows
                if global_window_idx % 10000 == 0:
                    fig, axes = plt.subplots(1, 2, figsize=(10, 4))
                    axes[0].imshow(v_spec.numpy(), aspect='auto', origin='lower')
                    axes[0].set_title(f"Vib STFT ({row['health_label']})")
                    axes[1].imshow(c_spec.numpy(), aspect='auto', origin='lower')
                    axes[1].set_title(f"Curr STFT ({row['health_label']})")
                    plt.savefig(qc_dir / f"qc_plot_{global_window_idx}.png")
                    plt.close()
                    
                global_window_idx += 1
        except Exception:
            continue

    pd.DataFrame(windows_index_data).to_csv(base_out_dir / "windows_index.csv", index=False)
    print(f"\n--- Done! Saved {global_window_idx} windows to {tensor_dir} ---")

if __name__ == "__main__":
    main()
