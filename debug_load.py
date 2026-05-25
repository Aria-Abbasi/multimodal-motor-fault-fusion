import pandas as pd
import numpy as np
from pathlib import Path
from src.data.signal_io import load_recording_signals

df = pd.read_csv('data/splits/paderborn_condition_generalization.csv')
train_df = df[df['split'].str.lower() == 'train']
row = train_df.iloc[0]

try:
    print(f"Loading: {row['source_path']}")
    vib, curr = load_recording_signals(Path(row['source_path']), 'paderborn')
    print(f"Vib shape: {np.shape(vib)}, Curr shape: {np.shape(curr)}")
    num_windows = (len(vib) - 4096) // 2048 + 1
    print(f"Calculated windows: {num_windows}")
except Exception as e:
    import traceback
    print(f"HIDDEN ERROR: {repr(e)}")
    traceback.print_exc()
