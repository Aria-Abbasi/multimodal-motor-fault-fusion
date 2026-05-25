import scipy.io as sio
from pathlib import Path
import numpy as np

file_path = 'data/raw/paderborn/K001/N15_M01_F10_K001_10.mat'
mat_id = Path(file_path).stem
print(f"Loading main struct: {mat_id}")
mat = sio.loadmat(file_path)
data = mat[mat_id]

# In Paderborn, sensor data is kept inside 'Y'
Y = data['Y'][0, 0]

vib_array = None
curr_array = None

print("--- Inspecting Channels ---")
# Scipy usually loads MATLAB struct arrays as 1D numpy object arrays
channels = Y if len(Y.shape) == 1 else Y[0]

for i, channel in enumerate(channels):
    try:
        # Extract the channel name and the actual array data
        name = str(channel['Name'][0])
        if "['" in name: name = channel['Name'][0][0] # Clean up string formatting
        
        array_data = channel['Data'][0]
        if len(array_data.shape) > 1 and array_data.shape[0] == 1:
            array_data = array_data.flatten()
            
        print(f"Index {i}: '{name}' -> Shape: {array_data.shape}")
        
        if 'vibration' in name.lower():
            vib_array = array_data
        elif 'current' in name.lower() and curr_array is None:
            curr_array = array_data
    except Exception as e:
        pass

if vib_array is not None and curr_array is not None:
    print(f"\nSUCCESS! Extracted Vib: {vib_array.shape}, Extracted Curr: {curr_array.shape}")
    num_windows = (len(vib_array) - 4096) // 2048 + 1
    print(f"Calculated windows: {num_windows}")
else:
    print("\nFAILED to identify channels. We need to dig deeper.")
