import pandas as pd
from pathlib import Path
import scipy.io as sio

# Load the split file
csv_path = 'data/splits/cwru_leave_one_load_out.csv'
df = pd.read_csv(csv_path)

# Check the first entry
sample_path_str = df.iloc[0]['source_path']
path = Path(sample_path_str)

print(f"--- CWRU Path Diagnostic ---")
print(f"Path in CSV: {sample_path_str}")
print(f"Absolute Path: {path.absolute()}")
print(f"File exists? {path.exists()}")

if not path.exists():
    print("\nCRITICAL: File not found. Checking current directory content...")
    # Try to see if we can find the file anywhere in data/raw/cwru
    raw_dir = Path('data/raw/cwru')
    if raw_dir.exists():
        found_files = list(raw_dir.glob('*.mat'))
        print(f"Found {len(found_files)} .mat files in {raw_dir}")
        if found_files:
            print(f"Sample file found: {found_files[0]}")
else:
    # If it exists, check the keys
    mat = sio.loadmat(path)
    keys = [k for k in mat.keys() if not k.startswith("__")]
    print(f"\nKeys found in .mat: {keys}")
