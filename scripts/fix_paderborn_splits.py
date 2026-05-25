import pandas as pd
from pathlib import Path

def fix_and_inspect(file_path):
    if not Path(file_path).exists():
        print(f"File not found: {file_path}")
        return
    
    df = pd.read_csv(file_path)
    print(f"\n--- Inspecting {file_path} ---")
    
    if 'split' in df.columns:
        # Show what we found originally
        print("Original labels found:", df['split'].unique())
        
        # Force to lowercase and strip whitespace
        df['split'] = df['split'].astype(str).str.strip().str.lower()
        df.to_csv(file_path, index=False)
        
        print("Fixed counts:")
        print(df['split'].value_counts())
    else:
        print("CRITICAL: 'split' column missing!")

fix_and_inspect('data/splits/paderborn_condition_generalization.csv')
fix_and_inspect('data/splits/paderborn_artificial_to_natural.csv')
