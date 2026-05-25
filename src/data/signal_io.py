"""
src/data/signal_io.py
Isolates raw data loading logic for NLN-EMP, Paderborn, and CWRU.
"""
import numpy as np
import scipy.io as sio
import pandas as pd
from pathlib import Path

def load_recording_signals(filepath: Path, dataset_name: str) -> tuple[np.ndarray, np.ndarray]:
    """
    Loads raw vibration and current signals from a given file path.
    Returns:
        vibration (1D np.ndarray), current (1D np.ndarray)
    """
    path_str = str(filepath)

    # ==========================================
    # 1. NLN-EMP (Numbered CSV Columns & Piped Paths)
    # ==========================================
    if dataset_name == "nln_emp":
        # NLN-EMP metadata joins multiple time segments with '|'
        if '|' in path_str:
            # We take the first chunk (e.g., ch1.csv). It has plenty of data for our windows.
            path_str = path_str.split('|')[0]
            
        actual_path = Path(path_str)
        if not actual_path.exists():
            raise FileNotFoundError(f"Missing file: {actual_path}")
            
        # NLN-EMP files have columns: time, 0, 1, 2, ..., 14
        df = pd.read_csv(actual_path)
        
        # Standard NLN-EMP mapping: '0' is vibration, '1' is phase current
        vib = df['0'].values.flatten()
        curr = df['1'].values.flatten()
        
        return vib, curr

    # ==========================================
    # 2. PADERBORN (Nested .mat structures)
    # ==========================================
    elif dataset_name == "paderborn":
        actual_path = Path(path_str)
        if not actual_path.exists():
            raise FileNotFoundError(f"Missing file: {actual_path}")
            
        mat_id = actual_path.stem
        mat = sio.loadmat(str(actual_path))
        
        # Try to use the stem name as the key, fallback to the first real key if it differs
        if mat_id in mat:
            data = mat[mat_id]
        else:
            struct_name = [k for k in mat.keys() if not k.startswith('__')][0]
            data = mat[struct_name]
        
        # Dig into the nested 'Y' struct where Paderborn hides the arrays
        Y = data['Y'][0, 0]
        channels = Y if len(Y.shape) == 1 else Y[0]
        
        vib, curr = None, None
        
        for channel in channels:
            # Clean up the name string
            name = str(channel['Name'][0])
            if "['" in name: 
                name = channel['Name'][0][0]
            
            # Extract and flatten the data array
            array_data = channel['Data'][0]
            if len(array_data.shape) > 1 and array_data.shape[0] == 1:
                array_data = array_data.flatten()
            else:
                array_data = array_data.flatten()
                
            # Assign to our variables
            if 'vibration' in name.lower():
                vib = array_data
            elif 'current' in name.lower() and curr is None:
                curr = array_data
                
        if vib is None or curr is None:
            raise ValueError(f"Failed to find Vib/Curr channels in {actual_path}")
            
        return vib, curr

    # ==========================================
    # 3. CWRU (Dynamic .mat keys)
    # ==========================================
    elif dataset_name == "cwru":
        actual_path = Path(path_str)
        if not actual_path.exists():
            raise FileNotFoundError(f"Missing file: {actual_path}")
            
        mat = sio.loadmat(actual_path)
        
        # CWRU keys change based on the file (e.g., 'X239_DE_time')
        de_key = [k for k in mat.keys() if 'DE_time' in k]
        if not de_key:
            raise ValueError(f"Could not find DE_time channel in {actual_path}")
            
        vib = mat[de_key[0]].flatten()
        
        # CWRU benchmark is vibration-only. We mock current with zeros for the fusion pipeline.
        curr = np.zeros_like(vib) 
        
        return vib, curr

    else:
        raise ValueError(f"Unknown dataset: {dataset_name}")
