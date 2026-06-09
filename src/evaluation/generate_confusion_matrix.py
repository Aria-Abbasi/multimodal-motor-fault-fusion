import torch
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from pathlib import Path
from sklearn.metrics import confusion_matrix
import sys

# Ensure local imports work
sys.path.append(str(Path(__file__).resolve().parents[2]))
from src.evaluation.checkpoint import load_model_from_checkpoint

def main():
    print("🎨 Generating Figure 3: Confusion Matrix...")
    device = torch.device("cpu") # Safe for your current VM
    
    # 1. Find the best NLN-EMP Fusion model checkpoint
    ckpt_dir = Path("artifacts/checkpoints")
    checkpoints = list(ckpt_dir.glob("model_fusion_currTrue_*.pth"))
    if not checkpoints:
        print("❌ Could not find a saved fusion model checkpoint!")
        return
    best_ckpt = checkpoints[0] # Grab the first available seed
    
    # 2. Load the Test Data Index
    data_dir = Path("data/processed/nln_emp/nln_emp_leave_one_speed_out")
    index_path = data_dir / "windows_index.csv"
    tensor_dir = data_dir / "tensors"
    
    df = pd.read_csv(index_path)
    test_df = df[df['split'] == 'test'].reset_index(drop=True)
    
    # 3. Initialize Model and Load Weights
    model, _ = load_model_from_checkpoint(best_ckpt, device)
    
    print(f"Loading {len(test_df)} test samples. This will take ~10-20 seconds on CPU...")
    
    y_true = []
    y_pred = []
    
    # 4. Run Inference Loop
    with torch.no_grad():
        for idx in range(len(test_df)):
            tensor_path = tensor_dir / test_df.iloc[idx]['tensor_id']
            # Load the 2-channel spectrogram and add batch dimension
            x = torch.load(tensor_path, map_location=device, weights_only=True).unsqueeze(0)
            
            # True label: 1 if faulty, 0 if healthy
            label_str = str(test_df.iloc[idx]['health_label']).lower()
            true_label = 1 if 'fault' in label_str else 0
            
            # Forward pass
            out_health, _ = model(x)
            pred_label = torch.argmax(out_health, dim=1).item()
            
            y_true.append(true_label)
            y_pred.append(pred_label)
            
            if idx % 500 == 0 and idx > 0:
                print(f"Processed {idx}/{len(test_df)} samples...")

    # 5. Calculate and Plot Confusion Matrix
    cm = confusion_matrix(y_true, y_pred)
    
    plt.style.use('seaborn-v0_8-paper')
    sns.set_context("paper", font_scale=1.5)
    
    fig, ax = plt.subplots(figsize=(7, 6))
    sns.heatmap(cm, annot=True, fmt='d', cmap='Blues', 
                xticklabels=['Healthy', 'Faulty'], 
                yticklabels=['Healthy', 'Faulty'],
                cbar=False, ax=ax, annot_kws={"size": 18})
    
    ax.set_ylabel('True Condition', fontweight='bold', labelpad=15)
    ax.set_xlabel('Predicted Condition', fontweight='bold', labelpad=15)
    ax.set_title('Confusion Matrix: Fusion Model (NLN-EMP Test Set)', pad=20)
    
    # 6. Save the Figure
    out_dir = Path("results/figures")
    out_dir.mkdir(parents=True, exist_ok=True)
    out_path = out_dir / "fig3_confusion_matrix.pdf"
    
    plt.tight_layout()
    fig.savefig(out_path, dpi=300, bbox_inches='tight')
    print(f"\n✅ Success! Saved Confusion Matrix to {out_path}")

if __name__ == "__main__":
    main()
