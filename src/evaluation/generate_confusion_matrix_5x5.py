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
from src.models.multimodal_cross_attention import MultimodalMotorModel

def main():
    print("🎨 Generating Figure 3: TRUE 5x5 Fault Family Confusion Matrix...")
    print("⚠️  NOTE: This requires a model trained without the 'families=[0]' bug!")
    device = torch.device("cpu")
    
    ckpt_dir = Path("artifacts/checkpoints")
    checkpoints = list(ckpt_dir.glob("model_fusion_currTrue_*.pth"))
    if not checkpoints:
        print("❌ Could not find a saved fusion checkpoint.")
        return
    best_ckpt = checkpoints[0] 
    
    data_dir = Path("data/processed/nln_emp/nln_emp_leave_one_speed_out")
    index_path = data_dir / "windows_index.csv"
    tensor_dir = data_dir / "tensors"
    
    df = pd.read_csv(index_path)
    test_df = df[df['split'] == 'test'].reset_index(drop=True)
    
    possible_cols = ['label', 'fault_family', 'health_label', 'condition']
    label_col = next((col for col in possible_cols if col in test_df.columns), None)
    
    if not label_col:
        print("❌ Could not find the fault family column in your CSV.")
        return

    # --- THE FIX: Map the 10 raw strings into the 5 Model Families ---
    family_mapping = {
        'Healthy': 0,
        'Bearing Ball': 1,
        'Bearing Ball Spin': 1,
        'Bearing Contamination': 1,
        'Bearing Inner Race': 1,
        'Bearing Outer Race': 1,
        'Rotor Fault': 2,
        'Looseness Soft Foot': 2, # Often grouped with rotor/mechanical looseness
        'Stator Winding Fault': 3,
        'Impeller Fault': 4
    }
    family_names = ['Healthy', 'Bearing Fault', 'Rotor/Looseness', 'Stator Fault', 'Impeller Fault']

    # Initialize model EXACTLY as it was trained (5 classes)
    model = MultimodalMotorModel(num_fault_families=5, ablation_mode=None).to(device)
    state = torch.load(best_ckpt, map_location=device)
    model.load_state_dict(state["state_dict"])
    model.eval()
    
    print(f"Evaluating {len(test_df)} samples on the 5-Class Diagnostic Head...")
    y_true = []
    y_pred = []
    
    with torch.no_grad():
        for idx in range(len(test_df)):
            tensor_path = tensor_dir / test_df.iloc[idx]['tensor_id']
            x = torch.load(tensor_path, map_location=device, weights_only=True).float().unsqueeze(0)
            
            _, out_family = model(x)
            pred_family = torch.argmax(out_family, dim=1).item()
            
            # Map the specific string to its 0-4 parent category
            raw_str = str(test_df.iloc[idx][label_col])
            true_family = family_mapping.get(raw_str, 0) # Default to 0 if unknown
            
            y_true.append(true_family)
            y_pred.append(pred_family)

    cm = confusion_matrix(y_true, y_pred)
    
    plt.style.use('seaborn-v0_8-paper')
    sns.set_context("paper", font_scale=1.2)
    
    fig, ax = plt.subplots(figsize=(9, 8))
    sns.heatmap(cm, annot=True, fmt='d', cmap='Blues', cbar=False, ax=ax, 
                xticklabels=family_names, yticklabels=family_names, annot_kws={"size": 14})
    
    ax.set_ylabel('True Fault Family', fontweight='bold', labelpad=15)
    ax.set_xlabel('Predicted Fault Family', fontweight='bold', labelpad=15)
    ax.set_title('Confusion Matrix: Fault Family Diagnosis', pad=20)
    
    plt.setp(ax.get_xticklabels(), rotation=45, ha="right", rotation_mode="anchor")
    plt.setp(ax.get_yticklabels(), rotation=0)
    
    out_dir = Path("results/figures")
    out_dir.mkdir(parents=True, exist_ok=True)
    out_path = out_dir / "fig3_confusion_matrix_5x5.pdf"
    
    plt.tight_layout()
    fig.savefig(out_path, dpi=300, bbox_inches='tight')
    print(f"\n✅ Success! Saved future-proof 5x5 matrix to {out_path}")

if __name__ == "__main__":
    main()
