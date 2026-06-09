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
    print("Generating checkpoint-aligned fault-family confusion matrix...")
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
    
    if "fault_family" not in test_df.columns:
        print("❌ Could not find the fault family column in your CSV.")
        return

    model, checkpoint = load_model_from_checkpoint(best_ckpt, device)
    family_mapping = checkpoint.get("family_to_index")
    if not family_mapping:
        raise ValueError(
            "Checkpoint has no family_to_index mapping; regenerate it with the "
            "current trainer before producing a family confusion matrix."
        )
    inverse_mapping = {index: name for name, index in family_mapping.items()}
    family_names = [
        inverse_mapping.get(index, f"class_{index}")
        for index in range(len(inverse_mapping))
    ]
    
    print(f"Evaluating {len(test_df)} samples on the 5-Class Diagnostic Head...")
    y_true = []
    y_pred = []
    
    with torch.no_grad():
        for idx in range(len(test_df)):
            tensor_path = tensor_dir / test_df.iloc[idx]['tensor_id']
            x = torch.load(tensor_path, map_location=device, weights_only=True).float().unsqueeze(0)
            
            _, out_family = model(x)
            pred_family = torch.argmax(out_family, dim=1).item()
            
            raw_family = str(test_df.iloc[idx]["fault_family"])
            true_family = family_mapping.get(raw_family, 0)
            
            y_true.append(true_family)
            y_pred.append(pred_family)

    labels = list(range(len(family_names)))
    cm = confusion_matrix(y_true, y_pred, labels=labels)
    
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
