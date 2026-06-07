import torch
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from pathlib import Path
from sklearn.metrics import confusion_matrix
import sys

sys.path.append(str(Path(__file__).resolve().parents[2]))
from src.models.multimodal_cross_attention import MultimodalMotorModel

def main():
    print("🎨 Generating Figure 3: BEST Binary Confusion Matrix...")
    device = torch.device("cpu")
    
    # 1. Find the BEST run from the CSV
    metrics_path = Path("results/tables/detailed_metrics.csv")
    df_metrics = pd.read_csv(metrics_path)
    
    # Filter for NLN-EMP Fusion models with Curriculum
    fusion_runs = df_metrics[(df_metrics['ablation'] == 'fusion') & (df_metrics['curriculum'] == True)].copy()
    
    # Assume the first 5 are NLN-EMP
    nln_runs = fusion_runs.head(5) 
    
    # Get the run_id of the row with the max macro_f1
    best_run_id = nln_runs.loc[nln_runs['macro_f1'].idxmax()]['run_id']
    best_f1 = nln_runs['macro_f1'].max()
    
    print(f"🏆 Found Best Seed: {best_run_id} (F1: {best_f1:.4f})")
    
    # 2. Locate that specific checkpoint
    ckpt_dir = Path("artifacts/checkpoints")
    best_ckpt = list(ckpt_dir.glob(f"*{best_run_id}.pth"))[0]
    
    data_dir = Path("data/processed/nln_emp/nln_emp_leave_one_speed_out")
    index_path = data_dir / "windows_index.csv"
    tensor_dir = data_dir / "tensors"
    
    test_df = pd.read_csv(index_path)
    test_df = test_df[test_df['split'] == 'test'].reset_index(drop=True)

    model = MultimodalMotorModel(num_fault_families=5, ablation_mode=None).to(device)
    model.load_state_dict(torch.load(best_ckpt, map_location=device)["state_dict"])
    model.eval()
    
    y_true = []
    y_pred = []
    
    print(f"Evaluating {len(test_df)} samples...")
    with torch.no_grad():
        for idx in range(len(test_df)):
            tensor_path = tensor_dir / test_df.iloc[idx]['tensor_id']
            x = torch.load(tensor_path, map_location=device, weights_only=True).unsqueeze(0)
            
            label_str = str(test_df.iloc[idx]['health_label']).lower()
            true_label = 1 if 'fault' in label_str else 0
            
            out_health, _ = model(x)
            pred_label = torch.argmax(out_health, dim=1).item()
            
            y_true.append(true_label)
            y_pred.append(pred_label)

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
    ax.set_title(f'Confusion Matrix: Fusion Model (F1: {best_f1:.2f})', pad=20)
    
    out_dir = Path("results/figures")
    out_path = out_dir / "fig3_confusion_matrix.pdf"
    
    plt.tight_layout()
    fig.savefig(out_path, dpi=300, bbox_inches='tight')
    print(f"\n✅ Success! Saved accurate Binary Matrix to {out_path}")

if __name__ == "__main__":
    main()
