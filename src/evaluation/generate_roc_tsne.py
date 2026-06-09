import torch
import torch.nn.functional as F
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from pathlib import Path
from sklearn.manifold import TSNE
from sklearn.metrics import roc_curve, auc
import sys

sys.path.append(str(Path(__file__).resolve().parents[2]))
from src.evaluation.checkpoint import load_model_from_checkpoint

def main():
    print("📈 Generating Figure 4 (ROC) and Figure 6 (t-SNE) using ALL samples...")
    device = torch.device("cpu")
    
    ckpt_dir = Path("artifacts/checkpoints")
    checkpoints = list(ckpt_dir.glob("model_fusion_currTrue_*.pth"))
    if not checkpoints:
        print("❌ Could not find a saved fusion model checkpoint!")
        return
    best_ckpt = checkpoints[0]
    
    data_dir = Path("data/processed/nln_emp/nln_emp_leave_one_speed_out")
    index_path = data_dir / "windows_index.csv"
    tensor_dir = data_dir / "tensors"
    
    df = pd.read_csv(index_path)
    test_df = df[df['split'] == 'test'].reset_index(drop=True)
    
    model, _ = load_model_from_checkpoint(best_ckpt, device)
    
    print(f"Loading FULL {len(test_df)} test samples. This will take ~15 seconds...")
    
    y_true = []
    y_scores = []
    latent_features = []
    
    with torch.no_grad():
        for idx in range(len(test_df)):
            tensor_path = tensor_dir / test_df.iloc[idx]['tensor_id']
            x = torch.load(tensor_path, map_location=device, weights_only=True).unsqueeze(0)
            
            label_str = str(test_df.iloc[idx]['health_label']).lower()
            true_label = 1 if 'fault' in label_str else 0
            
            out_health, out_family = model(x)
            
            probs = F.softmax(out_health, dim=1)
            y_scores.append(probs[0, 1].item())
            y_true.append(true_label)
            
            combined_logits = torch.cat((out_health, out_family), dim=1).squeeze().numpy()
            latent_features.append(combined_logits)

    latent_features = np.array(latent_features)
    y_true = np.array(y_true)
    y_scores = np.array(y_scores)
    
    plt.style.use('seaborn-v0_8-paper')
    sns.set_context("paper", font_scale=1.5)
    out_dir = Path("results/figures")
    out_dir.mkdir(parents=True, exist_ok=True)
    
    # --- 1. Generate ROC Curve (FULL Dataset) ---
    print("\n📊 Calculating ROC Curve on FULL dataset...")
    fpr, tpr, _ = roc_curve(y_true, y_scores)
    roc_auc = auc(fpr, tpr)
    
    fig_roc, ax_roc = plt.subplots(figsize=(7, 6))
    ax_roc.plot(fpr, tpr, color='#1b9e77', lw=2, label=f'Fusion Model (AUC = {roc_auc:.3f})')
    ax_roc.plot([0, 1], [0, 1], color='gray', lw=2, linestyle='--')
    ax_roc.set_xlim([0.0, 1.0])
    ax_roc.set_ylim([0.0, 1.05])
    ax_roc.set_xlabel('False Positive Rate', fontweight='bold')
    ax_roc.set_ylabel('True Positive Rate', fontweight='bold')
    ax_roc.set_title('Receiver Operating Characteristic (ROC)', pad=20)
    ax_roc.legend(loc="lower right")
    fig_roc.tight_layout()
    fig_roc.savefig(out_dir / "fig4_roc_curve.pdf", dpi=300, bbox_inches='tight')
    print("✅ Saved fig4_roc_curve.pdf")
    
    # --- 2. Generate t-SNE Plot (FULL Dataset, Optimized for Memory/Speed) ---
    print(f"\n🌌 Running Bulletproof t-SNE on ALL {len(latent_features)} points (Estimated time: 1-2 minutes)...")
    
    # FIXED: Changed n_iter to max_iter
    tsne = TSNE(
        n_components=2, 
        perplexity=40, 
        max_iter=1000, 
        method='barnes_hut', 
        n_jobs=-1, 
        random_state=42
    )
    tsne_results = tsne.fit_transform(latent_features)
    
    fig_tsne, ax_tsne = plt.subplots(figsize=(8, 6))
    
    # Rasterized=True makes the PDF lightweight even with 6000+ points
    scatter = ax_tsne.scatter(
        tsne_results[:, 0], tsne_results[:, 1], 
        c=y_true, cmap='coolwarm', alpha=0.6, edgecolors='none', s=15, rasterized=True
    )
    
    from matplotlib.lines import Line2D
    legend_elements = [
        Line2D([0], [0], marker='o', color='w', label='Healthy', markerfacecolor=scatter.cmap(0.0), markersize=10),
        Line2D([0], [0], marker='o', color='w', label='Faulty', markerfacecolor=scatter.cmap(1.0), markersize=10)
    ]
    ax_tsne.legend(handles=legend_elements, loc='best')
    ax_tsne.set_title('t-SNE Feature Visualization (Full Test Set)', pad=20)
    ax_tsne.set_xlabel('t-SNE Dimension 1')
    ax_tsne.set_ylabel('t-SNE Dimension 2')
    
    fig_tsne.tight_layout()
    fig_tsne.savefig(out_dir / "fig6_tsne_clusters.pdf", dpi=300, bbox_inches='tight')
    print("✅ Saved fig6_tsne_clusters.pdf")

if __name__ == "__main__":
    main()
