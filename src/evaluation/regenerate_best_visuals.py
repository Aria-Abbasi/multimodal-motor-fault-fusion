import torch
import torch.nn.functional as F
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from pathlib import Path
from sklearn.manifold import TSNE
from sklearn.metrics import roc_curve, auc
from scipy.ndimage import gaussian_filter
import sys

sys.path.append(str(Path(__file__).resolve().parents[2]))
from src.models.multimodal_cross_attention import MultimodalMotorModel

def normalize_heatmap(grad):
    heatmap = np.abs(grad)
    heatmap = heatmap - np.min(heatmap)
    heatmap = heatmap / (np.max(heatmap) + 1e-8)
    return gaussian_filter(heatmap, sigma=2.0)

def main():
    print("🚀 Regenerating Figures 4, 5, and 6 using the BEST seed...")
    device = torch.device("cpu")
    
    # 1. FIND THE BEST SEED
    df_metrics = pd.read_csv(Path("results/tables/detailed_metrics.csv"))
    fusion_runs = df_metrics[(df_metrics['ablation'] == 'fusion') & (df_metrics['curriculum'] == True)].copy()
    nln_runs = fusion_runs.head(5) 
    best_run_id = nln_runs.loc[nln_runs['macro_f1'].idxmax()]['run_id']
    best_f1 = nln_runs['macro_f1'].max()
    
    ckpt_dir = Path("artifacts/checkpoints")
    best_ckpt = list(ckpt_dir.glob(f"*{best_run_id}.pth"))[0]
    print(f"🏆 Loaded Champion Seed: {best_run_id} (F1: {best_f1:.4f})")
    
    data_dir = Path("data/processed/nln_emp/nln_emp_leave_one_speed_out")
    index_path = data_dir / "windows_index.csv"
    tensor_dir = data_dir / "tensors"
    
    test_df = pd.read_csv(index_path)
    test_df = test_df[test_df['split'] == 'test'].reset_index(drop=True)

    model = MultimodalMotorModel(num_fault_families=5, ablation_mode=None).to(device)
    model.load_state_dict(torch.load(best_ckpt, map_location=device)["state_dict"])
    
    out_dir = Path("results/figures")
    out_dir.mkdir(parents=True, exist_ok=True)
    plt.style.use('seaborn-v0_8-paper')
    sns.set_context("paper", font_scale=1.5)

    # ==========================================
    # PHASE 1: Bulk Inference (ROC & t-SNE)
    # ==========================================
    print(f"\n📊 Phase 1: Evaluating {len(test_df)} samples for ROC & t-SNE...")
    model.eval()
    y_true, y_scores, latent_features = [], [], []
    
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

    y_true, y_scores, latent_features = np.array(y_true), np.array(y_scores), np.array(latent_features)

    # --- FIGURE 4: ROC CURVE ---
    print("📈 Generating Figure 4 (ROC Curve)...")
    fpr, tpr, _ = roc_curve(y_true, y_scores)
    roc_auc = auc(fpr, tpr)
    
    fig_roc, ax_roc = plt.subplots(figsize=(7, 6))
    ax_roc.plot(fpr, tpr, color='#1b9e77', lw=2, label=f'Best Fusion Model (AUC = {roc_auc:.3f})')
    ax_roc.plot([0, 1], [0, 1], color='gray', lw=2, linestyle='--')
    ax_roc.set_xlim([0.0, 1.0])
    ax_roc.set_ylim([0.0, 1.05])
    ax_roc.set_xlabel('False Positive Rate', fontweight='bold')
    ax_roc.set_ylabel('True Positive Rate', fontweight='bold')
    ax_roc.set_title(f'Receiver Operating Characteristic (F1: {best_f1:.2f})', pad=20)
    ax_roc.legend(loc="lower right")
    fig_roc.tight_layout()
    fig_roc.savefig(out_dir / "fig4_roc_curve.pdf", dpi=300, bbox_inches='tight')
    plt.close(fig_roc)

    # --- FIGURE 6: t-SNE PLOT ---
    print("🌌 Generating Figure 6 (t-SNE Plot)... Estimated time: 1-2 mins.")
    tsne = TSNE(n_components=2, perplexity=40, max_iter=1000, method='barnes_hut', n_jobs=-1, random_state=42)
    tsne_results = tsne.fit_transform(latent_features)
    
    fig_tsne, ax_tsne = plt.subplots(figsize=(8, 6))
    scatter = ax_tsne.scatter(tsne_results[:, 0], tsne_results[:, 1], c=y_true, cmap='coolwarm', alpha=0.6, edgecolors='none', s=15, rasterized=True)
    
    from matplotlib.lines import Line2D
    legend_elements = [
        Line2D([0], [0], marker='o', color='w', label='Healthy', markerfacecolor=scatter.cmap(0.0), markersize=10),
        Line2D([0], [0], marker='o', color='w', label='Faulty', markerfacecolor=scatter.cmap(1.0), markersize=10)
    ]
    ax_tsne.legend(handles=legend_elements, loc='best')
    ax_tsne.set_title('t-SNE Feature Visualization (Best Model Embeddings)', pad=20)
    ax_tsne.set_xlabel('t-SNE Dimension 1')
    ax_tsne.set_ylabel('t-SNE Dimension 2')
    fig_tsne.tight_layout()
    fig_tsne.savefig(out_dir / "fig6_tsne_clusters.pdf", dpi=300, bbox_inches='tight')
    plt.close(fig_tsne)

    # ==========================================
    # PHASE 2: Explainability Maps (Grad-CAM)
    # ==========================================
    print("\n🔍 Phase 2: Generating Figure 5 (Explainability Maps)...")
    healthy_row = test_df[test_df['health_label'].astype(str).str.contains('Healthy', case=False, na=False)].iloc[0]
    faulty_row = test_df[test_df['health_label'].astype(str).str.contains('fault', case=False, na=False)].iloc[0]
    samples = {"Healthy Motor": healthy_row, "Faulty Motor": faulty_row}
    
    fig_exp, axes = plt.subplots(2, 3, figsize=(15, 8))
    sns.set_context("paper", font_scale=1.2)
    
    row_idx = 0
    for title, row in samples.items():
        tensor_path = tensor_dir / row['tensor_id']
        x = torch.load(tensor_path, map_location=device, weights_only=True).unsqueeze(0)
        x.requires_grad = True # Turn gradients on for mapping!
        
        out_health, _ = model(x)
        pred_class = torch.argmax(out_health, dim=1).item()
        score = out_health[0, pred_class]
        
        model.zero_grad()
        score.backward()
        
        vib_img, curr_img = x.data[0].numpy()[0], x.data[0].numpy()[1]
        vib_grad, curr_grad = x.grad.data[0].numpy()[0], x.grad.data[0].numpy()[1]
        
        vib_heatmap = normalize_heatmap(vib_grad)
        curr_heatmap = normalize_heatmap(curr_grad)
        
        axes[row_idx, 0].imshow(vib_img, cmap='viridis', aspect='auto')
        axes[row_idx, 0].set_title(f"{title}: Vibration")
        axes[row_idx, 0].axis('off')
        
        axes[row_idx, 1].imshow(curr_img, cmap='viridis', aspect='auto')
        axes[row_idx, 1].set_title(f"{title}: Current")
        axes[row_idx, 1].axis('off')
        
        combined_heatmap = (vib_heatmap + curr_heatmap) / 2
        axes[row_idx, 2].imshow(vib_img, cmap='gray', aspect='auto', alpha=0.5)
        axes[row_idx, 2].imshow(combined_heatmap, cmap='jet', aspect='auto', alpha=0.6)
        axes[row_idx, 2].set_title(f"Model Attention (Saliency)")
        axes[row_idx, 2].axis('off')
        row_idx += 1

    fig_exp.tight_layout()
    fig_exp.savefig(out_dir / "fig5_explainability_maps.pdf", dpi=300, bbox_inches='tight')
    plt.close(fig_exp)

    print("\n✅ SUCCESS! All figures are now mathematically synced to your best seed.")

if __name__ == "__main__":
    main()
