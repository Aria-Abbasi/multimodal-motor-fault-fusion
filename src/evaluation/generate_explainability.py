import torch
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from pathlib import Path
from scipy.ndimage import gaussian_filter
import sys

sys.path.append(str(Path(__file__).resolve().parents[2]))
from src.models.multimodal_cross_attention import MultimodalMotorModel

def normalize_heatmap(grad):
    # Absolute value, then scale 0 to 1
    heatmap = np.abs(grad)
    heatmap = heatmap - np.min(heatmap)
    heatmap = heatmap / (np.max(heatmap) + 1e-8)
    # Apply a slight Gaussian blur so it looks like a smooth Grad-CAM
    heatmap = gaussian_filter(heatmap, sigma=2.0)
    return heatmap

def main():
    print("🔍 Generating Figure 5: Explainability Heatmaps...")
    device = torch.device("cpu")
    
    ckpt_dir = Path("artifacts/checkpoints")
    best_ckpt = list(ckpt_dir.glob("model_fusion_currTrue_*.pth"))[0]
    
    data_dir = Path("data/processed/nln_emp/nln_emp_leave_one_speed_out")
    index_path = data_dir / "windows_index.csv"
    tensor_dir = data_dir / "tensors"
    
    df = pd.read_csv(index_path)
    test_df = df[df['split'] == 'test'].reset_index(drop=True)
    
    # Find 1 Healthy and 1 Early Fault sample
    healthy_row = test_df[test_df['health_label'].astype(str).str.contains('Healthy', case=False, na=False)].iloc[0]
    faulty_row = test_df[test_df['health_label'].astype(str).str.contains('fault', case=False, na=False)].iloc[0]
    
    samples = {"Healthy Motor": healthy_row, "Faulty Motor": faulty_row}
    
    model = MultimodalMotorModel(num_fault_families=5, ablation_mode=None).to(device)
    model.load_state_dict(torch.load(best_ckpt, map_location=device)["state_dict"])
    model.eval()
    
    fig, axes = plt.subplots(2, 3, figsize=(15, 8))
    plt.style.use('seaborn-v0_8-paper')
    sns.set_context("paper", font_scale=1.2)
    
    row_idx = 0
    for title, row in samples.items():
        tensor_path = tensor_dir / row['tensor_id']
        x = torch.load(tensor_path, map_location=device, weights_only=True).unsqueeze(0)
        
        # Enable gradients for the input image!
        x.requires_grad = True
        
        out_health, _ = model(x)
        
        # Get the score of the predicted class
        pred_class = torch.argmax(out_health, dim=1).item()
        score = out_health[0, pred_class]
        
        # Backpropagate to the input image
        model.zero_grad()
        score.backward()
        
        # Extract gradients and original images
        grads = x.grad.data[0].numpy()
        imgs = x.data[0].numpy()
        
        vib_img, curr_img = imgs[0], imgs[1]
        vib_grad, curr_grad = grads[0], grads[1]
        
        # Create heatmaps
        vib_heatmap = normalize_heatmap(vib_grad)
        curr_heatmap = normalize_heatmap(curr_grad)
        
        # 1. Plot Vibration Input
        axes[row_idx, 0].imshow(vib_img, cmap='viridis', aspect='auto')
        axes[row_idx, 0].set_title(f"{title}: Vibration")
        axes[row_idx, 0].axis('off')
        
        # 2. Plot Current Input
        axes[row_idx, 1].imshow(curr_img, cmap='viridis', aspect='auto')
        axes[row_idx, 1].set_title(f"{title}: Current")
        axes[row_idx, 1].axis('off')
        
        # 3. Plot Explainability Overlay (Combining Both Heatmaps for visualization)
        combined_heatmap = (vib_heatmap + curr_heatmap) / 2
        axes[row_idx, 2].imshow(vib_img, cmap='gray', aspect='auto', alpha=0.5)
        im = axes[row_idx, 2].imshow(combined_heatmap, cmap='jet', aspect='auto', alpha=0.6)
        axes[row_idx, 2].set_title(f"Model Attention (Saliency)")
        axes[row_idx, 2].axis('off')
        
        row_idx += 1

    plt.tight_layout()
    out_dir = Path("results/figures")
    out_dir.mkdir(parents=True, exist_ok=True)
    fig.savefig(out_dir / "fig5_explainability_maps.pdf", dpi=300, bbox_inches='tight')
    print("✅ Saved fig5_explainability_maps.pdf")

if __name__ == "__main__":
    main()
