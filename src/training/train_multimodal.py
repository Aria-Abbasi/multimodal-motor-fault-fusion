import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader, Dataset
import pandas as pd
import numpy as np
from pathlib import Path
from tqdm import tqdm
import argparse
import random
import os
from sklearn.metrics import f1_score, accuracy_score, recall_score

import sys
sys.path.append(str(Path(__file__).resolve().parents[2]))
from src.models.multimodal_cross_attention import MultimodalMotorModel

def set_seed(seed: int):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False

class MultimodalDataset(Dataset):
    def __init__(self, df, tensor_dir, preload=False):
        self.df = df.reset_index(drop=True)
        self.tensor_dir = Path(tensor_dir)
        self.labels = [1 if 'fault' in str(l).lower() else 0 for l in self.df['health_label']]
        self.families = [0] * len(self.df) 
        self.severities = []
        for _, row in self.df.iterrows():
            if 'severity' in row and pd.notna(row['severity']):
                self.severities.append(str(row['severity']))
            else:
                self.severities.append(str(row['health_label']))
        self.preload = preload
        self.tensors = []
        if self.preload:
            print(f"Loading {len(df)} tensors into System RAM for fast training...")
            for idx in tqdm(range(len(df)), desc="Caching to RAM"):
                t = torch.load(self.tensor_dir / self.df.iloc[idx]['tensor_id'], map_location='cpu', weights_only=True)
                self.tensors.append(t)

    def __len__(self): return len(self.df)
    def __getitem__(self, idx):
        if self.preload: 
            return self.tensors[idx], self.labels[idx], self.families[idx], self.severities[idx]
        t = torch.load(self.tensor_dir / self.df.iloc[idx]['tensor_id'], map_location='cpu', weights_only=True)
        return t, self.labels[idx], self.families[idx], self.severities[idx]

def train_multimodal(args):
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"--- Running Step 10 Curriculum Pipeline on: {device} ---")
    
    index_df = pd.read_csv(Path(args.processed_dir) / "windows_index.csv")
    tensor_dir = Path(args.processed_dir) / "tensors"
    
    if args.smoke_test:
        train_full_df = index_df[index_df['split'] == 'train'].head(50)
        test_full_df = index_df[index_df['split'] == 'test'].head(50)
    else:
        train_full_df = index_df[index_df['split'] == 'train']
        test_full_df = index_df[index_df['split'] == 'test']

    model = MultimodalMotorModel(num_fault_families=5, ablation_mode=args.ablation).to(device)
    optimizer = optim.Adam(model.parameters(), lr=0.001)
    criterion_health = nn.CrossEntropyLoss()
    criterion_family = nn.CrossEntropyLoss()

    # --- STAGE 1 ---
    print("\n" + "="*40)
    print("🎓 STAGE 1: General Pre-training (All Severities)")
    print("="*40)
    train_dataset = MultimodalDataset(train_full_df, tensor_dir, preload=not args.smoke_test)
    loader_kwargs = {'batch_size': 16 if args.smoke_test else 128, 'shuffle': True}
    if not args.smoke_test:
        loader_kwargs.update({'num_workers': 2, 'pin_memory': True})
    train_loader = DataLoader(train_dataset, **loader_kwargs)
    
    model.train()
    stage1_epochs = 1 if args.smoke_test else 10
    for epoch in range(stage1_epochs):
        total_loss = 0
        for batch_x, batch_health, batch_family, _ in train_loader:
            batch_x, batch_health, batch_family = batch_x.to(device), batch_health.to(device), batch_family.to(device)
            optimizer.zero_grad()
            out_health, out_family = model(batch_x)
            loss = criterion_health(out_health, batch_health) + (0.5 * criterion_family(out_family, batch_family))
            loss.backward()
            optimizer.step()
            total_loss += loss.item()
        
        # ADDED THE LOG BACK HERE
        print(f"Stage 1 - Epoch {epoch+1}/{stage1_epochs} | Loss: {total_loss/len(train_loader):.4f}")

    # --- STAGE 2 ---
    if args.use_curriculum:
        print("\n" + "="*40)
        print("🔬 STAGE 2: Weighted Fine-Tuning (Early Fault Focus)")
        print("="*40)
        criterion_weighted = nn.CrossEntropyLoss(reduction='none')
        for param_group in optimizer.param_groups:
            param_group['lr'] = 0.0001
        model.train()
        stage2_epochs = 1 if args.smoke_test else 5
        for epoch in range(stage2_epochs):
            total_loss = 0
            for batch_x, batch_health, batch_family, batch_sev in train_loader:
                batch_x, batch_health, batch_family = batch_x.to(device), batch_health.to(device), batch_family.to(device)
                optimizer.zero_grad()
                out_health, out_family = model(batch_x)
                raw_loss_h = criterion_weighted(out_health, batch_health)
                weights = torch.tensor([5.0 if (s == '1' or s in ['01', '05', '07'] or '007' in s or '0.007' in s or 'early' in s.lower()) else 1.0 for s in batch_sev]).to(device)
                loss_h = (raw_loss_h * weights).mean()
                loss_f = criterion_family(out_family, batch_family)
                loss = loss_h + (0.5 * loss_f)
                loss.backward()
                optimizer.step()
                total_loss += loss.item()
            
            # ADDED THE LOG BACK HERE
            print(f"Stage 2 - Epoch {epoch+1}/{stage2_epochs} | Weighted Loss: {total_loss/len(train_loader):.4f}")

    # --- EVALUATION AND SAVING BLOCK (NEW) ---
    print("\n🔬 Grading the Model on the Test Set...")
    model.eval()
    test_dataset = MultimodalDataset(test_full_df, tensor_dir, preload=False)
    test_loader = DataLoader(test_dataset, batch_size=128, shuffle=False)
    
    all_preds = []
    all_targets = []
    all_sevs = []
    
    with torch.no_grad():
        for batch_x, batch_health, _, batch_sev in test_loader:
            batch_x = batch_x.to(device)
            out_health, _ = model(batch_x)
            preds = torch.argmax(out_health, dim=1).cpu().numpy()
            all_preds.extend(preds)
            all_targets.extend(batch_health.numpy())
            all_sevs.extend(batch_sev)
            
    macro_f1 = f1_score(all_targets, all_preds, average='macro')
    acc = accuracy_score(all_targets, all_preds)
    
    # Early fault recall calculation
    early_indices = [i for i, s in enumerate(all_sevs) if s == '1' or s in ['01', '05', '07'] or '007' in s or '0.007' in s or 'early' in s.lower()]
    if len(early_indices) > 0:
        early_targets = [all_targets[i] for i in early_indices]
        early_preds = [all_preds[i] for i in early_indices]
        early_recall = recall_score(early_targets, early_preds, zero_division=0)
    else:
        early_recall = 0.0

    print(f"📊 Test Macro F1: {macro_f1:.4f} | Early Recall: {early_recall:.4f}")

    # Save to dedicated CSV
    import uuid
    run_hash = str(uuid.uuid4())[:6]
    ablation_str = args.ablation if args.ablation else "fusion"
    
    metrics_file = Path("results/tables/detailed_metrics.csv")
    metrics_file.parent.mkdir(parents=True, exist_ok=True)
    
    new_row = pd.DataFrame([{
        "run_id": run_hash,
        "ablation": ablation_str,
        "curriculum": args.use_curriculum,
        "macro_f1": macro_f1,
        "balanced_acc": acc,
        "early_fault_recall": early_recall
    }])
    
    if metrics_file.exists():
        new_row.to_csv(metrics_file, mode='a', header=False, index=False)
    else:
        new_row.to_csv(metrics_file, mode='w', header=True, index=False)

    # Save the actual model
    ckpt_dir = Path("artifacts/checkpoints")
    ckpt_dir.mkdir(parents=True, exist_ok=True)
    ckpt_path = ckpt_dir / f"model_{ablation_str}_curr{args.use_curriculum}_{run_hash}.pth"
    torch.save({"state_dict": model.state_dict(), "metrics": {"macro_f1": macro_f1}}, ckpt_path)
    
    print("✅ Run Complete. Metrics and Checkpoint Saved!")

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--processed_dir", type=str, required=True)
    parser.add_argument("--use_curriculum", action="store_true")
    parser.add_argument("--ablation", type=str, default=None, choices=["vibration_only", "current_only"])
    parser.add_argument("--smoke_test", action="store_true")
    os.environ['PYTHONPATH'] = str(Path(__file__).resolve().parents[2])
    train_multimodal(parser.parse_args())
