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
from sklearn.metrics import f1_score, balanced_accuracy_score, recall_score, roc_auc_score

# --- Seed Control ---
def set_seed(seed: int):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False

# --- Architectures ---

class Simple1DCNN(nn.Module):
    def __init__(self, in_channels=2, num_classes=2):
        super().__init__()
        self.conv = nn.Sequential(
            nn.Conv1d(in_channels, 16, kernel_size=64, stride=8),
            nn.BatchNorm1d(16),
            nn.ReLU(),
            nn.MaxPool1d(4),
            nn.Conv1d(16, 32, kernel_size=16, stride=4),
            nn.BatchNorm1d(32),
            nn.ReLU(),
            nn.AdaptiveAvgPool1d(1)
        )
        self.fc = nn.Linear(32, num_classes)

    def forward(self, x):
        x = self.conv(x)
        return self.fc(x.squeeze(-1))

class SimpleLSTM(nn.Module):
    def __init__(self, input_dim=256, hidden_dim=64, num_classes=2):
        super().__init__()
        self.lstm = nn.LSTM(input_dim, hidden_dim, batch_first=True)
        self.fc = nn.Linear(hidden_dim, num_classes)

    def forward(self, x):
        batch_size = x.size(0)
        x = x.permute(0, 2, 1, 3).reshape(batch_size, 128, -1)
        _, (hn, _) = self.lstm(x)
        return self.fc(hn[-1])

# --- Dataset Logic (Upgraded for 16GB RAM) ---

class MotorDataset(Dataset):
    def __init__(self, df, tensor_dir, preload=False):
        self.df = df
        self.tensor_dir = Path(tensor_dir)
        self.labels = [1 if 'fault' in str(l).lower() else 0 for l in df['health_label']]
        self.preload = preload
        self.tensors = []
        
        # --- NEW: RAM Caching ---
        if self.preload:
            print(f"Loading {len(df)} tensors into System RAM for fast training...")
            for idx in tqdm(range(len(df)), desc="Caching to RAM"):
                row = self.df.iloc[idx]
                t = torch.load(self.tensor_dir / row['tensor_id'], map_location='cpu', weights_only=True)
                self.tensors.append(t)

    def __len__(self): return len(self.df)
    def __getitem__(self, idx):
        if self.preload:
            return self.tensors[idx], self.labels[idx]
        
        # Fallback for lazy-loading (e.g., val/test set)
        row = self.df.iloc[idx]
        t = torch.load(self.tensor_dir / row['tensor_id'], map_location='cpu', weights_only=True)
        return t, self.labels[idx]

# --- Training and Evaluation ---

def train_one_epoch(model, loader, optimizer, criterion, device, model_type):
    model.train()
    total_loss = 0
    for batch_x, batch_y in loader:
        batch_x, batch_y = batch_x.to(device), batch_y.to(device)
        optimizer.zero_grad()
        inputs = batch_x.view(batch_x.size(0), 2, -1) if model_type == "cnn" else batch_x
        outputs = model(inputs)
        loss = criterion(outputs, batch_y)
        loss.backward()
        optimizer.step()
        total_loss += loss.item()
    return total_loss / len(loader)

@torch.no_grad()
def evaluate(model, loader, device, model_type):
    model.eval()
    all_preds, all_probs, all_y = [], [], []
    for batch_x, batch_y in loader:
        batch_x = batch_x.to(device)
        inputs = batch_x.view(batch_x.size(0), 2, -1) if model_type == "cnn" else batch_x
        outputs = model(inputs)
        probs = torch.softmax(outputs, dim=1)[:, 1]
        preds = torch.argmax(outputs, dim=1)
        
        all_preds.extend(preds.cpu().numpy())
        all_probs.extend(probs.cpu().numpy())
        all_y.extend(batch_y.numpy())
    
    return {
        'macro_f1': f1_score(all_y, all_preds, average='macro'),
        'balanced_acc': balanced_accuracy_score(all_y, all_preds),
        'early_fault_recall': recall_score(all_y, all_preds),
        'auroc': roc_auc_score(all_y, all_probs) if len(set(all_y)) > 1 else np.nan
    }

def run_experiment(args):
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"--- Running on: {device} ---")
    
    index_df = pd.read_csv(Path(args.processed_dir) / "windows_index.csv")
    tensor_dir = Path(args.processed_dir) / "tensors"
    
    Path("artifacts/checkpoints").mkdir(parents=True, exist_ok=True)
    out_path = Path("results/tables/deep_baselines.csv")
    out_path.parent.mkdir(parents=True, exist_ok=True)
    
    latest_ckpt_path = Path(f"artifacts/checkpoints/{args.dataset}_{args.model}_latest.pth")
    best_ckpt_path = Path(f"artifacts/checkpoints/{args.dataset}_{args.model}_best.pth")

    train_full = index_df[index_df['split'] == 'train']
    val_df = index_df[index_df['split'] == 'val'].sample(n=min(len(index_df[index_df['split'] == 'val']), 2000), random_state=42)
    test_df = index_df[index_df['split'] == 'test'].sample(n=min(len(index_df[index_df['split'] == 'test']), 5000), random_state=42)

    seeds = [42, 123, 999, 7, 88] if args.full_run else [42]
    
    # --- SPOT RESUME LOGIC ---
    resume_state = None
    if latest_ckpt_path.exists():
        print(f"🔄 Spot VM Preemption Detected! Loading state from {latest_ckpt_path}")
        resume_state = torch.load(latest_ckpt_path, map_location=device, weights_only=False)

    for seed_idx, seed in enumerate(seeds):
        # Skip seeds that are already completely finished
        if resume_state and seed_idx < resume_state['seed_idx']:
            continue

        print(f"\n🌱 Running Seed: {seed} ({seed_idx + 1}/{len(seeds)})")
        set_seed(seed)
        
        # --- Bulletproof Manual Balanced Subsampling ---
        sampled_dfs = []
        for label in train_full['health_label'].unique():
            class_df = train_full[train_full['health_label'] == label]
            sampled_dfs.append(class_df.sample(n=min(len(class_df), 10000), random_state=seed))
        
        train_df = pd.concat(sampled_dfs).sample(frac=1, random_state=seed).reset_index(drop=True)
        
        # --- NEW: High-Performance DataLoaders ---
        # Preload the training set, but leave Val/Test lazy to save a bit of RAM
        train_dataset = MotorDataset(train_df, tensor_dir, preload=True)
        val_dataset = MotorDataset(val_df, tensor_dir, preload=False)
        test_dataset = MotorDataset(test_df, tensor_dir, preload=False)

        # Batch size 256 uses the 24GB VRAM efficiently.
        # num_workers=2 uses multiple CPU cores.
        # pin_memory=True speeds up CPU-to-GPU data transfer.
        train_loader = DataLoader(train_dataset, batch_size=256, shuffle=True, num_workers=2, pin_memory=True)
        val_loader = DataLoader(val_dataset, batch_size=256, num_workers=2, pin_memory=True)
        test_loader = DataLoader(test_dataset, batch_size=256, num_workers=2, pin_memory=True)

        model = Simple1DCNN().to(device) if args.model == "cnn" else SimpleLSTM().to(device)
        optimizer = optim.Adam(model.parameters(), lr=0.001)
        criterion = nn.CrossEntropyLoss()
        
        best_val_f1 = 0
        patience, counter = 5, 0 
        start_epoch = 0

        # Inject Spot Resume State
        if resume_state and seed_idx == resume_state['seed_idx']:
            print(f"   -> Restoring model & optimizer state from Epoch {resume_state['epoch']}...")
            model.load_state_dict(resume_state['model_state_dict'])
            optimizer.load_state_dict(resume_state['optimizer_state_dict'])
            best_val_f1 = resume_state['best_val_f1']
            counter = resume_state['counter']
            start_epoch = resume_state['epoch'] + 1
            resume_state = None # Clear it so subsequent seeds start fresh

        for epoch in range(start_epoch, args.epochs):
            loss = train_one_epoch(model, train_loader, optimizer, criterion, device, args.model)
            metrics = evaluate(model, val_loader, device, args.model)
            print(f"Epoch {epoch+1} | Loss: {loss:.4f} | Val F1: {metrics['macro_f1']:.4f}")
            
            if metrics['macro_f1'] > best_val_f1:
                best_val_f1 = metrics['macro_f1']
                torch.save(model.state_dict(), best_ckpt_path)
                counter = 0
            else:
                counter += 1
            
            # Save Latest State for Spot VMs
            torch.save({
                'seed_idx': seed_idx,
                'epoch': epoch,
                'model_state_dict': model.state_dict(),
                'optimizer_state_dict': optimizer.state_dict(),
                'best_val_f1': best_val_f1,
                'counter': counter
            }, latest_ckpt_path)

            if counter >= patience: 
                print(f"Early stopping triggered at epoch {epoch+1}")
                break

        # Final Test Evaluation for this Seed
        model.load_state_dict(torch.load(best_ckpt_path, map_location=device, weights_only=True))
        test_results = evaluate(model, test_loader, device, args.model)
        test_results.update({'seed': seed, 'dataset': args.dataset, 'model': args.model})
        
        # Immediate CSV Flush (Spot Safe)
        res_df = pd.DataFrame([test_results])
        if out_path.exists():
            res_df = pd.concat([pd.read_csv(out_path), res_df], ignore_index=True)
        res_df.to_csv(out_path, index=False)
        print(f"✅ Seed {seed} completed and saved to CSV.")

    # Cleanup the spot resume file once the whole script finishes successfully
    if latest_ckpt_path.exists():
        os.remove(latest_ckpt_path)
    print(f"\n🎉 ALL RUNS COMPLETE for {args.dataset} {args.model.upper()}")

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--processed_dir", type=str, required=True)
    parser.add_argument("--dataset", type=str, required=True)
    parser.add_argument("--model", type=str, choices=["cnn", "lstm"], required=True)
    parser.add_argument("--epochs", type=int, default=20)
    parser.add_argument("--full_run", action="store_true")
    run_experiment(parser.parse_args())
