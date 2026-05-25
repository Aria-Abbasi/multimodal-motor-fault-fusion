import argparse
import pandas as pd
import numpy as np
import torch
import joblib
from pathlib import Path
from tqdm import tqdm
from scipy.stats import kurtosis, skew
from sklearn.ensemble import RandomForestClassifier
from sklearn.svm import SVC
from sklearn.metrics import f1_score, balanced_accuracy_score, recall_score, roc_auc_score

def extract_tensor_features(tensor: torch.Tensor) -> np.ndarray:
    """Extracts 20 spectral-statistical features (10 per channel)."""
    feats = []
    for i in range(2): 
        spec = tensor[i].numpy().flatten()
        feats.extend([
            np.mean(spec), np.std(spec), np.max(spec), np.min(spec),
            kurtosis(spec), skew(spec), np.median(spec),
            np.percentile(spec, 75) - np.percentile(spec, 25),
            np.sqrt(np.mean(spec**2)),
            np.sum(np.abs(spec) > np.mean(spec)) / len(spec)
        ])
    return np.array(feats)

def run_experiment(processed_dir: Path, dataset_name: str, sample_size: int = 5000):
    index_df = pd.read_csv(processed_dir / "windows_index.csv")
    tensor_dir = processed_dir / "tensors"
    
    print(f"DEBUG: Initial Columns: {index_df.columns.tolist()}")

    # 1. Manually Subsample to avoid column loss
    train_full = index_df[index_df['split'] == 'train']
    
    sampled_dfs = []
    for label in train_full['health_label'].unique():
        class_df = train_full[train_full['health_label'] == label]
        sampled_dfs.append(class_df.sample(n=min(len(class_df), sample_size), random_state=42))
    
    train_sub = pd.concat(sampled_dfs).sample(frac=1, random_state=42).reset_index(drop=True)
    
    test_df = index_df[index_df['split'] == 'test']
    if len(test_df) > 5000:
        test_df = test_df.sample(n=5000, random_state=42).reset_index(drop=True)

    def get_features(df, desc):
        X, y = [], []
        print(f"DEBUG: Processing {desc} with columns: {df.columns.tolist()}")
        
        for _, row in tqdm(df.iterrows(), total=len(df), desc=desc):
            tensor_path = tensor_dir / row['tensor_id']
            if not tensor_path.exists(): continue
            
            tensor = torch.load(tensor_path, map_location='cpu', weights_only=True)
            X.append(extract_tensor_features(tensor))
            
            # Map labels to binary
            lbl = str(row['health_label']).lower()
            y.append(1 if 'fault' in lbl else 0)
            
        return np.array(X), np.array(y)

    print(f"\n--- Establishing Classical Baseline for {dataset_name} ---")
    X_train, y_train = get_features(train_sub, "Processing Train")
    X_test, y_test = get_features(test_df, "Processing Test")

    results = []
    models = {
        "RandomForest": RandomForestClassifier(n_estimators=100, n_jobs=-1, random_state=42),
        "SVM": SVC(probability=True, kernel='rbf', random_state=42)
    }

    for name, model in models.items():
        print(f"Training {name}...")
        model.fit(X_train, y_train)
        y_pred = model.predict(X_test)
        y_prob = model.predict_proba(X_test)[:, 1]
        
        # Calculate paper-critical metrics
        results.append({
            'dataset': dataset_name,
            'model': name,
            'macro_f1': f1_score(y_test, y_pred, average='macro'),
            'balanced_acc': balanced_accuracy_score(y_test, y_pred),
            'early_fault_recall': recall_score(y_test, y_pred),
            'auroc': roc_auc_score(y_test, y_prob)
        })
    
    return results

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--processed_dir", type=str, required=True)
    parser.add_argument("--dataset", type=str, required=True)
    args = parser.parse_args()
    
    res = run_experiment(Path(args.processed_dir), args.dataset)
    
    out_file = Path("results/tables/classical_baselines.csv")
    out_file.parent.mkdir(parents=True, exist_ok=True)
    
    new_df = pd.DataFrame(res)
    if out_file.exists():
        existing = pd.read_csv(out_file)
        existing = existing[existing['dataset'] != args.dataset]
        new_df = pd.concat([existing, new_df], ignore_index=True)
    
    new_df.to_csv(out_file, index=False)
    print(f"\n✅ Results updated in {out_file}\n")
    print(new_df[new_df['dataset'] == args.dataset])
