import argparse
import pandas as pd
from pathlib import Path
import sys
import os

# Ensure local imports work by adding project root to path
sys.path.append(str(Path(__file__).resolve().parents[2]))
from src.training.train_multimodal import train_multimodal

def run_experiment_matrix(args):
    # 1. THE FULL L4 GPU MATRIX
    experiments = [
        {"name": "E1_Main_Fusion", "dataset": "nln_emp", "folder": "nln_emp/nln_emp_leave_one_speed_out", "ablation": None, "curriculum": True},
        {"name": "E2_Vib_Only", "dataset": "nln_emp", "folder": "nln_emp/nln_emp_leave_one_speed_out", "ablation": "vibration_only", "curriculum": True},
        {"name": "E3_Curr_Only", "dataset": "nln_emp", "folder": "nln_emp/nln_emp_leave_one_speed_out", "ablation": "current_only", "curriculum": True},
        {"name": "E4_No_Curriculum", "dataset": "nln_emp", "folder": "nln_emp/nln_emp_leave_one_speed_out", "ablation": None, "curriculum": False},
        {"name": "E5_Paderborn_Robustness", "dataset": "paderborn", "folder": "paderborn/paderborn_condition_generalization", "ablation": None, "curriculum": True},
    ]

    out_path = Path("results/tables/main_results.csv")
    out_path.parent.mkdir(parents=True, exist_ok=True)

    # Use 5 seeds for the real run, 1 seed if smoke testing
    seeds = [42, 123, 999, 7, 88] if not args.smoke_test else [42]

    for exp in experiments:
        print(f"\n" + "="*50)
        print(f"🚀 STARTING EXPERIMENT: {exp['name']}")
        print("="*50)
        
        for seed in seeds:
            # 2. SEED-LEVEL SPOT-SAFE RESUME
            if out_path.exists():
                existing = pd.read_csv(out_path)
                # Check if this specific experiment + seed combo is already completed
                if not existing[(existing['experiment'] == exp['name']) & (existing['seed'] == seed)].empty:
                    print(f"⏭  Skipping {exp['name']} - Seed {seed} (Already found in CSV)")
                    continue
            
            print(f"\n🌱 Running Seed {seed}...")

            # 3. FUNCTIONAL EXECUTION (Calls your Step 10 Code)
            exp_args = argparse.Namespace(
                processed_dir=str(Path(args.data_root) / exp['folder']),
                use_curriculum=exp['curriculum'],
                ablation=exp['ablation'],
                smoke_test=args.smoke_test, 
                epochs=20 # Set this to your desired full training length
            )
            
            # Execute the actual training loop
            train_multimodal(exp_args)
            
            # 4. LOG THE COMPLETION
            result = {
                "experiment": exp['name'],
                "dataset": exp['dataset'],
                "seed": seed,
                "status": "COMPLETED",
                "curriculum": exp['curriculum']
            }
            
            # Spot-Safe Flush: Append and save immediately
            new_row = pd.DataFrame([result])
            if out_path.exists():
                res_df = pd.concat([pd.read_csv(out_path), new_row], ignore_index=True)
            else:
                res_df = new_row
                
            res_df.to_csv(out_path, index=False)
            print(f"✅ {exp['name']} (Seed {seed}) banked in CSV.")

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--data_root", type=str, default="data/processed")
    parser.add_argument("--smoke_test", action="store_true", help="Run safe limits for testing")
    run_experiment_matrix(parser.parse_args())
