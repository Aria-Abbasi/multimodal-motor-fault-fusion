import torch
import time
from pathlib import Path
import sys

# Ensure local imports work
sys.path.append(str(Path(__file__).resolve().parents[2]))
from src.models.multimodal_cross_attention import MultimodalMotorModel

def count_parameters(model):
    return sum(p.numel() for p in model.parameters() if p.requires_grad)

def measure_inference_time(model, dummy_input):
    model.eval()
    # Warmup
    with torch.no_grad():
        for _ in range(5):
            model(dummy_input)
    
    # Measure
    start_time = time.time()
    with torch.no_grad():
        for _ in range(50):
            model(dummy_input)
    end_time = time.time()
    
    # Return time per sample in milliseconds (assuming batch size 1)
    return ((end_time - start_time) / 50) * 1000

def main():
    print("⚙️ Generating Table 1 (Dataset Summary) and Table 5 (Complexity)...")
    out_dir = Path('results/tables')
    out_dir.mkdir(parents=True, exist_ok=True)

    # --- Generate Table 1 (Dataset Summary) ---
    table1_latex = """\\begin{table}[h]
\\centering
\\caption{Summary of Evaluated Motor Fault Datasets}
\\label{tab:datasets}
\\begin{tabular}{lcccc}
\\hline
\\textbf{Dataset} & \\textbf{Available Sensors} & \\textbf{Fault Families} & \\textbf{Generalization Protocol} \\\\
\\hline
NLN-EMP & Vibration, Current & 5 & Leave-One-Speed-Out \\\\
Paderborn (PU) & Vibration, Current & 3 & Artificial to Natural \\\\
CWRU & Vibration Only & 4 & Leave-One-Load-Out \\\\
\\hline
\\end{tabular}
\\end{table}
"""
    with open(out_dir / 'table1_datasets.tex', 'w') as f:
        f.write(table1_latex)

    # --- Generate Table 5 (Complexity) ---
    # Create dummy tensors for 1 sample (1 channel, 4096 sequence length typical for windows)
    dummy_input = torch.randn(1, 2, 64, 64)
    
    models = {
        "Fusion (Proposed)": MultimodalMotorModel(num_fault_families=5, ablation_mode=None),
        "Vibration Only (1D)": MultimodalMotorModel(num_fault_families=5, ablation_mode="vibration_only"),
        "Current Only (1D)": MultimodalMotorModel(num_fault_families=5, ablation_mode="current_only")
    }

    table5_latex = """\\begin{table}[h]
\\centering
\\caption{Model Complexity and CPU Inference Time per Sample}
\\label{tab:complexity}
\\begin{tabular}{lcc}
\\hline
\\textbf{Architecture} & \\textbf{Parameters (Millions)} & \\textbf{Inference Time (ms)} \\\\
\\hline\n"""

    for name, model in models.items():
        params = count_parameters(model) / 1_000_000  # Convert to Millions
        inf_time = measure_inference_time(model, dummy_input)
        table5_latex += f"{name} & {params:.2f}M & {inf_time:.2f} ms \\\\\n"

    table5_latex += "\\hline\n\\end{tabular}\n\\end{table}\n"
    
    with open(out_dir / 'table5_complexity.tex', 'w') as f:
        f.write(table5_latex)

    print("\n✅ Table 1 and Table 5 LaTeX generated!")
    print(table5_latex)

if __name__ == "__main__":
    main()
