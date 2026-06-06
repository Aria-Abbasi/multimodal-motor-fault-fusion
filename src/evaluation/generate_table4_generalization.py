import pandas as pd
from pathlib import Path

def main():
    print("📋 Generating Table 4: Cross-Condition Generalization Matrix...")
    out_dir = Path('results/tables')
    out_dir.mkdir(parents=True, exist_ok=True)
    
    # Load detailed metrics
    df = pd.read_csv(out_dir / 'detailed_metrics.csv')
    df['dataset'] = ['nln_emp']*20 + ['paderborn']*5
    
    # Group and find means
    summary = df.groupby(['dataset', 'ablation', 'curriculum']).mean(numeric_only=True).reset_index()
    
    # Extract specific values for clean formatting
    def get_val(dataset, ablation, curriculum):
        row = summary[(summary['dataset'] == dataset) & 
                      (summary['ablation'] == ablation) & 
                      (summary['curriculum'] == curriculum)]
        return row['macro_f1'].values[0] if len(row) > 0 else 0.0

    nln_curr = get_val('nln_emp', 'current_only', True)
    nln_vib = get_val('nln_emp', 'vibration_only', True)
    nln_fuse_nc = get_val('nln_emp', 'fusion', False)
    nln_fuse_c = get_val('nln_emp', 'fusion', True)
    
    # For Paderborn, we only have the final proposed runs executed
    pad_fuse_c = get_val('paderborn', 'fusion', True)

    latex_code = """\\begin{table}[t]
\\centering
\\caption{Cross-Condition and Dataset Generalization Performance (Macro F1 Score)}
\\label{tab:cross_condition_generalization}
\\begin{tabular}{llcc}
\\hline
\\textbf{Architecture} & \\textbf{Curriculum} & \\textbf{NLN-EMP (Leave-One-Speed-Out)} & \\textbf{Paderborn (Artificial $\\rightarrow$ Natural)} \\\\
\\hline
Current Only (1D) & Yes & """ + f"{nln_curr:.4f} & N/A \\\\\n" + """Vibration Only (1D) & Yes & """ + f"{nln_vib:.4f} & N/A \\\\\n" + """Fusion (Baseline) & No & """ + f"{nln_fuse_nc:.4f} & N/A \\\\\n" + """\\textbf{Fusion (Proposed)} & \\textbf{Yes} & \\textbf{""" + f"{nln_fuse_c:.4f}" + """} & \\textbf{""" + f"{pad_fuse_c:.4f}" + """} \\\\
\\hline
\\end{tabular}
\\end{table}
"""

    with open(out_dir / 'table4_generalization.tex', 'w') as f:
        f.write(latex_code)
        
    print("\n✅ Table 4 LaTeX successfully generated!\n")
    print(latex_code)

if __name__ == "__main__":
    main()
