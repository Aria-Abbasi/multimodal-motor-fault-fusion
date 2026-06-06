import pandas as pd
from scipy import stats
import numpy as np

def main():
    print("📊 Calculating Statistical Significance (Table 6)...")
    
    # Load the data
    df = pd.read_csv('results/tables/detailed_metrics.csv')
    
    # We only care about NLN-EMP for the ablation comparison (the first 20 rows)
    nln_df = df.iloc[:20].copy()
    
    # Extract the F1 scores for the 5 seeds of each architecture (Curriculum = True)
    fusion_f1 = nln_df[(nln_df['ablation'] == 'fusion') & (nln_df['curriculum'] == True)]['macro_f1'].values
    vib_f1 = nln_df[(nln_df['ablation'] == 'vibration_only') & (nln_df['curriculum'] == True)]['macro_f1'].values
    curr_f1 = nln_df[(nln_df['ablation'] == 'current_only') & (nln_df['curriculum'] == True)]['macro_f1'].values
    fusion_no_curr_f1 = nln_df[(nln_df['ablation'] == 'fusion') & (nln_df['curriculum'] == False)]['macro_f1'].values

    # Calculate Mean ± Std
    results = {
        "Current (1D)": (np.mean(curr_f1), np.std(curr_f1)),
        "Vibration (1D)": (np.mean(vib_f1), np.std(vib_f1)),
        "Fusion (No Curr.)": (np.mean(fusion_no_curr_f1), np.std(fusion_no_curr_f1)),
        "Fusion (Proposed)": (np.mean(fusion_f1), np.std(fusion_f1))
    }

    # Run Independent T-Tests (Comparing against Proposed)
    p_vib = stats.ttest_ind(fusion_f1, vib_f1)[1]
    p_curr = stats.ttest_ind(fusion_f1, curr_f1)[1]
    p_no_curr = stats.ttest_ind(fusion_f1, fusion_no_curr_f1)[1]

    # Generate LaTeX Table
    latex_code = """\\begin{table}[h]
\\centering
\\caption{Statistical Significance of Proposed Fusion Model vs. Baselines (NLN-EMP Dataset, 5 Seeds)}
\\label{tab:statistical_significance}
\\begin{tabular}{lcc}
\\hline
\\textbf{Architecture} & \\textbf{Macro F1 (Mean $\\pm$ SD)} & \\textbf{$p$-value (vs. Proposed)} \\\\
\\hline\n"""
    
    latex_code += f"Current (1D) & {results['Current (1D)'][0]:.4f} $\\pm$ {results['Current (1D)'][1]:.4f} & {p_curr:.4e} \\\\\n"
    latex_code += f"Vibration (1D) & {results['Vibration (1D)'][0]:.4f} $\\pm$ {results['Vibration (1D)'][1]:.4f} & {p_vib:.4e} \\\\\n"
    latex_code += f"Fusion (No Curriculum) & {results['Fusion (No Curr.)'][0]:.4f} $\\pm$ {results['Fusion (No Curr.)'][1]:.4f} & {p_no_curr:.4e} \\\\\n"
    latex_code += f"\\textbf{{Fusion (Proposed)}} & \\textbf{{{results['Fusion (Proposed)'][0]:.4f} $\\pm$ {results['Fusion (Proposed)'][1]:.4f}}} & \\textbf{{-}} \\\\\n"
    
    latex_code += "\\hline\n\\end{tabular}\n\\end{table}\n"

    # Save to file and print
    with open('results/tables/table6_stats.tex', 'w') as f:
        f.write(latex_code)
    
    print("\n✅ Table 6 LaTeX successfully generated!\n")
    print(latex_code)

if __name__ == "__main__":
    main()
