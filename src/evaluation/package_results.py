import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from pathlib import Path
import zipfile
import os

def main():
    print("📦 Starting Step 12: Packaging Final Results...")
    
    # Setup paths
    base_dir = Path("~/data/multimodal-motor-fault-fusion").expanduser()
    tables_dir = base_dir / "results/tables"
    figures_dir = base_dir / "results/figures"
    figures_dir.mkdir(parents=True, exist_ok=True)
    csv_path = tables_dir / "detailed_metrics.csv"
    
    df = pd.read_csv(csv_path)
    
    # 1. Add Dataset Tags (First 20 runs are NLN-EMP, last 5 are Paderborn)
    df['dataset'] = ['nln_emp']*20 + ['paderborn']*5
    
    # 2. Setup Plotting
    plt.style.use('seaborn-v0_8-paper')
    sns.set_context("paper", font_scale=1.5)
    sns.set_palette("deep")
    label_map = {'current_only': 'Current (1D)', 'vibration_only': 'Vibration (1D)', 'fusion': 'Fusion (Proposed)'}
    
    # Figure 1: Modality Ablation (NLN-EMP)
    fig1_df = df[(df['dataset'] == 'nln_emp') & (df['curriculum'] == True)].copy()
    fig1_df['Model'] = fig1_df['ablation'].map(label_map)
    fig1, ax1 = plt.subplots(figsize=(8, 6))
    sns.barplot(data=fig1_df, x='Model', y='macro_f1', capsize=0.1, ax=ax1)
    ax1.set_ylabel('Macro F1 Score')
    ax1.set_xlabel('')
    ax1.set_title('Modality Ablation (NLN-EMP)', pad=15)
    ax1.set_ylim(0.40, 0.75)
    fig1.savefig(figures_dir / 'fig1_modality_ablation.pdf', dpi=300, bbox_inches='tight')
    
    # Figure 2: Curriculum Impact
    fig2_df = df[(df['dataset'] == 'nln_emp') & (df['ablation'] == 'fusion')].copy()
    fig2_df['Training Method'] = fig2_df['curriculum'].map({False: 'Standard Training', True: 'Curriculum (Proposed)'})
    fig2, ax2 = plt.subplots(figsize=(8, 6))
    sns.barplot(data=fig2_df, x='Training Method', y='macro_f1', capsize=0.1, palette=['#d95f02', '#1b9e77'], ax=ax2)
    ax2.set_ylabel('Macro F1 Score')
    ax2.set_xlabel('')
    ax2.set_title('Impact of Curriculum Training', pad=15)
    ax2.set_ylim(0.40, 0.75)
    fig2.savefig(figures_dir / 'fig2_curriculum_impact.pdf', dpi=300, bbox_inches='tight')
    print("✅ PDFs generated.")

    # 3. Generate LaTeX
    summary = df.groupby(['dataset', 'ablation', 'curriculum']).mean(numeric_only=True).reset_index()
    latex_code = """\\begin{table}[h]
\\centering
\\caption{Main Experimental Results across Modalities and Datasets}
\\label{tab:main_results}
\\begin{tabular}{llccc}
\\hline
\\textbf{Dataset} & \\textbf{Architecture} & \\textbf{Curriculum} & \\textbf{Macro F1} & \\textbf{Early Recall} \\\\
\\hline\n"""
    
    for _, row in summary.iterrows():
        d_name = "NLN-EMP" if row['dataset'] == 'nln_emp' else "Paderborn"
        m_name = label_map.get(row['ablation'], "Fusion")
        c_str = "Yes" if row['curriculum'] else "No"
        latex_code += f"{d_name} & {m_name} & {c_str} & {row['macro_f1']:.4f} & {row['early_fault_recall']:.4f} \\\\\n"
        
    latex_code += "\\hline\n\\end{tabular}\n\\end{table}\n"
    
    with open(tables_dir / 'results_latex.tex', 'w') as f:
        f.write(latex_code)
    print("✅ LaTeX code generated.")

    # 4. Zip Everything
    zip_path = base_dir / 'paper_results_archive.zip'
    with zipfile.ZipFile(zip_path, 'w') as zipf:
        for root, _, files in os.walk(tables_dir):
            for file in files:
                zipf.write(os.path.join(root, file), arcname=f"tables/{file}")
        for root, _, files in os.walk(figures_dir):
            for file in files:
                zipf.write(os.path.join(root, file), arcname=f"figures/{file}")
        zipf.write(__file__, arcname="scripts/package_results.py")
        
    print(f"✅ Success! Everything is packed into: {zip_path.name}")

if __name__ == "__main__":
    main()
