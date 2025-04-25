import scipy.stats as stats
import pandas as pd, numpy as np
import statsmodels.api as sm
import argparse
import json
from types import SimpleNamespace
from statsmodels.formula.api import ols


def make_chi_squared(df, group_col, target_col):
    # Create a contingency table
    data = pd.crosstab(df[group_col], df[target_col])
    chi2, p, dof, expected = stats.chi2_contingency(data)
    return chi2, p, dof, expected

def make_anova(df, group_col, target_col):
    # Perform ANOVA
    model = ols(f'{target_col} ~ C({group_col})', data=df).fit()
    anova_table = sm.stats.anova_lm(model, typ=2)
    return anova_table




if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument("config", type=str)
    parser.add_argument("--tex_path", type=str, default="comparison.txt")
    args = parser.parse_args()

    with open(args.config) as cfg_file:
        cfg = json.load(cfg_file, object_hook=lambda d: SimpleNamespace(**d))
    
    stats_df = pd.read_csv(cfg.cluster_path, index_col=0)

    cat_groups = [('Cluster', 'Owner type'), 
                ('Cluster', 'CHP'), ('Owner type', 'CHP'), 
                ('Cluster', 'Technology'), ('Owner type', 'Technology'), 
                ('Cluster', 'Free market'), ('Owner type', 'Free market')]

    cont_groups = [('Cluster', 'Age'), ('Cluster', 'Capacity'), ('Cluster', 'Year')]

    
    with open(args.tex_path, "w") as tex_file:

        for group_col, target_col in cat_groups:
            chi2, p, dof, expected = make_chi_squared(stats, group_col=group_col, target_col=target_col)
            alpha = '0.1%' if p <= 0.001 else '1%' if p <= 0.01 else '5%' if p <= 0.05 else 'no'
            text = f"{alpha} significance in {target_col} between {group_col}"
            text += f"\nChi-Square Statistic: {chi2:.3}, p-value: {p:.3}\n--"
            tex_file.write(text)
        
        for group_col, target_col in cont_groups:
            anova_table = make_anova(stats, group_col=group_col, target_col=target_col)
            p = anova_table['PR(>F)'][0]
            alpha = '0.1%' if p <= 0.001 else '1%' if p <= 0.01 else '5%' if p <= 0.05 else 'no'
            text = f"{alpha} significance in {target_col} between {group_col}"
            text += f"\nANOVA p-value: {p:.3}\n--"
            tex_file.write(text)
