"""Replot archived metrics with correct dates and explicit historical status."""
from pathlib import Path

import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import pandas as pd

ROOT = Path(__file__).resolve().parents[1]


def main():
    run = ROOT / 'results/05-14-2026-1407'
    df = pd.read_csv(run / 'summary_table.csv')
    styles = {
        'PCA': ('#56758b', 'o'), 'IPCA': ('#c78536', 's'),
        'ResCAE-Fixed': ('#8866aa', 'D'), 'ResCAE': ('#217567', 'o'),
        'CAE-NL (robustness)': ('#999999', '^'),
    }
    plt.rcParams.update({'font.size': 10, 'axes.spines.top': False, 'axes.spines.right': False})
    fig, axes = plt.subplots(1, 2, figsize=(12, 5.4))
    for model, (color, marker) in styles.items():
        rows = df[df.Model == model]
        for ax, metric, scale in zip(axes, ['Pred_R2', 'Sharpe'], [100, 1]):
            values = pd.to_numeric(rows[metric], errors='coerce') * scale
            if values.notna().any():
                ax.plot(rows.K, values, marker=marker, color=color, lw=2,
                        label=model.replace(' (robustness)', ''), markersize=5)
    for ax in axes:
        ax.axhline(0, color='#777777', lw=0.7)
        ax.axvline(5, color='#217567', lw=0.8, ls=':', alpha=0.7)
        ax.set_xticks([2, 3, 5, 8, 10])
        ax.set_xlabel('Number of factors, K')
        ax.grid(axis='y', alpha=0.18)
    axes[0].set(title='Return forecast accuracy', ylabel='Predictive R² (%)')
    axes[1].set(title='Monthly decile-spread portfolio', ylabel='Annualized Sharpe')
    handles, labels = axes[0].get_legend_handles_labels()
    fig.legend(handles, labels, loc='upper center', bbox_to_anchor=(0.5, 0.9),
               ncol=5, frameon=False, fontsize=10)
    axes[1].text(0.03, 0.48, 'CAE-NL: flat predictions; Sharpe undefined',
                 transform=axes[1].transAxes, fontsize=9, color='#555555')
    fig.suptitle('Archived May 2026 experiment · test period 2015–2024', fontsize=15, y=0.99)
    fig.text(0.5, 0.92, 'Dotted line: validation-selected ResCAE K = 5. Each family uses its selected penalties at each K.',
             ha='center', fontsize=10)
    fig.text(0.5, 0.015,
             'Original metrics, before audit corrections. ResCAE and CAE-NL: five seeds; Fixed: one seed. See docs/results_audit.md.',
             ha='center', fontsize=9, color='#555555')
    fig.tight_layout(rect=[0, 0.06, 1, 0.89])
    output = ROOT / 'docs/figures/historical_results.png'
    output.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(output, dpi=180)
    plt.close(fig)
    print(output)


if __name__ == '__main__':
    main()
