#!/usr/bin/env python3
"""
Plot comparison of Random 70/30 split vs LOSO F1 scores for Ninapro.
Shows the gap between within-subject (random) and cross-subject (LOSO) performance.
"""

import matplotlib.pyplot as plt
import numpy as np

# Data from experiments (ws=400)
models = ['LDA', 'LogReg', 'NaiveBayes', 'LinearSVM']

# Random 70/30 split results (from results_ninapro_full_LDA-LinearSVM-LogReg-NaiveBayes_random_20251228_183242.csv)

random_f1 = {
    'LDA': 0.7312,
    'LogReg': 0.9258,
    'NaiveBayes': 0.4690,
    'LinearSVM': 0.9063,
}

# LOSO results (from results_ninapro_full_LDA-LinearSVM-LogReg-NaiveBayes_loso_20251227_212423.csv)
# Average across 5 folds, with PCA=50
loso_f1 = {
    'LDA': np.mean([0.2140, 0.4011, 0.2392, 0.3622, 0.3850]),
    'LogReg': np.mean([0.2051, 0.4172, 0.2300, 0.3769, 0.3848]),
    'NaiveBayes': np.mean([0.1802, 0.2995, 0.1914, 0.3275, 0.3044]),
    'LinearSVM': np.mean([0.1521, 0.2549, 0.1816, 0.2926, 0.2836]),
}

# LOSO std for error bars
loso_std = {
    'LDA': np.std([0.2140, 0.4011, 0.2392, 0.3622, 0.3850]),
    'LogReg': np.std([0.2051, 0.4172, 0.2300, 0.3769, 0.3848]),
    'NaiveBayes': np.std([0.1802, 0.2995, 0.1914, 0.3275, 0.3044]),
    'LinearSVM': np.std([0.1521, 0.2549, 0.1816, 0.2926, 0.2836]),
}

# Create figure
fig, ax = plt.subplots(figsize=(5, 4))

x = np.arange(len(models))
width = 0.35

# Bars
bars1 = ax.bar(x - width/2, [random_f1[m] for m in models], width, 
               label='Random 70/30 Split', color='#2ecc71', edgecolor='black')
bars2 = ax.bar(x + width/2, [loso_f1[m] for m in models], width,
               yerr=[loso_std[m] for m in models], capsize=5,
               label='LOSO (Cross-Subject)', color='#e74c3c', edgecolor='black')

# Labels and formatting
ax.set_ylabel('F1 Score', fontsize=11)
ax.set_xlabel('Classifier', fontsize=11)
ax.set_title('Ninapro DB5 (52 classes):\nRandom Split vs LOSO', fontsize=12)
ax.set_xticks(x)
ax.set_xticklabels(models, fontsize=10)
ax.legend(loc='upper right', fontsize=9)
ax.set_ylim(0, 1.05)

# Add value labels on bars
def add_labels(bars, fmt='{:.1%}'):
    for bar in bars:
        height = bar.get_height()
        ax.annotate(fmt.format(height),
                    xy=(bar.get_x() + bar.get_width() / 2, height),
                    xytext=(0, 3), textcoords="offset points",
                    ha='center', va='bottom', fontsize=9)

add_labels(bars1)
add_labels(bars2)

# Add gap annotation
for i, m in enumerate(models):
    gap = random_f1[m] - loso_f1[m]
    mid_y = (random_f1[m] + loso_f1[m]) / 2
    ax.annotate(f'Δ={gap:.1%}', xy=(i, mid_y), fontsize=8, ha='center', 
                color='gray', style='italic')

# Grid
ax.yaxis.grid(True, linestyle='--', alpha=0.7)
ax.set_axisbelow(True)

plt.tight_layout()
plt.savefig('results_ninapro_comparison/random_vs_loso_comparison.png', dpi=150)
plt.savefig('results_ninapro_comparison/random_vs_loso_comparison.pdf')
print("Saved: results_ninapro_comparison/random_vs_loso_comparison.png")
print("Saved: results_ninapro_comparison/random_vs_loso_comparison.pdf")

plt.show()
