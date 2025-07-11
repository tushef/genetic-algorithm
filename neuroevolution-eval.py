import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns

if __name__ == "__main__":
    # Load data
    df = pd.read_csv("evolution_log_cvar.csv")

    generations = df['generation']
    top_score = df['top_score']
    std_score = df['std_score']
    score_iqr = df['score_iqr']
    noise_range = df['noise_range']
    param_l2_diversity = df['param_l2_diversity']
    param_std = df['param_std']

    # Set plot style
    sns.set(style="whitegrid")
    plt.figure(figsize=(16, 12))

    # Plot 1 — Top Score Progression with Std Mask
    plt.subplot(3, 1, 1)
    plt.plot(generations, top_score, label='Top Score', color='blue')
    plt.fill_between(
        generations,
        top_score - std_score,
        top_score + std_score,
        alpha=0.3,
        color='blue',
        label='Std Range'
    )
    plt.title("Top Score Progression with Standard Deviation")
    plt.xlabel("Generation")
    plt.ylabel("Fitness")
    plt.legend()
    plt.grid(True)

    # Plot 2 — Distribution Indicators: IQR, Noise Range, Std
    plt.subplot(3, 1, 2)
    plt.errorbar(
        generations, top_score, yerr=score_iqr / 2,
        fmt='o', label='IQR', color='purple',
        elinewidth=15, capsize=5, alpha=1
    )
    plt.errorbar(
        generations, top_score, yerr=noise_range / 2,
        fmt='s', label='Noise Range', color='orange',
        elinewidth=2, capsize=5, alpha=0.5
    )
    plt.errorbar(
        generations, top_score, yerr=std_score,
        fmt='^', label='Std Dev', color='red',
        elinewidth=8, capsize=5, alpha=0.7
    )
    plt.title("Fitness Distribution Indicators per Generation")
    plt.xlabel("Generation")
    plt.ylabel("Fitness")
    plt.legend()
    plt.grid(True)

    # Plot 3 — L2 Diversity with Param Std Mask
    plt.subplot(3, 1, 3)
    plt.plot(generations, param_l2_diversity, label='L2 Diversity', color='green', linewidth=2)

    plt.title("Parameter Diversity Over Generations")
    plt.xlabel("Generation")
    plt.ylabel("L2 Distance")
    plt.legend()
    plt.grid(True)

    # Layout and show
    plt.tight_layout()
    plt.show()