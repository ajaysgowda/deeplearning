"""
Utils
1. plot_learning_curve
"""

import numpy as np
import matplotlib.pyplot as plt


def plot_learning_curve(x, scores, epsilons, filename):
    """
    Plot
    """
    fig, ax1 = plt.subplots()

    ax1.plot(x, epsilons, color="C0")
    ax1.set_xlabel("Training Steps", color="C0")
    ax1.set_ylabel("Epsilon", color="C0")
    ax1.tick_params(axis="x", color="C0")
    ax1.tick_params(axis="y", color="C0")

    ax2 = ax1.twinx()

    N = len(scores)

    running_avg = np.empty(N)
    for t in range(N):
        running_avg[t] = np.mean(scores[max(0, t-100):(t+1)])

    ax2.scatter(x, running_avg, color="C1")
    ax2.axes.get_xaxis().set_visible(False)
    ax2.set_ylabel("Score", color="C1")
    ax2.tick_params(axis='y', colors="C1")

    fig.tight_layout()

    plt.savefig(filename)

