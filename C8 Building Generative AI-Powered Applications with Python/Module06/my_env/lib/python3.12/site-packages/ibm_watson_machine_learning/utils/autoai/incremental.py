#  -----------------------------------------------------------------------------------------
#  (C) Copyright IBM Corp. 2022-2024.
#  https://opensource.org/licenses/BSD-3-Clause
#  -----------------------------------------------------------------------------------------

import numpy as np
import matplotlib.pyplot as plt
from IPython.display import display, clear_output


def calculate_cum_limits(scores):
    def lower_limit(values):
        mean = np.mean(values)
        std = np.std(values)
        return mean - std

    def upper_limit(values):
        mean = np.mean(values)
        std = np.std(values)
        return mean + std

    n = len(scores) + 1
    lower_limits = [lower_limit(scores[:i]) for i in range(1, n)]
    upper_limits = [upper_limit(scores[:i]) for i in range(1, n)]

    return lower_limits, upper_limits


def plot_subcurve(fig, ax, values, x=None, color='b', title='Learning Curve', xlabel='Training batches', ylabel='Score',
                  refresh=True):
    if x is None:
        x = list(range(0, len(values)))

    lower_limits, upper_limits = calculate_cum_limits(values)
    ax.cla()
    ax.grid()
    ax.plot(x, values, "o-")
    ax.fill_between(
        x,
        lower_limits,
        upper_limits,
        alpha=0.1,
        color=color,
    )
    ax.set_title(title)
    ax.set_xlabel(xlabel)
    ax.set_ylabel(ylabel)

    if refresh:
        display(fig)
        clear_output(wait=True)


def plot_learning_curve(fig, axes, scores, fit_times=None, xlabels: dict=None, titles: dict=None):
    if isinstance(axes, np.ndarray) and fit_times is not None:
        # plot score vs. batch number

        plot_subcurve(fig, axes[0], scores)

        if titles and titles.get('first_plot'):
            axes[0].set_title(titles['first_plot'])

        if xlabels and xlabels.get('first_xlabel'):
            axes[0].set_xlabel(xlabels['first_xlabel'])

        if len(fit_times) < len(scores):
            x = list(range(len(scores) - len(fit_times), len(scores)))
            scores_aligned = scores[len(scores) - len(fit_times):]
        else:
            x = list(range(0, len(scores)))
            scores_aligned = scores

        # plot fit_time vs. batch number

        plot_subcurve(fig, axes[1], fit_times, x, color='g', title='Scalability of the model',
                      xlabel='Training batches', ylabel='Fit time')

        if titles and titles.get('second_plot'):
            axes[1].set_title(titles['second_plot'])

        if xlabels and xlabels.get('second_xlabel'):
            axes[1].set_xlabel(xlabels['second_xlabel'])

        if len(axes) == 3:
            # plot fit_time vs. score
            fit_times_np = np.array(fit_times)
            fit_times_cumsum = np.cumsum(fit_times_np)
            plot_subcurve(fig, axes[2], scores_aligned, x=fit_times_cumsum, color='r', title='Performance of the model',
                          xlabel='Fit time', ylabel='Score')

        if titles and titles.get('third_plot'):
            axes[2].set_title(titles['third_plot'])

        if xlabels and xlabels.get('third_xlabel'):
            axes[2].set_xlabel(xlabels['third_xlabel'])

    else:
        # plot score vs. batch number
        plot_subcurve(fig, axes[0], scores)
