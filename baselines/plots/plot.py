import json
import os

import matplotlib.pyplot as plt
import pandas as pd
from scipy.stats import wilcoxon
import seaborn as sns
sns.set(
    rc={
        'figure.figsize': (11.7, 8.27),
        'font.size': 31,
        'axes.titlesize': 31,
        'axes.labelsize': 31,
        'xtick.labelsize': 31,
        'ytick.labelsize': 31,
    },
    style="white"
)

from utilities import generate_ranks_data, read_baseline_values, read_cocktail_values


def plot_models(
    baseline_dir: str,
    cocktail_dir: str,
):
    """Plot a comparison of the models and generate descriptive
    statistics based on the results of all the models.

    Generates plots which showcase the gain of the cocktail versus
    the baseline. (Plots the error rate of the baseline divided
    by the error rate of the cocktail.) Furthermore, it
    generates information regarding the wins, looses and draws
    of both methods, including a significance result.

    Parameters:
    -----------
    baseline_dir: str
        The directory where the results are located for the baseline
        methods.
    cocktail_dir: str
        The directory where the results are located for the regularization
        cocktails.
    """
    cocktail_wins = 0
    cocktail_draws = 0
    cocktail_looses = 0
    stat_reg_results = []
    stat_baseline_results = []
    comparison_test_accuracies = []
    task_nr_features = []
    task_nr_examples = []

    # xgboost_results = read_xgboost_values(xgboost_dir, model_name='xgboost')
    benchmark_results = read_baseline_values(baseline_dir, model_name='tabnet')
    cocktail_results = read_cocktail_values(cocktail_dir, baseline_dir)
    task_ids = benchmark_results.keys()

    with open(os.path.join(cocktail_dir, 'task_metadata.json'), 'r') as file:
        task_metadata = json.load(file)

    for task_id in task_ids:

        benchmark_task_result = benchmark_results[task_id]
        cocktail_task_result = cocktail_results[task_id]
        if benchmark_task_result is None:
            continue

        stat_reg_results.append(cocktail_task_result)
        stat_baseline_results.append(benchmark_task_result)
        if cocktail_task_result > benchmark_task_result:
            cocktail_wins += 1
        elif cocktail_task_result == benchmark_task_result:
            cocktail_draws += 1
        else:
            cocktail_looses += 1
        cocktail_task_result_error = 1 - cocktail_task_result
        benchmark_task_result_error = 1 - benchmark_task_result
        comparison_test_accuracies.append(benchmark_task_result_error / cocktail_task_result_error)
        task_nr_examples.append(task_metadata[f'{task_id}'][0])
        task_nr_features.append(task_metadata[f'{task_id}'][1])

    plt.scatter(task_nr_examples, comparison_test_accuracies, s=100, c='#273E47', label='Test accuracy')
    plt.axhline(y=1, color='r', linestyle='-', linewidth=3)
    plt.xscale('log')
    plt.xlabel("Number of data points")
    plt.ylabel("Gain")
    plt.ylim((0, 6))
    plt.tick_params(
        axis='x',  # changes apply to the x-axis
        which='both',  # both major and minor ticks are affected
        top=False,
        bottom=True,
        # ticks along the top edge are off
    )
    plt.tick_params(
        axis='y',
        which='both',
        left=True,
        right=False,
    )

    _, p_value = wilcoxon(stat_reg_results, stat_baseline_results)
    print(f'P Value: {p_value:.5f}')
    print(f'Cocktail Win'
          f''
          f's: {cocktail_wins}, Draws:{cocktail_draws}, Loses: {cocktail_looses}')
    plt.title('TabNet')
    # plt.title(f'Wins: {cocktail_wins}, '
    #          f'Losses: {cocktail_looses}, '
    #          f'Draws: {cocktail_draws} \n p-value: {p_value:.4f}')
    plt.savefig(
        'cocktail_improvement_tabnet_examples.pdf',
        bbox_inches='tight',
        pad_inches=0.15,
        margins=0.1,
    )


def patch_violinplot():
    """Patch seaborn's violinplot in current axis
    to workaround matplotlib's bug ##5423."""
    from matplotlib.collections import PolyCollection
    ax = plt.gca()
    for art in ax.get_children():
        if isinstance(art, PolyCollection):
            art.set_edgecolor((0.3, 0.3, 0.3))


def generate_ranks_comparison(
    all_data: pd.DataFrame,
):
    """Generate a ranks comparison between all methods.

    Creates a violin plot that showcases the ranks that
    the different methods achieve over all the tasks/datasets.

    Parameters
    ----------
    all_data: pd.DataFrame
        A dataframe where each row consists method ranks
        over a certain task.
    """
    all_data_ranked = generate_ranks_data(all_data)
    all_data = pd.melt(
        all_data_ranked,
        value_vars=all_data.columns,
        var_name='Method',
        value_name='Rank',
    )

    fig, _ = plt.subplots()
    sns.violinplot(x='Method', y='Rank', linewidth=3, data=all_data, cut=0, kind='violin')
    patch_violinplot()
    plt.title('Ranks of the baselines and the cocktail')
    plt.xlabel("")
    # plt.xticks(rotation=60)
    plt.tick_params(
        axis='x',  # changes apply to the x-axis
        which='both',  # both major and minor ticks are affected
        top=False,
        bottom=True,
        # ticks along the top edge are off
    )
    fig.autofmt_xdate()
    plt.savefig(
        'violin_ranks.pdf',
        bbox_inches='tight',
        pad_inches=0.15,
        margins=0.1,
    )


xgboost_dir = os.path.expanduser(
    os.path.join(
        '~',
        'Desktop',
        'xgboost_results',
    )
)


cocktail_dir = os.path.expanduser(
    os.path.join(
        '~',
        'Desktop',
        'PhD',
        'Rezultate',
        'RegularizationCocktail',
        'NEMO',
    )
)
