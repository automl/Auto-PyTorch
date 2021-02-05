from collections import Counter
import json
import os
from typing import List

import numpy as np
import openml
import pandas as pd
import scipy
from scipy.stats import wilcoxon, rankdata
from sklearn.impute import SimpleImputer
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler, LabelEncoder, OneHotEncoder

import pandas as pd
import seaborn as sns
from scipy.stats import wilcoxon
import matplotlib.pyplot as plt
sns.set(rc={'figure.figsize':(11.7, 8.27),"font.size": 35,"axes.titlesize": 35, "axes.labelsize": 35, "xtick.labelsize": 35, "ytick.labelsize": 35}, style="white")

# openml split, not the one used for my experiments.
"""
train_indices, test_indices = task.get_train_test_split_indices()
X_train = X[train_indices]
y_train = y[train_indices]
X_test = X[test_indices]
y_test = y[test_indices]
"""


def get_dataset_split(dataset, val_fraction=0.2, test_fraction=0.2, seed=11):

    X, y, categorical_indicator, _ = dataset.get_data(
        dataset_format='array',
        target=dataset.default_target_attribute,
    )
    # TODO move the imputer and scaler into its own method in the future.
    imputer = SimpleImputer(strategy='most_frequent')
    label_encoder = LabelEncoder()

    empty_features = []
    # detect features that are null
    for feature_index in range(0, X.shape[1]):
        nan_mask = np.isnan(X[:, feature_index])
        nan_verdict = np.all(nan_mask)
        if nan_verdict:
            empty_features.append(feature_index)
    # remove feature indicators from categorical indicator since
    # they will be deleted from simple imputer.
    for feature_index in sorted(empty_features, reverse=True):
        del categorical_indicator[feature_index]

    X = imputer.fit_transform(X)
    y = label_encoder.fit_transform(y)
    X_train, X_test, y_train, y_test = train_test_split(
        X,
        y,
        test_size=test_fraction,
        random_state=seed,
        stratify=y,
    )
    # Center data on
    center_data = not scipy.sparse.issparse(X_train)
    scaler = StandardScaler(with_mean=center_data).fit(X_train)
    scaler.transform(X_train)
    scaler.transform(X_test)

    dataset_splits = {
        'X_train': X_train,
        'X_test': X_test,
        'y_train': y_train,
        'y_test': y_test,
    }

    if val_fraction != 0:
        new_val_fraction = val_fraction / (1 - test_fraction)
        X_train, X_val, y_train, y_val = train_test_split(
            X_train,
            y_train,
            test_size=new_val_fraction,
            random_state=seed,
            stratify=y_train,
        )
        dataset_splits['X_train'] = X_train
        dataset_splits['X_val'] = X_val
        dataset_splits['y_train'] = y_train
        dataset_splits['y_val'] = y_val

    categorical_columns = []
    categorical_dimensions = []

    for index, categorical_column in enumerate(categorical_indicator):
        if categorical_column:
            column_unique_values = len(set(X[:, index]))
            column_max_index = int(max(X[:, index]))
            # categorical columns with only one unique value
            # do not need an embedding.
            if column_unique_values == 1:
                continue
            categorical_columns.append(index)
            categorical_dimensions.append(column_max_index + 1)

    categorical_information = {
        'categorical_ind': categorical_indicator,
        'categorical_columns': categorical_columns,
        'categorical_dimensions': categorical_dimensions,
    }

    return categorical_information, dataset_splits


def get_dataset_openml(task_id=11):

    task = openml.tasks.get_task(task_id)
    dataset = task.get_dataset()

    return dataset


def check_leak_status(splits):

    X_train = splits['X_train']
    X_valid = splits['X_val']
    X_test = splits['X_test']

    for train_example in X_train:
        for valid_example in X_valid:
            if np.array_equal(train_example, valid_example):
                raise AssertionError('Leak between the training and validation set')
        for test_example in X_test:
            if np.array_equal(train_example, test_example):
                raise AssertionError('Leak between the training and test set')
    for valid_example in X_valid:
        for test_example in X_test:
            if np.array_equal(valid_example, test_example):
                raise AssertionError('Leak between the validation and test set')

    print('Leak check passed')

def check_split_stratification(splits):

    X_train = splits['X_train']
    X_val = splits['X_val']
    X_test = splits['X_test']
    y_train = splits['y_train']
    y_val = splits['y_val']
    y_test = splits['y_test']
    train_occurences = Counter(y_train)
    val_occurences = Counter(y_val)
    test_occurences = Counter(y_test)

    print(train_occurences)
    print(val_occurences)
    print(test_occurences)
    print(X_train.shape)
    print(X_val.shape)
    print(X_test.shape)
    assert X_train.shape[0] == y_train.shape[0]
    assert X_val.shape[0] == y_val.shape[0]
    assert X_test.shape[0] == y_test.shape[0]
    
def get_task_list(
    benchmark_task_file: str = 'path/to/tasks.txt',
) -> List[int]:
    with open(os.path.join(benchmark_task_file), 'r') as f:
        benchmark_info_str = f.readline()
        benchmark_task_ids = [int(task_id) for task_id in benchmark_info_str.split(' ')]

    return benchmark_task_ids


def status_exp_tasks(working_directory, seed=11, model_name='xgboost'):

    not_finished=0
    finished=0
    benchmark_task_file = 'benchmark_datasets.txt'
    benchmark_task_file_path = os.path.join(working_directory, benchmark_task_file)
    result_directory = os.path.join(working_directory, model_name)
    task_ids = get_task_list(benchmark_task_file_path)
    for task_id in task_ids:
        task_result_directory = os.path.join(result_directory, f'{task_id}', f'{seed}')
        print(task_result_directory)
        try:
            with open(os.path.join(task_result_directory, 'refit_result.json'), 'r') as file:
                task_result = json.load(file)
                print(f'Task {task_id} finished.')
                finished += 1
                # TODO do something with the result
        except FileNotFoundError:
            print(f'Task {task_id} not finished.')
            not_finished += 1
    print(f'Finished tasks: {finished} , not finished tasks: {not_finished}')


def read_xgboost_values(working_directory, seed=11, model_name='xgboost'):

    xgboost_result_dir = {}
    benchmark_task_file = 'benchmark_datasets.txt'
    benchmark_task_file_path = os.path.join(working_directory, benchmark_task_file)
    result_directory = os.path.join(working_directory, model_name)
    task_ids = get_task_list(benchmark_task_file_path)
    for task_id in task_ids:
        task_result_directory = os.path.join(result_directory, f'{task_id}', f'{seed}')
        try:
            with open(os.path.join(task_result_directory, 'refit_result.json'), 'r') as file:
                task_result = json.load(file)
            xgboost_result_dir[task_id] = task_result['test_accuracy']
        except FileNotFoundError:
            print(f'Task {task_id} not finished.')
            xgboost_result_dir[task_id] = None

    return xgboost_result_dir


def read_autosklearn_values(working_directory, seed=11, model_name='autosklearn'):

    autosklearn_result_dir = {}
    benchmark_task_file = 'benchmark_datasets.txt'
    benchmark_task_file_path = os.path.join(working_directory, benchmark_task_file)
    result_directory = os.path.join(working_directory, model_name)
    task_ids = get_task_list(benchmark_task_file_path)
    for task_id in task_ids:
        task_result_directory = os.path.join(result_directory, f'{seed}', f'{task_id}', 'results')
        try:
            with open(os.path.join(task_result_directory, 'performance.txt'), 'r') as baseline_file:
                baseline_test_acc = float(baseline_file.readline())
                autosklearn_result_dir[task_id] = baseline_test_acc
        except FileNotFoundError:
            print(f'Task {task_id} not finished.')
            autosklearn_result_dir[task_id] = None
            continue

    return autosklearn_result_dir



def read_cocktail_values(cocktail_result_dir, benchmark_task_file_dir):

    cocktail_result_dict = {}
    result_path = os.path.join(
        cocktail_result_dir,
        'cocktail',
        '512',
    )
    benchmark_task_file = 'benchmark_datasets.txt'
    benchmark_task_file_path = os.path.join(
        benchmark_task_file_dir,
        benchmark_task_file
    )
    task_ids = get_task_list(benchmark_task_file_path)
    for task_id in task_ids:

            task_result_path = os.path.join(
                result_path,
                f'{task_id}',
                'refit_run',
                '11',
            )

            if os.path.exists(task_result_path):
                if not os.path.isdir(task_result_path):
                    task_result_path = os.path.join(
                        result_path,
                        f'{task_id}',
                    )
            else:
                task_result_path = os.path.join(
                    result_path,
                    f'{task_id}',
                )

            try:
                with open(os.path.join(task_result_path, 'run_results.txt')) as f:
                    test_results = json.load(f)
                cocktail_result_dict[task_id] = test_results['mean_test_bal_acc']
            except FileNotFoundError:
                cocktail_result_dict[task_id] = None

    return cocktail_result_dict

def compare_models(xgboost_dir, cocktail_dir):

    xgboost_results = read_xgboost_values(xgboost_dir, model_name='xgboost')
    tabnet_results = read_xgboost_values(xgboost_dir, model_name='tabnet')
    cocktail_results = read_cocktail_values(cocktail_dir, xgboost_dir)
    autosklearn_results = read_autosklearn_values(cocktail_dir)

    table_dict = {
        'Task Id': [],
        'Tabnet': [],
        'XGBoost': [],
        'AutoSklearn': [],
        'Cocktail': [],
    }

    cocktail_wins = 0
    cocktail_losses = 0
    cocktail_ties = 0
    autosklearn_looses = 0
    autosklearn_ties = 0
    autosklearn_wins = 0
    cocktail_performances = []
    xgboost_performances = []
    autosklearn_performances = []
    print(cocktail_results)
    print(xgboost_results)

    for task_id in xgboost_results:
        xgboost_task_result = xgboost_results[task_id]
        if xgboost_task_result is None:
            continue
        tabnet_task_result = tabnet_results[task_id]
        cocktail_task_result = cocktail_results[task_id]
        autosklearn_task_result = autosklearn_results[task_id]
        cocktail_performances.append(cocktail_task_result)
        xgboost_performances.append(xgboost_task_result)
        autosklearn_performances.append(autosklearn_task_result)
        if cocktail_task_result > xgboost_task_result:
            cocktail_wins += 1
        elif cocktail_task_result < xgboost_task_result:
            cocktail_losses += 1
        else:
            cocktail_ties += 1
        if autosklearn_task_result > xgboost_task_result:
            autosklearn_wins += 1
        elif autosklearn_task_result < xgboost_task_result:
            autosklearn_looses += 1
        else:
            autosklearn_ties += 1
        table_dict['Task Id'].append(task_id)
        if tabnet_task_result is not None:
            table_dict['Tabnet'].append(tabnet_task_result)
        else:
            table_dict['Tabnet'].append(tabnet_task_result)
        table_dict['XGBoost'].append(xgboost_task_result)
        table_dict['Cocktail'].append(cocktail_task_result)
        table_dict['AutoSklearn'].append(autosklearn_task_result)

        comparison_table = pd.DataFrame.from_dict(table_dict)
    print(
        comparison_table.to_latex(
            index=False,
            caption='The performances of the Regularization Cocktail and the state-of-the-art competitors over the different datasets.',
            label='app:cocktail_vs_benchmarks_table',
        )
    )
    comparison_table.to_csv(os.path.join(xgboost_dir, 'table_comparison.csv'), index=False)



    _, p_value = wilcoxon(cocktail_performances, xgboost_performances)
    print(f'Cocktail wins: {cocktail_wins}, ties: {cocktail_ties}, looses: {cocktail_losses} against XGBoost')
    print(f'P-value: {p_value}')
    _, p_value = wilcoxon(xgboost_performances, autosklearn_performances)
    print(f'Xgboost vs AutoSklearn, P-value: {p_value}')
    print(f'AutoSklearn wins: {autosklearn_wins}, ties: {autosklearn_ties}, looses: {autosklearn_looses} against XGBoost')

    return comparison_table

def build_cd_diagram(
    xgboost_dir,
    cocktail_dir,
):
    xgboost_results = read_xgboost_values(xgboost_dir, model_name='xgboost')
    tabnet_results = read_xgboost_values(xgboost_dir, model_name='tabnet')
    cocktail_results = read_cocktail_values(cocktail_dir, xgboost_dir)
    autosklearn_results = read_autosklearn_values(cocktail_dir)

    models = ['Regularization Cocktail', 'XGBoost', 'AutoSklearn-GB', 'TabNet']
    table_results = {
        'Network': [],
        'Task Id': [],
        'Balanced Accuracy': [],
    }
    for task_id in cocktail_results:
        for model_name in models:
            try:
                if model_name == 'Regularization Cocktail':
                    task_result = cocktail_results[task_id]
                elif model_name == 'XGBoost':
                    task_result = xgboost_results[task_id]
                elif model_name == 'TabNet':
                    task_result = tabnet_results[task_id]
                elif model_name == 'AutoSklearn-GB':
                    task_result = autosklearn_results[task_id]
                else:
                    raise ValueError("Illegal model value")
            except Exception:
                task_result = 0
                print(f'No results for task: {task_id} for model: {model_name}')

            table_results['Network'].append(model_name)
            table_results['Task Id'].append(task_id)
            table_results['Balanced Accuracy'].append(task_result)

    result_df = pd.DataFrame(data=table_results)
    result_df.to_csv(os.path.join(xgboost_dir, f'cd_data.csv'), index=False)


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
#benchmark_table = compare_models(xgboost_dir, cocktail_dir)
#build_cd_diagram(xgboost_dir, cocktail_dir)

def plot_models(
    xgboost_dir,
    cocktail_dir,
):
    cocktail_wins = 0
    cocktail_draws = 0
    cocktail_looses = 0
    stat_reg_results = []
    stat_baseline_results = []
    comparison_train_accuracies = []
    comparison_test_accuracies = []
    task_nr_features = []
    task_nr_examples = []

    #xgboost_results = read_xgboost_values(xgboost_dir, model_name='xgboost')
    benchmark_results = read_xgboost_values(xgboost_dir, model_name='tabnet')
    cocktail_results = read_cocktail_values(cocktail_dir, xgboost_dir)
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
            cocktail_wins +=1
        elif cocktail_task_result == benchmark_task_result:
            cocktail_draws += 1
        else:
            cocktail_looses +=1
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
        axis='y',  # changes apply to the x-axis
        which='both',  # both major and minor ticks are affected
        left=True,
        right=False,
        # ticks along the top edge are off
    )

    _, p_value = wilcoxon(stat_reg_results, stat_baseline_results)
    print(f'P Value: {p_value:.5f}')
    print(f'Cocktail Win'
          f''
          f's: {cocktail_wins}, Draws:{cocktail_draws}, Loses: {cocktail_looses}')
    plt.title('TabNet')
    #plt.title(f'Wins: {cocktail_wins}, '
    #          f'Losses: {cocktail_looses}, '
    #          f'Draws: {cocktail_draws} \n p-value: {p_value:.4f}')
    plt.savefig(
        'cocktail_improvement_tabnet_examples.pdf',
        bbox_inches='tight',
        pad_inches=0.15,
        margins=0.1,
    )

plot_models(xgboost_dir, cocktail_dir)

def generate_ranks_data(
    all_data: pd.DataFrame,
):
    """
    Parameters
    ----------
    all_data: pd.DataFrame
        A dataframe where each row consists of a
        tasks values across networks with different
        regularization techniques.
    """
    all_ranked_data = []
    all_data.drop(columns=['Task Id'], inplace=True)
    column_names = all_data.columns



    for row in all_data.itertuples(index=False):
        task_regularization_data = list(row)
        task_ranked_data = rankdata(
            task_regularization_data,
            method='dense',
        )

        reversed_data = len(task_ranked_data) - task_ranked_data.astype(int)
        """for i, column_name in enumerate(column_names):
            all_ranked_data.append([column_name, task_ranked_data[i]])
        """
        all_ranked_data.append(reversed_data)
    ranks_df = pd.DataFrame(all_ranked_data, columns=column_names)

    return ranks_df


def patch_violinplot():
    """Patch seaborn's violinplot in current axis to workaround matplotlib's bug ##5423."""
    from matplotlib.collections import PolyCollection
    ax = plt.gca()
    for art in ax.get_children():
        if isinstance(art, PolyCollection):
            art.set_edgecolor((0.3, 0.3, 0.3))

def generate_ranks_comparison(
    all_data: pd.DataFrame,
):
    """
    Parameters
    ----------
    all_data: pd.DataFrame
        A dataframe where each row consists of a
        tasks values across networks with different
        regularization techniques.
    """
    all_data_ranked = generate_ranks_data(all_data)
    all_data = pd.melt(
        all_data_ranked,
        value_vars=all_data.columns,
        var_name='Method',
        value_name='Rank',
    )

    fig, _ = plt.subplots()
    print(all_data)
    sns.violinplot(x='Method', y='Rank', linewidth=3, data=all_data, cut=0, kind='violin')
    patch_violinplot()
    plt.title('Ranks of the baselines and the cocktail')
    plt.xlabel("")
    #plt.xticks(rotation=60)
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

#generate_ranks_comparison(benchmark_table)