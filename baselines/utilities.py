from collections import Counter
import json
import os
from typing import Dict, List, Tuple, Union

import numpy as np
import openml
import pandas as pd
import scipy
from scipy.stats import wilcoxon, rankdata
from sklearn.impute import SimpleImputer
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler, LabelEncoder, OneHotEncoder


"""
train_indices, test_indices = task.get_train_test_split_indices()
X_train = X[train_indices]
y_train = y[train_indices]
X_test = X[test_indices]
y_test = y[test_indices]
"""


def get_dataset_split(
        dataset: openml.datasets.OpenMLDataset,
        val_fraction: float = 0.2,
        test_fraction: float = 0.2,
        seed: int = 11,
) -> Tuple[Dict[str, Union[List, np.ndarray]], Dict[str, np.ndarray]]:
    """Split the dataset into training, test and possibly validation set.

    Based on the arguments given, splits the datasets into the corresponding
    sets.

    Parameters:
    -----------
    dataset: openml.datasets.OpenMLDataset
        The dataset that will be split into the corresponding sets.
    val_fraction: float
        The fraction for the size of the validation set from the whole dataset.
    test_fraction: float
        The fraction for the size of the test set from the whole dataset.
    seed: int
        The seed used for the splitting of the dataset.

    Returns:
    --------
    (categorical_information, dataset_splits): tuple(np.array, dict)
        Returns a tuple, where the first arguments provides categorical information
        about the features. While the second argument, is a dictionary with the splits
        for the different sets.
    """
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
    # Center data only on not sparse matrices
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


def get_dataset_openml(
        task_id: int = 11,
) -> openml.datasets.OpenMLDataset:
    """Download a dataset from OpenML

    Based on a given task id, download the task and retrieve
    the dataset that belongs to the corresponding task.

    Parameters:
    -----------
    task_id: int
        The task id that represents the task for which the dataset will be downloaded.

    Returns:
    --------
    dataset: openml.datasets.OpenMLDataset
        The OpenML dataset that is requested..
    """
    task = openml.tasks.get_task(task_id)
    dataset = task.get_dataset()

    return dataset


def check_leak_status(splits):
    """Check the leak status.

    This function goes through the different splits of the dataset
    and checks if there is a leak between the different sets.

    Parameters:
    -----------
    splits: dict
        A dictionary that contains the different sets train, test (possibly validation)
        of the whole dataset.

    Returns:
    --------
    None - Does not return anything, only raises an error if there is a leak.
    """
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
    """Check the split stratification and the shape of the examples and labels
    for the different sets.

    This function goes through the different splits of the dataset
    and checks if there is stratification. In this example, if there
    is nearly the same number of examples for each class in the corresponding
    splits. The function also verifies that the shape of the examples and
    labels is the same for the different splits.

    Parameters:
    -----------
    splits: dict
        A dictionary that contains the different sets train, test (possibly validation)
        of the whole dataset.
    """
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
    """Get the task id list.

    Goes through the given file and collects all of the task
    ids.

    Parameters:
    -----------
    benchmark_task_file: str
        A string to the path of the benchmark task file. Including
        the task file name.

    Returns:
    --------
    benchmark_task_ids - list
        A list of all the task ids for the benchmark.
    """
    with open(os.path.join(benchmark_task_file), 'r') as f:
        benchmark_info_str = f.readline()
        benchmark_task_ids = [int(task_id) for task_id in benchmark_info_str.split(' ')]

    return benchmark_task_ids


def status_exp_tasks(
        working_directory: str,
        seed: int = 11,
        model_name: str = 'xgboost',
):
    """Analyze the different tasks of the experiment.

    Goes through the results in the directory given and
    it analyzes which one finished succesfully and which one
    did not.

    Parameters:
    -----------
    working_directory: str
        The directory where the results are located.
    seed: int
        The seed that was used for the experiment.
    model_name: int
        The name of the model that was used.
    """
    not_finished = 0
    finished = 0
    benchmark_task_file = 'benchmark_datasets.txt'
    benchmark_task_file_path = os.path.join(working_directory, benchmark_task_file)
    result_directory = os.path.join(working_directory, model_name)
    task_ids = get_task_list(benchmark_task_file_path)
    for task_id in task_ids:
        task_result_directory = os.path.join(result_directory, f'{task_id}', f'{seed}')
        print(task_result_directory)
        try:
            with open(os.path.join(task_result_directory, 'refit_result.json'), 'r') as file:
                # do nothing with the result for now
                _ = json.load(file)
                print(f'Task {task_id} finished.')
                finished += 1
                # TODO do something with the result
        except FileNotFoundError:
            print(f'Task {task_id} not finished.')
            not_finished += 1
    print(f'Finished tasks: {finished} , not finished tasks: {not_finished}')


def read_baseline_values(
        working_directory: str,
        seed: int = 11,
        model_name: str = 'xgboost',
) -> Dict[int, float]:
    """Prepares the results of the experiment with the baselines.

    Goes through the results at the given directory and it generates a
    dictionary for the baseline with the performances on every task
    of the benchmark.

    Parameters:
    -----------
    working_directory: str
        The directory where the results are located.
    seed: int
        The seed that was used for the experiment.
    model_name: int
        The name of the model that was used.

    Returns:
    --------
    baseline_results - dict
        A dictionary with the results of the baseline algorithm.
        Each key of the dictionary represents a task id, while,
        each value corresponds to the performance of the algorithm.
    """
    baseline_results = {}
    benchmark_task_file = 'benchmark_datasets.txt'
    benchmark_task_file_path = os.path.join(working_directory, benchmark_task_file)
    result_directory = os.path.join(working_directory, model_name)
    task_ids = get_task_list(benchmark_task_file_path)
    for task_id in task_ids:
        task_result_directory = os.path.join(result_directory, f'{task_id}', f'{seed}')
        try:
            with open(os.path.join(task_result_directory, 'refit_result.json'), 'r') as file:
                task_result = json.load(file)
            baseline_results[task_id] = task_result['test_accuracy']
        except FileNotFoundError:
            print(f'Task {task_id} not finished.')
            baseline_results[task_id] = None

    return baseline_results


def read_autosklearn_values(
        working_directory,
        seed=11,
        model_name='autosklearn'
) -> Dict[int, float]:
    """Prepares the results of the experiment with auto-sklearn.

    Goes through the results at the given directory and it generates a
    dictionary for autosklearn with the performances on every task
    of the benchmark.

    Parameters:
    -----------
    working_directory: str
        The directory where the results are located.
    seed: int
        The seed that was used for the experiment.
    model_name: int
        The name of the model that was used.

    Returns:
    --------
    autosklearn_results - dict
        A dictionary with the results of the autosklearn algorithm.
        Each key of the dictionary represents a task id, while,
        each value corresponds to the performance of the algorithm.
    """
    autosklearn_results = {}
    benchmark_task_file = 'benchmark_datasets.txt'
    benchmark_task_file_path = os.path.join(working_directory, benchmark_task_file)
    result_directory = os.path.join(working_directory, model_name)
    task_ids = get_task_list(benchmark_task_file_path)
    for task_id in task_ids:
        task_result_directory = os.path.join(result_directory, f'{seed}', f'{task_id}', 'results')
        try:
            with open(os.path.join(task_result_directory, 'performance.txt'), 'r') as baseline_file:
                baseline_test_acc = float(baseline_file.readline())
                autosklearn_results[task_id] = baseline_test_acc
        except FileNotFoundError:
            print(f'Task {task_id} not finished.')
            autosklearn_results[task_id] = None
            continue

    return autosklearn_results


def read_cocktail_values(
        cocktail_result_dir: str,
        benchmark_task_file_dir: str,
        seed: int = 11
) -> Dict[int, float]:
    """Prepares the results of the experiment with the regularization
    cocktail.

    Goes through the results at the given directory and it generates a
    dictionary for the regularization cocktails with the performances
    on every task of the benchmark.

    Parameters:
    -----------
    cocktail_result_dir: str
        The directory where the results are located for the regularization
        cocktails.
    benchmark_task_file_dir: str
        The directory where the benchmark task file is located.
        The file contains all the task ids. The file name is
        not needed to be given.
    seed: int
        The seed that was used for the experiment.

    Returns:
    --------
    cocktail_results - dict
        A dictionary with the results of the regularization cocktail method.
        Each key of the dictionary represents a task id, while,
        each value corresponds to the performance of the algorithm.
    """
    cocktail_results = {}

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
            f'{seed}',
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
            cocktail_results[task_id] = test_results['mean_test_bal_acc']
        except FileNotFoundError:
            cocktail_results[task_id] = None

    return cocktail_results


def compare_models(
        baseline_dir: str,
        cocktail_dir: str,
) -> pd.DataFrame:
    """Prepares the results of the experiments with all methods.

    Goes through the results at the given directories and builds
    a table with all the methods over the different tasks.

    Parameters:
    -----------
    baseline_dir: str
        The directory where the results are located for the baseline
        methods.
    cocktail_dir: str
        The directory where the results are located for the regularization
        cocktails.

    Returns:
    --------
    comparison_table - pd.DataFrame
        A DataFrame with the results for all methods over the different tasks.
    """
    xgboost_results = read_baseline_values(baseline_dir, model_name='xgboost')
    tabnet_results = read_baseline_values(baseline_dir, model_name='tabnet')
    cocktail_results = read_cocktail_values(cocktail_dir, baseline_dir)
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
            caption='The performances of the Regularization Cocktail '
                    'and the state-of-the-art competitors '
                    'over the different datasets.',
            label='app:cocktail_vs_benchmarks_table',
        )
    )
    comparison_table.to_csv(os.path.join(baseline_dir, 'table_comparison.csv'), index=False)
    _, p_value = wilcoxon(cocktail_performances, xgboost_performances)
    print(f'Cocktail wins: {cocktail_wins}, ties: {cocktail_ties}, looses: {cocktail_losses} against XGBoost')
    print(f'P-value: {p_value}')
    _, p_value = wilcoxon(xgboost_performances, autosklearn_performances)
    print(f'Xgboost vs AutoSklearn, P-value: {p_value}')
    print(f'AutoSklearn wins: {autosklearn_wins}, '
          f'ties: {autosklearn_ties}, '
          f'looses: {autosklearn_looses} against XGBoost')

    return comparison_table


def build_cd_diagram(
    baseline_dir: str,
    cocktail_dir: str,
) -> pd.DataFrame:
    """Prepare the results for a critical difference diagram.

    This function prepares all the results into a pandas dataframe
    so that it can be used to create a critical difference diagram
    of all the methods.

    Parameters:
    -----------
    baseline_dir: str
        The directory where the results are located for the baseline
        methods.
    cocktail_dir: str
        The directory where the results are located for the regularization
        cocktails.

    Returns:
    --------
        result_df: pd.DataFrame
            A table with the accuracies of all methods over the different tasks.
            The results are prepared in such a way that a critical difference
            diagram can be generated from the pandas dataframe.
    """
    xgboost_results = read_baseline_values(baseline_dir, model_name='xgboost')
    tabnet_results = read_baseline_values(baseline_dir, model_name='tabnet')
    cocktail_results = read_cocktail_values(cocktail_dir, baseline_dir)
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
            except FileNotFoundError:
                task_result = 0
                print(f'No results for task: {task_id} for model: {model_name}')

            table_results['Network'].append(model_name)
            table_results['Task Id'].append(task_id)
            table_results['Balanced Accuracy'].append(task_result)

    result_df = pd.DataFrame(data=table_results)
    result_df.to_csv(os.path.join(baseline_dir, f'cd_data.csv'), index=False)

    return result_df


def generate_ranks_data(
    all_data: pd.DataFrame,
):
    """
    Parameters
    ----------
    all_data: pd.DataFrame
        A dataframe where each row consists of
        tasks values across different models.

    Returns
    -------
    ranks_df: pd.DataFrame
        A dataframe of the ranks of all methods over
        the different tasks.
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

        reversed_data = len(task_ranked_data) + 1 - task_ranked_data.astype(int)
        """for i, column_name in enumerate(column_names):
            all_ranked_data.append([column_name, task_ranked_data[i]])
        """
        all_ranked_data.append(reversed_data)
    ranks_df = pd.DataFrame(all_ranked_data, columns=column_names)

    return ranks_df
