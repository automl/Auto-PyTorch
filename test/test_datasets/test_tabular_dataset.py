import pytest

from autoPyTorch.utils.pipeline import get_dataset_requirements


@pytest.mark.parametrize("fit_dictionary", ['fit_dictionary_numerical_only',
                                            'fit_dictionary_categorical_only',
                                            'fit_dictionary_num_and_categorical'], indirect=True)
def test_get_dataset_properties(backend, fit_dictionary):

    # The fixture creates a datamanager by itself
    datamanager = backend.load_datamanager()

    info = {'task_type': datamanager.task_type,
            'output_type': datamanager.output_type,
            'issparse': datamanager.issparse,
            'numerical_columns': datamanager.numerical_columns,
            'categorical_columns': datamanager.categorical_columns}
    dataset_requirements = get_dataset_requirements(info)

    dataset_properties = datamanager.get_dataset_properties(dataset_requirements)
    for expected in [
        'categorical_columns',
        'numerical_columns',
        'issparse',
        'is_small_preprocess',
        'task_type',
        'output_type',
        'input_shape',
        'output_shape',
        'num_classes',
    ]:
        assert expected in dataset_properties

    assert isinstance(dataset_properties, dict)
    for dataset_requirement in dataset_requirements:
        assert dataset_requirement.name in dataset_properties.keys()
        assert isinstance(dataset_properties[dataset_requirement.name], dataset_requirement.supported_types)

    assert datamanager.train_tensors[0].shape == fit_dictionary['X_train'].shape
    assert datamanager.train_tensors[1].shape == fit_dictionary['y_train'].shape
    assert datamanager.task_type == 'tabular_classification'
