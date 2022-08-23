import unittest.mock
from test.test_api.utils import dummy_eval_train_function
from test.test_evaluation.evaluation_util import get_binary_classification_datamanager

from ConfigSpace import Configuration

from smac.tae import StatusType

from autoPyTorch.pipeline.components.training.metrics.utils import get_metrics
from autoPyTorch.utils.logging_ import PicklableClientLogger
from autoPyTorch.utils.parallel_model_runner import run_models_on_dataset
from autoPyTorch.utils.pipeline import get_configuration_space, get_dataset_requirements
from autoPyTorch.utils.single_thread_client import SingleThreadedClient


@unittest.mock.patch('autoPyTorch.evaluation.tae.eval_train_function',
                     new=dummy_eval_train_function)
def test_run_models_on_dataset(backend):
    dataset = get_binary_classification_datamanager()
    backend.save_datamanager(dataset)
    # Search for a good configuration
    dataset_requirements = get_dataset_requirements(
        info=dataset.get_required_dataset_info()
    )
    dataset_properties = dataset.get_dataset_properties(dataset_requirements)
    search_space = get_configuration_space(info=dataset_properties)
    num_random_configs = 5
    model_configurations = [(search_space.sample_configuration(), 1) for _ in range(num_random_configs)]
    # Add a traditional model
    model_configurations.append(('lgb', 1))

    metric = get_metrics(dataset_properties=dataset_properties,
                         names=["accuracy"],
                         all_supported_metrics=False).pop()
    logger = unittest.mock.Mock(spec=PicklableClientLogger)

    dask_client = SingleThreadedClient()

    runhistory = run_models_on_dataset(
        time_left=15,
        func_eval_time_limit_secs=5,
        model_configs=model_configurations,
        logger=logger,
        metric=metric,
        dask_client=dask_client,
        backend=backend,
        seed=1,
        multiprocessing_context="fork",
        current_search_space=search_space,
    )

    has_successful_model = False
    has_matching_config = False
    # assert atleast 1 successfully fitted model
    for run_key, run_value in runhistory.data.items():
        if run_value.status == StatusType.SUCCESS:
            has_successful_model = True
        configuration = run_value.additional_info['configuration']
        for (config, _) in model_configurations:
            if isinstance(config, Configuration):
                config = config.get_dictionary()
            if config == configuration:
                has_matching_config = True

    assert has_successful_model, "Atleast 1 model should be successfully trained"
    assert has_matching_config, "Configurations should match with the passed model configurations"
