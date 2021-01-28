from typing import Any, Dict, Optional

from autoPyTorch.api.base_task import BaseTask
from autoPyTorch.constants import (
    TABULAR_CLASSIFICATION,
    TASK_TYPES_TO_STRING,
)
from autoPyTorch.datasets.base_dataset import BaseDataset
from autoPyTorch.datasets.tabular_dataset import TabularDataset
from autoPyTorch.pipeline.tabular_classification import TabularClassificationPipeline
from autoPyTorch.utils.backend import Backend
from autoPyTorch.utils.hyperparameter_search_space_update import HyperparameterSearchSpaceUpdates

class TabularClassificationTask(BaseTask):
    """
    Tabular Classification API to the pipelines.
    Args:
        seed (int): seed to be used for reproducibility.
        n_jobs (int), (default=1): number of consecutive processes to spawn.
        logging_config (Optional[Dict]): specifies configuration
            for logging, if None, it is loaded from the logging.yaml
        ensemble_size (int), (default=50): Number of models added to the ensemble built by
            Ensemble selection from libraries of models.
            Models are drawn with replacement.
        ensemble_nbest (int), (default=50): only consider the ensemble_nbest
            models to build the ensemble
        max_models_on_disc (int), (default=50): maximum number of models saved to disc.
            Also, controls the size of the ensemble as any additional models will be deleted.
            Must be greater than or equal to 1.
        temporary_directory (str): folder to store configuration output and log file
        output_directory (str): folder to store predictions for optional test set
        delete_tmp_folder_after_terminate (bool): determines whether to delete the temporary directory,
            when finished
        include_components (Optional[Dict]): If None, all possible components are used.
            Otherwise specifies set of components to use.
        exclude_components (Optional[Dict]): If None, all possible components are used.
            Otherwise specifies set of components not to use. Incompatible with include
            components
    """
    def __init__(
        self,
        seed: int = 1,
        n_jobs: int = 1,
        logging_config: Optional[Dict] = None,
        ensemble_size: int = 50,
        ensemble_nbest: int = 50,
        max_models_on_disc: int = 50,
        temporary_directory: Optional[str] = None,
        output_directory: Optional[str] = None,
        delete_tmp_folder_after_terminate: bool = True,
        delete_output_folder_after_terminate: bool = True,
        include_components: Optional[Dict] = None,
        exclude_components: Optional[Dict] = None,
        backend: Optional[Backend] = None,
        search_space_updates: Optional[HyperparameterSearchSpaceUpdates] = None
    ):
        super().__init__(
            seed=seed,
            n_jobs=n_jobs,
            logging_config=logging_config,
            ensemble_size=ensemble_size,
            ensemble_nbest=ensemble_nbest,
            max_models_on_disc=max_models_on_disc,
            temporary_directory=temporary_directory,
            output_directory=output_directory,
            delete_tmp_folder_after_terminate=delete_tmp_folder_after_terminate,
            delete_output_folder_after_terminate=delete_output_folder_after_terminate,
            include_components=include_components,
            exclude_components=exclude_components,
            backend=backend,
            search_space_updates=search_space_updates
        )
        self.task_type = TASK_TYPES_TO_STRING[TABULAR_CLASSIFICATION]

    def _get_required_dataset_properties(self, dataset: BaseDataset) -> Dict[str, Any]:
        if not isinstance(dataset, TabularDataset):
            raise ValueError("Dataset is incompatible for the given task,: {}".format(
                type(dataset)
            ))
        return {'task_type': dataset.task_type,
                'output_type': dataset.output_type,
                'issparse': dataset.issparse,
                'numerical_columns': dataset.numerical_columns,
                'categorical_columns': dataset.categorical_columns}

    def build_pipeline(self, dataset_properties: Dict[str, Any]) -> TabularClassificationPipeline:
        return TabularClassificationPipeline(dataset_properties=dataset_properties)
