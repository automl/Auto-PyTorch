import os

from smac.runhistory.runhistory import DataOrigin, RunHistory, RunKey, RunValue, StatusType

from autoPyTorch.constants import REGRESSION_TASKS
from autoPyTorch.evaluation.abstract_evaluator import fit_pipeline
from autoPyTorch.evaluation.pipeline_class_collection import (
    DummyClassificationPipeline,
    DummyRegressionPipeline
)
from autoPyTorch.evaluation.train_evaluator import TrainEvaluator
from autoPyTorch.pipeline.traditional_tabular_classification import TraditionalTabularClassificationPipeline
from autoPyTorch.utils.common import subsampler


def dummy_traditional_classification(self, time_left: int, func_eval_time_limit_secs: int) -> None:
    run_history = RunHistory()
    run_history.load_json('./.tmp_api/traditional_run_history.json',
                          TraditionalTabularClassificationPipeline(dataset_properties={
                              'numerical_columns': [10]
                          }).get_hyperparameter_search_space())
    self.run_history.update(run_history, DataOrigin.EXTERNAL_SAME_INSTANCES)
    run_history.save_json(os.path.join(self._backend.internals_directory, 'traditional_run_history.json'),
                          save_external=True)
    return


# ========
# Fixtures
# ========
class DummyTrainEvaluator(TrainEvaluator):
    def _get_pipeline(self):
        if self.task_type in REGRESSION_TASKS:
            pipeline = DummyRegressionPipeline(config=1)
        else:
            pipeline = DummyClassificationPipeline(config=1)

        return pipeline

    def _fit_and_evaluate_loss(self, pipeline, split_id, train_indices, opt_indices):
        X = dict(train_indices=train_indices, val_indices=opt_indices, split_id=split_id, num_run=self.num_run)
        X.update(self.fit_dictionary)
        fit_pipeline(self.logger, pipeline, X, y=None)
        self.logger.info("Model fitted, now predicting")

        kwargs = {'pipeline': pipeline, 'unique_train_labels': self.unique_train_labels[split_id]}
        train_pred = self.predict(subsampler(self.X_train, train_indices), **kwargs)
        opt_pred = self.predict(subsampler(self.X_train, opt_indices), **kwargs)
        valid_pred = self.predict(self.X_valid, **kwargs)
        test_pred = self.predict(self.X_test, **kwargs)

        assert train_pred is not None and opt_pred is not None  # mypy check
        return train_pred, opt_pred, valid_pred, test_pred


# create closure for evaluating an algorithm
def dummy_eval_train_function(
        backend,
        queue,
        metric,
        budget: float,
        config,
        seed: int,
        output_y_hat_optimization: bool,
        num_run: int,
        include,
        exclude,
        disable_file_output,
        pipeline_config=None,
        budget_type=None,
        init_params=None,
        logger_port=None,
        all_supported_metrics=True,
        search_space_updates=None,
        instance: str = None,
) -> None:
    evaluator = DummyTrainEvaluator(
        queue=queue,
        fixed_pipeline_params=fixed_pipeline_params,
        evaluator_params=evaluator_params
    )
    evaluator.evaluate_loss()


def dummy_do_dummy_prediction():
    return


def make_dict_run_history_data(data):
    run_history_data = dict()
    for row in data:
        run_key = RunKey(
            config_id=row[0][0],
            instance_id=row[0][1],
            seed=row[0][2],
            budget=row[0][3])

        run_value = RunValue(
            cost=row[1][0],
            time=row[1][1],
            status=getattr(StatusType, row[1][2]['__enum__'].split(".")[-1]),
            starttime=row[1][3],
            endtime=row[1][4],
            additional_info=row[1][5],
        )
        run_history_data[run_key] = run_value
    return run_history_data
