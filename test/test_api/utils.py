import os

from smac.runhistory.runhistory import RunHistory, DataOrigin

from autoPyTorch.constants import REGRESSION_TASKS
from autoPyTorch.evaluation.abstract_evaluator import (
    DummyClassificationPipeline,
    DummyRegressionPipeline,
    fit_and_suppress_warnings
)
from autoPyTorch.evaluation.train_evaluator import TrainEvaluator
from autoPyTorch.pipeline.traditional_tabular_classification import TraditionalTabularClassificationPipeline


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

    def _fit_and_predict(self, pipeline, fold: int, train_indices,
                         test_indices,
                         add_pipeline_to_self
                         ):

        if self.task_type in REGRESSION_TASKS:
            pipeline = DummyRegressionPipeline(config=1)
        else:
            pipeline = DummyClassificationPipeline(config=1)

        self.indices[fold] = ((train_indices, test_indices))

        X = {'train_indices': train_indices,
             'val_indices': test_indices,
             'split_id': fold,
             'num_run': self.num_run,
             **self.fit_dictionary}  # fit dictionary
        y = None
        fit_and_suppress_warnings(self.logger, pipeline, X, y)
        self.logger.info("Model fitted, now predicting")
        (
            Y_train_pred,
            Y_opt_pred,
            Y_valid_pred,
            Y_test_pred
        ) = self._predict(
            pipeline,
            train_indices=train_indices,
            test_indices=test_indices,
        )

        if add_pipeline_to_self:
            self.pipeline = pipeline
        else:
            self.pipelines[fold] = pipeline

        return Y_train_pred, Y_opt_pred, Y_valid_pred, Y_test_pred


# create closure for evaluating an algorithm
def dummy_eval_function(
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
        backend=backend,
        queue=queue,
        metric=metric,
        configuration=config,
        seed=seed,
        num_run=num_run,
        output_y_hat_optimization=output_y_hat_optimization,
        include=include,
        exclude=exclude,
        disable_file_output=disable_file_output,
        init_params=init_params,
        budget=budget,
        budget_type=budget_type,
        logger_port=logger_port,
        all_supported_metrics=all_supported_metrics,
        pipeline_config=pipeline_config,
        search_space_updates=search_space_updates
    )
    evaluator.fit_predict_and_loss()


def dummy_do_dummy_prediction():
    return
