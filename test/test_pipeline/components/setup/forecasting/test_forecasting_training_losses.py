import copy
import unittest

from autoPyTorch.constants import TASK_TYPES_TO_STRING, TIMESERIES_FORECASTING
from autoPyTorch.pipeline.components.setup.forecasting_training_loss import ForecastingLossChoices
from autoPyTorch.pipeline.components.setup.forecasting_training_loss.DistributionLoss import DistributionLoss
from autoPyTorch.pipeline.components.setup.forecasting_training_loss.QuantileLoss import NetworkQuantileLoss
from autoPyTorch.pipeline.components.setup.forecasting_training_loss.RegressionLoss import RegressionLoss
from autoPyTorch.pipeline.components.setup.network_head.forecasting_network_head.distribution import (
    ALL_DISTRIBUTIONS,
    DisForecastingStrategy
)
from autoPyTorch.pipeline.components.training.losses import (
    L1Loss,
    LogProbLoss,
    MAPELoss,
    MASELoss,
    MSELoss,
    QuantileLoss
)


class TestForecastingTrainingLoss(unittest.TestCase):
    def test_get_set_config_space(self):
        """Make sure that we can setup a valid choice in the encoder
        choice"""
        dataset_properties = {'task_type': TASK_TYPES_TO_STRING[TIMESERIES_FORECASTING]}
        loss_choice = ForecastingLossChoices(dataset_properties)
        cs = loss_choice.get_hyperparameter_search_space(dataset_properties)

        # Make sure that all hyperparameters are part of the search space
        self.assertListEqual(
            sorted(cs.get_hyperparameter('__choice__').choices),
            sorted(list(loss_choice.get_components().keys()))
        )

        # Make sure we can properly set some random configs
        # Whereas just one iteration will make sure the algorithm works,
        # doing five iterations increase the confidence. We will be able to
        # catch component specific crashes
        for i in range(5):
            config = cs.sample_configuration()
            config_dict = copy.deepcopy(config.get_dictionary())
            loss_choice.set_hyperparameters(config)

            self.assertEqual(loss_choice.choice.__class__,
                             loss_choice.get_components()[config_dict['__choice__']])

        include = ['DistributionLoss', 'QuantileLoss']
        cs = loss_choice.get_hyperparameter_search_space(dataset_properties=dataset_properties, include=include)
        self.assertTrue(
            sorted(cs.get_hyperparameter('__choice__').choices),
            sorted(include),
        )

    def test_distribution_loss(self):
        for dist_cls in ALL_DISTRIBUTIONS.keys():
            loss = DistributionLoss(dist_cls)
            self.assertEqual(loss.dist_cls, dist_cls)

            dataset_properties = {'task_type': TASK_TYPES_TO_STRING[TIMESERIES_FORECASTING]}
            fit_dictionary = {'dataset_properties': dataset_properties}
            loss = loss.fit(fit_dictionary)
            fit_dictionary = loss.transform(fit_dictionary)

            self.assertEqual(fit_dictionary['loss'], LogProbLoss)
            self.assertEqual(fit_dictionary['required_padding_value'], ALL_DISTRIBUTIONS[dist_cls].value_in_support)
            self.assertIsInstance(fit_dictionary['dist_forecasting_strategy'], DisForecastingStrategy)

    def test_quantile_loss(self):
        lower = 0.2
        upper = 0.8
        loss = NetworkQuantileLoss(lower_quantile=lower, upper_quantile=upper)
        self.assertEqual(loss.quantiles, [0.5, lower, upper])

        dataset_properties = {'task_type': TASK_TYPES_TO_STRING[TIMESERIES_FORECASTING]}
        fit_dictionary = {'dataset_properties': dataset_properties}
        loss = loss.fit(fit_dictionary)
        fit_dictionary = loss.transform(fit_dictionary)
        train_loss = fit_dictionary['loss']()

        self.assertIsInstance(train_loss, QuantileLoss)
        self.assertListEqual(train_loss.quantiles, loss.quantiles)
        self.assertListEqual(fit_dictionary['quantile_values'], loss.quantiles)

    def test_regression_loss(self):
        loss_dict = dict(l1=L1Loss,
                         mse=MSELoss,
                         mape=MAPELoss,
                         mase=MASELoss)
        for loss_name, loss_type in loss_dict.items():
            loss = RegressionLoss(loss_name)

            dataset_properties = {'task_type': TASK_TYPES_TO_STRING[TIMESERIES_FORECASTING]}
            fit_dictionary = {'dataset_properties': dataset_properties}
            loss = loss.fit(fit_dictionary)
            fit_dictionary = loss.transform(fit_dictionary)
            train_loss = fit_dictionary['loss']

            self.assertEqual(train_loss, loss_type)
