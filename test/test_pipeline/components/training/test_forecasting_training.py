import unittest

from autoPyTorch.constants import FORECASTING_BUDGET_TYPE
from autoPyTorch.pipeline.components.training.trainer.forecasting_trainer import ForecastingTrainerChoice


class TestGetBudgetTracker(unittest.TestCase):
    def test_get_budget_tracker(self):
        trainer = ForecastingTrainerChoice({})
        max_epoch = 50

        X = {'budget_type': 'epochs',
             'epochs': 5,
             }
        budget_tracker = trainer.get_budget_tracker(X)
        self.assertEqual(budget_tracker.max_epochs, 5)

        for budeget_type in FORECASTING_BUDGET_TYPE:
            budget_tracker = trainer.get_budget_tracker({'budget_type': budeget_type})
            self.assertEqual(budget_tracker.max_epochs, max_epoch)

        budget_tracker = trainer.get_budget_tracker({'budget_type': 'runtime'})
        self.assertIsNone(budget_tracker.max_epochs)
