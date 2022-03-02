import copy
import glob
import os
import shutil
import sys
import tempfile
import unittest
import unittest.mock

import numpy as np

import pytest

from sklearn.base import clone

import torch

from autoPyTorch import constants
from autoPyTorch.pipeline.components.training.data_loader.base_data_loader import (
    BaseDataLoaderComponent,
)
from autoPyTorch.pipeline.components.training.trainer import (
    TrainerChoice,
)
from autoPyTorch.pipeline.components.training.trainer.MixUpTrainer import (
    MixUpTrainer
)
from autoPyTorch.pipeline.components.training.trainer.StandardTrainer import (
    StandardTrainer
)
from autoPyTorch.pipeline.components.training.trainer.base_trainer import (
    BaseTrainerComponent,
    BudgetTracker,
    StepIntervalUnit
)

sys.path.append(os.path.dirname(__file__))
from test.test_pipeline.components.training.base import BaseTraining  # noqa (E402: module level import not at top of file)


OVERFIT_EPOCHS = 1000
N_SAMPLES = 500


class TestBaseDataLoader(unittest.TestCase):
    def test_get_set_config_space(self):
        """
        Makes sure that the configuration space of the base data loader
        is properly working"""
        loader = BaseDataLoaderComponent()

        cs = loader.get_hyperparameter_search_space()

        # Make sure that the batch size is a valid hyperparameter
        self.assertEqual(cs.get_hyperparameter('batch_size').default_value, 64)

        # Make sure we can properly set some random configs
        for _ in range(5):
            config = cs.sample_configuration()
            config_dict = copy.deepcopy(config.get_dictionary())
            loader.set_hyperparameters(config)

            self.assertEqual(loader.batch_size,
                             config_dict['batch_size'])

    def test_check_requirements(self):
        """ Makes sure that we catch the proper requirements for the
        data loader"""

        fit_dictionary = {'dataset_properties': {}}

        loader = BaseDataLoaderComponent()

        # Make sure we catch all possible errors in check requirements

        # No input in fit dictionary
        with self.assertRaisesRegex(
                ValueError,
                'To fit a data loader, expected fit dictionary to have split_id.'
        ):
            loader.fit(fit_dictionary)

        # Backend Missing
        fit_dictionary.update({'split_id': 0})
        with self.assertRaisesRegex(ValueError,
                                    'backend is needed to load the data from'):
            loader.fit(fit_dictionary)

        # Then the is small fit
        fit_dictionary.update({'backend': unittest.mock.Mock()})
        with self.assertRaisesRegex(ValueError,
                                    'is_small_pre-process is required to know if th'):
            loader.fit(fit_dictionary)

    def test_fit_transform(self):
        """ Makes sure that fit and transform work as intended """
        backend = unittest.mock.Mock()
        fit_dictionary = {
            'X_train': np.array([[1, 2, 3], [4, 5, 6], [7, 8, 9]]),
            'y_train': np.array([0, 1, 0]),
            'train_indices': [0, 1],
            'val_indices': [2],
            'dataset_properties': {'is_small_preprocess': True},
            'working_dir': '/tmp',
            'split_id': 0,
            'backend': backend,
        }
        dataset = unittest.mock.MagicMock()
        dataset.__len__.return_value = 1
        datamanager = unittest.mock.MagicMock()
        datamanager.get_dataset.return_value = (dataset, dataset)
        fit_dictionary['backend'].load_datamanager.return_value = datamanager

        # Mock child classes requirements
        loader = BaseDataLoaderComponent()
        loader.build_transform = unittest.mock.Mock()
        loader._check_transform_requirements = unittest.mock.Mock()

        loader.fit(fit_dictionary)

        # Fit means that we created the data loaders
        self.assertIsInstance(loader.train_data_loader, torch.utils.data.DataLoader)
        self.assertIsInstance(loader.val_data_loader, torch.utils.data.DataLoader)

        # Transforms adds this fit dictionaries
        transformed_fit_dictionary = loader.transform(fit_dictionary)
        self.assertIn('train_data_loader', transformed_fit_dictionary)
        self.assertIn('val_data_loader', transformed_fit_dictionary)

        self.assertEqual(transformed_fit_dictionary['train_data_loader'],
                         loader.train_data_loader)
        self.assertEqual(transformed_fit_dictionary['val_data_loader'],
                         loader.val_data_loader)


class TestBaseTrainerComponent(BaseTraining):
    def test_evaluate(self):
        """
        Makes sure we properly evaluate data, returning a proper loss
        and metric
        """

        (trainer,
         model,
         optimizer,
         loader,
         criterion,
         epochs,
         _) = self.prepare_trainer(N_SAMPLES,
                                   BaseTrainerComponent(),
                                   constants.TABULAR_CLASSIFICATION)

        prev_loss, prev_metrics = trainer.evaluate(loader, epoch=1, writer=None)
        assert 'accuracy' in prev_metrics

        # Fit the model
        self.train_model(model,
                         optimizer,
                         loader,
                         criterion,
                         epochs)

        # Loss and metrics should have improved after fit
        # And the prediction should be better than random
        loss, metrics = trainer.evaluate(loader, epoch=1, writer=None)
        assert prev_loss > loss
        assert metrics['accuracy'] > prev_metrics['accuracy']
        assert metrics['accuracy'] > 0.5

    def test_scheduler_step(self):
        trainer = BaseTrainerComponent()
        model = torch.nn.Linear(1, 1)

        base_lr, factor = 1, 10
        optimizer = torch.optim.SGD(model.parameters(), lr=base_lr)
        trainer.scheduler = torch.optim.lr_scheduler.MultiStepLR(
            optimizer,
            milestones=list(range(1, 5)),
            gamma=factor
        )

        target_lr = base_lr
        for trainer_step_interval in StepIntervalUnit:
            trainer.step_interval = trainer_step_interval
            for step_interval in StepIntervalUnit:
                if step_interval == trainer_step_interval:
                    target_lr *= factor

                trainer._scheduler_step(step_interval=step_interval)
                lr = optimizer.param_groups[0]['lr']
                assert target_lr - 1e-6 <= lr <= target_lr + 1e-6

    def test_train_step(self):
        device = torch.device('cpu')
        model = torch.nn.Linear(1, 1).to(device)
        optimizer = torch.optim.Adam(model.parameters(), lr=1)
        data, targets = torch.Tensor([1.]).to(device), torch.Tensor([1.]).to(device)
        ms = [3, 5, 6]
        params = {
            'metrics': [],
            'device': device,
            'task_type': constants.TABULAR_REGRESSION,
            'labels': torch.Tensor([]),
            'metrics_during_training': False,
            'budget_tracker': BudgetTracker(budget_type=''),
            'criterion': torch.nn.MSELoss,
            'optimizer': optimizer,
            'scheduler': torch.optim.lr_scheduler.MultiStepLR(optimizer, milestones=ms, gamma=2),
            'model': model,
            'step_interval': StepIntervalUnit.epoch
        }
        trainer = StandardTrainer()
        trainer.prepare(**params)

        for _ in range(10):
            trainer.train_step(
                data=data,
                targets=targets
            )
            lr = optimizer.param_groups[0]['lr']
            assert lr == 1

        params.update(step_interval=StepIntervalUnit.batch)
        trainer = StandardTrainer()
        trainer.prepare(**params)

        target_lr = 1
        for i in range(10):
            trainer.train_step(
                data=data,
                targets=targets
            )
            if i + 1 in ms:
                target_lr *= 2

            lr = optimizer.param_groups[0]['lr']
            assert lr == target_lr

    def test_train_epoch_no_step(self):
        """
        This test checks if max runtime is reached
        for an epoch before any train_step has been
        completed. In this case we would like to
        return None for train_loss and an empty
        dictionary for the metrics.
        """
        device = torch.device('cpu')
        model = torch.nn.Linear(1, 1).to(device)
        optimizer = torch.optim.Adam(model.parameters(), lr=1)
        data_loader = unittest.mock.MagicMock(spec=torch.utils.data.DataLoader)
        ms = [3, 5, 6]
        params = {
            'metrics': [],
            'device': device,
            'task_type': constants.TABULAR_REGRESSION,
            'labels': torch.Tensor([]),
            'metrics_during_training': False,
            'budget_tracker': BudgetTracker(budget_type='runtime', max_runtime=0),
            'criterion': torch.nn.MSELoss,
            'optimizer': optimizer,
            'scheduler': torch.optim.lr_scheduler.MultiStepLR(optimizer, milestones=ms, gamma=2),
            'model': model,
            'step_interval': StepIntervalUnit.epoch
        }
        trainer = StandardTrainer()
        trainer.prepare(**params)

        loss, metrics = trainer.train_epoch(
            train_loader=data_loader,
            epoch=0,
            writer=None
        )
        assert loss is None
        assert metrics == {}


class TestStandardTrainer(BaseTraining):
    def test_regression_epoch_training(self, n_samples):
        (trainer,
         _,
         _,
         loader,
         _,
         epochs,
         _) = self.prepare_trainer(n_samples,
                                   StandardTrainer(),
                                   constants.TABULAR_REGRESSION,
                                   OVERFIT_EPOCHS)

        # Train the model
        counter = 0
        r2 = 0
        while r2 < 0.7:
            _, metrics = trainer.train_epoch(loader, epoch=1, writer=None)
            counter += 1
            r2 = metrics['r2']

            if counter > epochs:
                pytest.fail(f"Could not overfit a dummy regression under {epochs} epochs")

    def test_classification_epoch_training(self, n_samples):
        (trainer,
         _,
         _,
         loader,
         _,
         epochs,
         _) = self.prepare_trainer(n_samples,
                                   StandardTrainer(),
                                   constants.TABULAR_CLASSIFICATION,
                                   OVERFIT_EPOCHS)

        # Train the model
        counter = 0
        accuracy = 0
        while accuracy < 0.7:
            _, metrics = trainer.train_epoch(loader, epoch=1, writer=None)
            counter += 1
            accuracy = metrics['accuracy']

            if counter > epochs:
                pytest.fail(f"Could not overfit a dummy classification under {epochs} epochs")


class TestMixUpTrainer(BaseTraining):
    def test_classification_epoch_training(self, n_samples):
        (trainer,
         _,
         _,
         loader,
         _,
         epochs,
         _) = self.prepare_trainer(n_samples,
                                   MixUpTrainer(alpha=0.5),
                                   constants.TABULAR_CLASSIFICATION,
                                   OVERFIT_EPOCHS)

        # Train the model
        counter = 0
        accuracy = 0
        while accuracy < 0.7:
            _, metrics = trainer.train_epoch(loader, epoch=1, writer=None)
            counter += 1
            accuracy = metrics['accuracy']

            if counter > epochs:
                pytest.fail(f"Could not overfit a dummy classification under {epochs} epochs")


class TestTrainer(unittest.TestCase):
    def test_every_trainer_is_valid(self):
        """
        Makes sure that every trainer is a valid estimator.
        That is, we can fully create an object via get/set params.

        This also test that we can properly initialize each one
        of them
        """
        trainer_choice = TrainerChoice(dataset_properties={})

        # Make sure all components are returned
        self.assertEqual(len(trainer_choice.get_components().keys()), 2)

        # For every optimizer in the components, make sure
        # that it complies with the scikit learn estimator.
        # This is important because usually components are forked to workers,
        # so the set/get params methods should recreate the same object
        for name, trainer in trainer_choice.get_components().items():
            config = trainer.get_hyperparameter_search_space().sample_configuration()
            estimator = trainer(**config)
            estimator_clone = clone(estimator)
            estimator_clone_params = estimator_clone.get_params()

            # Make sure all keys are copied properly
            for k in estimator.get_params().keys():
                self.assertIn(k, estimator_clone_params)

            # Make sure the params getter of estimator are honored
            klass = estimator.__class__
            new_object_params = estimator.get_params(deep=False)
            for name, param in new_object_params.items():
                new_object_params[name] = clone(param, safe=False)
            new_object = klass(**new_object_params)
            params_set = new_object.get_params(deep=False)

            for name in new_object_params:
                param1 = new_object_params[name]
                param2 = params_set[name]
                self.assertEqual(param1, param2)

    def test_get_set_config_space(self):
        """Make sure that we can setup a valid choice in the trainer
        choice"""
        trainer_choice = TrainerChoice(dataset_properties={'task_type': 'tabular_classification'})
        cs = trainer_choice.get_hyperparameter_search_space()

        # Make sure that all hyperparameters are part of the serach space
        self.assertListEqual(
            sorted(cs.get_hyperparameter('__choice__').choices),
            sorted(list(trainer_choice.get_components().keys()))
        )

        # Make sure we can properly set some random configs
        # Whereas just one iteration will make sure the algorithm works,
        # doing five iterations increase the confidence. We will be able to
        # catch component specific crashes
        for _ in range(5):
            config = cs.sample_configuration()
            config_dict = copy.deepcopy(config.get_dictionary())
            trainer_choice.set_hyperparameters(config)

            self.assertEqual(trainer_choice.choice.__class__,
                             trainer_choice.get_components()[config_dict['__choice__']])

            # Then check the choice configuration
            selected_choice = config_dict.pop('__choice__', None)
            for key, value in config_dict.items():
                # Remove the selected_choice string from the parameter
                # so we can query in the object for it
                key = key.replace(selected_choice + ':', '')
                self.assertIn(key, vars(trainer_choice.choice))
                self.assertEqual(value, trainer_choice.choice.__dict__[key])


def test_early_stopping():
    dataset_properties = {'task_type': 'tabular_classification', 'output_type': 'binary', 'output_shape': 0}
    trainer_choice = TrainerChoice(dataset_properties=dataset_properties)

    def dummy_performance(*args, **kwargs):
        return (-time.time(), {'accuracy': -time.time()})

    # Fake the training so that the first epoch was the best one
    import time
    trainer_choice.choice = unittest.mock.MagicMock()
    trainer_choice.choice.train_epoch = dummy_performance
    trainer_choice.choice.evaluate = dummy_performance
    trainer_choice.choice.on_epoch_end.return_value = False

    fit_dictionary = {
        'logger_port': 1000,
        'budget_type': 'epochs',
        'epochs': 6,
        'budget': 10,
        'num_run': 1,
        'torch_num_threads': 1,
        'early_stopping': 5,
        'metrics_during_training': True,
        'dataset_properties': dataset_properties,
        'split_id': 0,
        'step_interval': StepIntervalUnit.batch
    }
    for item in ['backend', 'lr_scheduler', 'network', 'optimizer', 'train_data_loader', 'val_data_loader',
                 'device', 'y_train']:
        fit_dictionary[item] = unittest.mock.MagicMock()

    fit_dictionary['backend'].temporary_directory = tempfile.mkdtemp()
    fit_dictionary['network'].state_dict.return_value = {'dummy': 1}
    trainer_choice.fit(fit_dictionary)
    epochs_since_best = trainer_choice.run_summary.get_last_epoch() - trainer_choice.run_summary.get_best_epoch()

    # Six epochs ran
    assert len(trainer_choice.run_summary.performance_tracker['val_metrics']) == 6

    # But the best performance was achieved on the first epoch
    assert epochs_since_best == 0

    # No files are left after training
    left_files = glob.glob(f"{fit_dictionary['backend'].temporary_directory}/*")
    assert len(left_files) == 0
    shutil.rmtree(fit_dictionary['backend'].temporary_directory)


if __name__ == '__main__':
    unittest.main()
