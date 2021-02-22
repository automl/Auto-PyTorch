import copy
import os
import sys
import unittest
import unittest.mock

import numpy as np

from sklearn.base import clone

import torch

from autoPyTorch import constants
from autoPyTorch.pipeline.components.training.data_loader.base_data_loader import (
    BaseDataLoaderComponent,
)
from autoPyTorch.pipeline.components.training.trainer.MixUpTrainer import (
    MixUpTrainer
)
from autoPyTorch.pipeline.components.training.trainer.StandardTrainer import (
    StandardTrainer
)
from autoPyTorch.pipeline.components.training.trainer.base_trainer import (
    BaseTrainerComponent, )
from autoPyTorch.pipeline.components.training.trainer.base_trainer_choice import (
    TrainerChoice,
)

sys.path.append(os.path.dirname(__file__))
from base import BaseTraining  # noqa (E402: module level import not at top of file)


class BaseDataLoaderTest(unittest.TestCase):
    def test_get_set_config_space(self):
        """
        Makes sure that the configuration space of the base data loader
        is properly working"""
        loader = BaseDataLoaderComponent()

        cs = loader.get_hyperparameter_search_space()

        # Make sure that the batch size is a valid hyperparameter
        self.assertEqual(cs.get_hyperparameter('batch_size').default_value, 64)

        # Make sure we can properly set some random configs
        for i in range(5):
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
        datamanager.get_dataset_for_training.return_value = (dataset, dataset)
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


class BaseTrainerComponentTest(BaseTraining, unittest.TestCase):

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
         logger) = self.prepare_trainer(BaseTrainerComponent(),
                                        constants.TABULAR_CLASSIFICATION)

        prev_loss, prev_metrics = trainer.evaluate(loader, epoch=1, writer=None)
        self.assertIn('accuracy', prev_metrics)

        # Fit the model
        self.train_model(model,
                         optimizer,
                         loader,
                         criterion,
                         epochs)

        # Loss and metrics should have improved after fit
        # And the prediction should be better than random
        loss, metrics = trainer.evaluate(loader, epoch=1, writer=None)
        self.assertGreater(prev_loss, loss)
        self.assertGreater(metrics['accuracy'], prev_metrics['accuracy'])
        self.assertGreater(metrics['accuracy'], 0.5)


class StandardTrainerTest(BaseTraining, unittest.TestCase):

    def test_regression_epoch_training(self):
        (trainer,
         _,
         _,
         loader,
         _,
         epochs,
         logger) = self.prepare_trainer(StandardTrainer(),
                                        constants.TABULAR_REGRESSION)

        # Train the model
        counter = 0
        r2 = 0
        while r2 < 0.7:
            loss, metrics = trainer.train_epoch(loader, epoch=1, logger=logger, writer=None)
            counter += 1
            r2 = metrics['r2']

            if counter > epochs:
                self.fail(f"Could not overfit a dummy regression under {epochs} epochs")

    def test_classification_epoch_training(self):
        (trainer,
         _,
         _,
         loader,
         _,
         epochs,
         logger) = self.prepare_trainer(StandardTrainer(),
                                        constants.TABULAR_CLASSIFICATION)

        # Train the model
        counter = 0
        accuracy = 0
        while accuracy < 0.7:
            loss, metrics = trainer.train_epoch(loader, epoch=1, logger=logger, writer=None)
            counter += 1
            accuracy = metrics['accuracy']

            if counter > epochs:
                self.fail(f"Could not overfit a dummy classification under {epochs} epochs")


class MixUpTrainerTest(BaseTraining, unittest.TestCase):
    def test_classification_epoch_training(self):
        (trainer,
         _,
         _,
         loader,
         _,
         epochs,
         logger) = self.prepare_trainer(MixUpTrainer(alpha=0.5),
                                        constants.TABULAR_CLASSIFICATION)

        # Train the model
        counter = 0
        accuracy = 0
        while accuracy < 0.7:
            loss, metrics = trainer.train_epoch(loader, epoch=1, logger=logger, writer=None)
            counter += 1
            accuracy = metrics['accuracy']

            if counter > epochs:
                self.fail(f"Could not overfit a dummy classification under {epochs} epochs")


class TrainerTest(unittest.TestCase):
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
            for k, v in estimator.get_params().items():
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
        for i in range(5):
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


if __name__ == '__main__':
    unittest.main()
