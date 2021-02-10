import logging
import unittest

from sklearn.datasets import make_classification, make_regression

import torch

from autoPyTorch import constants
from autoPyTorch.pipeline.components.training.metrics.utils import get_metrics
from autoPyTorch.pipeline.components.training.trainer.base_trainer import BaseTrainerComponent, BudgetTracker


class BaseTraining(unittest.TestCase):

    def prepare_trainer(self,
                        trainer: BaseTrainerComponent,
                        task_type: int):
        if task_type in constants.CLASSIFICATION_TASKS:
            X, y = make_classification(
                n_samples=5000,
                n_features=4,
                n_informative=3,
                n_redundant=1,
                n_repeated=0,
                n_classes=2,
                n_clusters_per_class=2,
                shuffle=True,
                random_state=0
            )
            X = torch.tensor(X, dtype=torch.float)
            y = torch.tensor(y, dtype=torch.long)
            output_type = constants.BINARY
            num_outputs = 2
            criterion = torch.nn.CrossEntropyLoss()

        elif task_type in constants.REGRESSION_TASKS:
            X, y = make_regression(
                n_samples=5000,
                n_features=4,
                n_informative=3,
                n_targets=1,
                shuffle=True,
                random_state=0
            )
            X = torch.tensor(X, dtype=torch.float)
            y = torch.tensor(y, dtype=torch.float)
            # normalize targets for regression since NNs are better when predicting small outputs
            y = ((y - y.mean()) / y.std()).unsqueeze(1)
            output_type = constants.CONTINUOUS
            num_outputs = 1
            criterion = torch.nn.MSELoss(reduction="sum")

        else:
            raise ValueError(f"task type {task_type} not supported for standard trainer test")

        dataset = torch.utils.data.TensorDataset(X, y)
        loader = torch.utils.data.DataLoader(dataset, batch_size=20)
        dataset_properties = {
            'task_type': constants.TASK_TYPES_TO_STRING[task_type],
            'output_type': constants.OUTPUT_TYPES_TO_STRING[output_type]
        }

        # training requirements
        model = torch.nn.Sequential(
            torch.nn.Linear(4, 4),
            torch.nn.Sigmoid(),
            torch.nn.Linear(4, num_outputs)
        )

        optimizer = torch.optim.SGD(model.parameters(), lr=0.01)
        device = torch.device('cpu')
        logger = logging.getLogger('StandardTrainer - test')
        metrics = get_metrics(dataset_properties)
        epochs = 1000
        budget_tracker = BudgetTracker(
            budget_type='epochs',
            max_epochs=epochs,
        )

        trainer.prepare(
            scheduler=None,
            model=model,
            metrics=metrics,
            criterion=criterion,
            budget_tracker=budget_tracker,
            optimizer=optimizer,
            device=device,
            metrics_during_training=True,
            task_type=task_type
        )
        return trainer, model, optimizer, loader, criterion, epochs, logger

    def train_model(self,
                    model: torch.nn.Module,
                    optimizer: torch.optim.Optimizer,
                    loader: torch.utils.data.DataLoader,
                    criterion: torch.nn.Module,
                    epochs: int):
        model.train()
        for epoch in range(epochs):
            for X, y in loader:
                optimizer.zero_grad()
                # Forward pass
                y_pred = model(X)
                # Compute Loss
                loss = criterion(y_pred, y)

                # Backward pass
                loss.backward()
                optimizer.step()
