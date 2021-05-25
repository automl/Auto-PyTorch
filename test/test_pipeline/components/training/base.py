import logging

from sklearn.datasets import make_classification, make_regression
from sklearn.preprocessing import StandardScaler

import torch

from autoPyTorch.constants import (
    BINARY,
    CLASSIFICATION_TASKS,
    CONTINUOUS,
    OUTPUT_TYPES_TO_STRING,
    REGRESSION_TASKS,
    TASK_TYPES_TO_STRING
)
from autoPyTorch.pipeline.components.training.metrics.utils import get_metrics
from autoPyTorch.pipeline.components.training.trainer.base_trainer import BaseTrainerComponent, BudgetTracker


class BaseTraining:

    def prepare_trainer(self,
                        n_samples: int,
                        trainer: BaseTrainerComponent,
                        task_type: int,
                        epochs=50):
        # make this test reproducible
        torch.manual_seed(1)
        if task_type in CLASSIFICATION_TASKS:
            X, y = make_classification(
                n_samples=n_samples,
                n_features=4,
                n_informative=3,
                n_redundant=1,
                n_repeated=0,
                n_classes=2,
                n_clusters_per_class=2,
                class_sep=3.0,
                shuffle=True,
                random_state=0
            )
            X = StandardScaler().fit_transform(X)
            X = torch.tensor(X, dtype=torch.float)
            y = torch.tensor(y, dtype=torch.long)
            output_type = BINARY
            num_outputs = 2
            criterion = torch.nn.CrossEntropyLoss

        elif task_type in REGRESSION_TASKS:
            X, y = make_regression(
                n_samples=n_samples,
                n_features=4,
                n_informative=3,
                n_targets=1,
                shuffle=True,
                random_state=0
            )
            X = StandardScaler().fit_transform(X)
            X = torch.tensor(X, dtype=torch.float)
            y = torch.tensor(y, dtype=torch.float)
            # normalize targets for regression since NNs are better when predicting small outputs
            y = ((y - y.mean()) / y.std()).unsqueeze(1)
            output_type = CONTINUOUS
            num_outputs = 1
            criterion = torch.nn.MSELoss

        else:
            raise ValueError(f"task type {task_type} not supported for standard trainer test")

        dataset = torch.utils.data.TensorDataset(X, y)
        loader = torch.utils.data.DataLoader(dataset, batch_size=20)
        dataset_properties = {
            'task_type': TASK_TYPES_TO_STRING[task_type],
            'output_type': OUTPUT_TYPES_TO_STRING[output_type]
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
        epochs = epochs
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
            task_type=task_type,
            labels=y
        )
        return trainer, model, optimizer, loader, criterion, epochs, logger

    def train_model(self,
                    model: torch.nn.Module,
                    optimizer: torch.optim.Optimizer,
                    loader: torch.utils.data.DataLoader,
                    criterion: torch.nn.Module,
                    epochs: int):
        model.train()

        criterion = criterion() if not isinstance(criterion, torch.nn.Module) else criterion
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
