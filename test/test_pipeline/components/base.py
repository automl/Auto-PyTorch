import logging
import unittest
from typing import Any, Dict, List, Optional, Tuple

from sklearn.datasets import make_classification, make_regression

import torch

from autoPyTorch.constants import BINARY, CLASSIFICATION_TASKS, CONTINUOUS, OUTPUT_TYPES_TO_STRING, REGRESSION_TASKS, \
    TASK_TYPES_TO_STRING
from autoPyTorch.pipeline.components.base_choice import autoPyTorchChoice
from autoPyTorch.pipeline.components.preprocessing.tabular_preprocessing.TabularColumnTransformer import \
    TabularColumnTransformer
from autoPyTorch.pipeline.components.preprocessing.tabular_preprocessing.encoding.base_encoder_choice import \
    EncoderChoice
from autoPyTorch.pipeline.components.preprocessing.tabular_preprocessing.imputation.SimpleImputer import SimpleImputer
from autoPyTorch.pipeline.components.preprocessing.tabular_preprocessing.scaling.base_scaler_choice import ScalerChoice
from autoPyTorch.pipeline.components.training.metrics.utils import get_metrics
from autoPyTorch.pipeline.components.training.trainer.base_trainer import BaseTrainerComponent, BudgetTracker
from autoPyTorch.pipeline.tabular_classification import TabularClassificationPipeline


class BaseTraining(unittest.TestCase):

    def prepare_trainer(self,
                        trainer: BaseTrainerComponent,
                        task_type: int):
        if task_type in CLASSIFICATION_TASKS:
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
            output_type = BINARY
            num_outputs = 2
            criterion = torch.nn.CrossEntropyLoss()

        elif task_type in REGRESSION_TASKS:
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
            output_type = CONTINUOUS
            num_outputs = 1
            criterion = torch.nn.MSELoss(reduction="sum")

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


class TabularPipeline(TabularClassificationPipeline):
    def _get_pipeline_steps(self, dataset_properties: Optional[Dict[str, Any]],
                            ) -> List[Tuple[str, autoPyTorchChoice]]:
        """
        Defines what steps a pipeline should follow.
        The step itself has choices given via autoPyTorchChoice.

        Returns:
            List[Tuple[str, autoPyTorchChoice]]: list of steps sequentially exercised
                by the pipeline.
        """
        steps = []  # type: List[Tuple[str, autoPyTorchChoice]]

        default_dataset_properties = {'target_type': 'tabular_classification'}
        if dataset_properties is not None:
            default_dataset_properties.update(dataset_properties)

        steps.extend([
            ("imputer", SimpleImputer()),
            ("encoder", EncoderChoice(default_dataset_properties)),
            ("scaler", ScalerChoice(default_dataset_properties)),
            ("tabular_transformer", TabularColumnTransformer()),
        ])
        return steps
