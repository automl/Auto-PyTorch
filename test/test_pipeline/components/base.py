import logging
import unittest
from typing import Any, Dict, List, Optional, Tuple

from sklearn.datasets import make_classification

import torch

from autoPyTorch.constants import STRING_TO_OUTPUT_TYPES, STRING_TO_TASK_TYPES
from autoPyTorch.pipeline.components.base_choice import autoPyTorchChoice
from autoPyTorch.pipeline.components.preprocessing.tabular_preprocessing.TabularColumnTransformer import \
    TabularColumnTransformer
from autoPyTorch.pipeline.components.preprocessing.tabular_preprocessing.encoding.base_encoder_choice import \
    EncoderChoice
from autoPyTorch.pipeline.components.preprocessing.tabular_preprocessing.imputation.SimpleImputer import SimpleImputer
from autoPyTorch.pipeline.components.preprocessing.tabular_preprocessing.scaling.base_scaler_choice import ScalerChoice
from autoPyTorch.pipeline.components.training.metrics.utils import get_metrics
from autoPyTorch.pipeline.components.training.trainer.base_trainer import BudgetTracker
from autoPyTorch.pipeline.tabular_classification import TabularClassificationPipeline


class BaseTraining(unittest.TestCase):

    def setUp(self):
        # Data
        self.X, self.y = make_classification(
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
        self.X = torch.FloatTensor(self.X)
        self.y = torch.LongTensor(self.y)
        self.dataset = torch.utils.data.TensorDataset(self.X, self.y)
        self.loader = torch.utils.data.DataLoader(self.dataset, batch_size=20)
        self.dataset_properties = {
            'task_type': 'tabular_classification',
            'output_type': 'binary'
        }

        # training requirements
        layers = []
        layers.append(torch.nn.Linear(4, 4))
        layers.append(torch.nn.Sigmoid())
        layers.append(torch.nn.Linear(4, 2))
        self.model = torch.nn.Sequential(*layers)
        self.criterion = torch.nn.CrossEntropyLoss
        self.optimizer = torch.optim.SGD(self.model.parameters(), lr=0.01)
        self.device = torch.device('cpu')
        self.logger = logging.getLogger('test')
        self.metrics = get_metrics(self.dataset_properties)
        self.epochs = 20
        self.budget_tracker = BudgetTracker(
            budget_type='epochs',
            max_epochs=self.epochs,
        )
        self.task_type = STRING_TO_TASK_TYPES[self.dataset_properties['task_type']]
        self.output_type = STRING_TO_OUTPUT_TYPES[self.dataset_properties['output_type']]

    def _overfit_model(self):
        self.model.train()
        # initialise the criterion as it is
        # not being done in __init__
        self.criterion = self.criterion()
        for epoch in range(self.epochs):
            total_loss = 0
            for x, y in self.loader:
                self.optimizer.zero_grad()
                # Forward pass
                y_pred = self.model(self.X)
                # Compute Loss
                loss = self.criterion(y_pred.squeeze(), self.y)
                total_loss += loss

                # Backward pass
                loss.backward()
                self.optimizer.step()


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
