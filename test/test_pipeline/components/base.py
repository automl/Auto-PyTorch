import logging
import unittest

from sklearn.datasets import make_classification

import torch

from autoPyTorch.constants import STRING_TO_TASK_TYPES
from autoPyTorch.pipeline.components.training.metrics.utils import get_metrics
from autoPyTorch.pipeline.components.training.trainer.base_trainer import BudgetTracker


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
        self.criterion = torch.nn.CrossEntropyLoss()
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

    def _overfit_model(self):
        self.model.train()
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
