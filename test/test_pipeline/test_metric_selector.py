__author__ = "Max Dippel, Michael Burkart and Matthias Urban"
__version__ = "0.0.1"
__license__ = "BSD"

import unittest
from autonet.pipeline.base.pipeline import Pipeline
from autonet.pipeline.nodes.metric_selector import MetricSelector

from autonet.components.metrics.standard_metrics import accuracy, auc_metric, mean_distance

class TestMetricSelectorMethods(unittest.TestCase):

    def test_selector(self):
        pipeline = Pipeline([
            MetricSelector()
        ])

        selector = pipeline[MetricSelector.get_name()]
        selector.add_metric("auc", auc_metric)
        selector.add_metric("accuracy", accuracy)
        selector.add_metric("mean", mean_distance)

        pipeline_config = pipeline.get_pipeline_config(train_metric="accuracy", additional_metrics=['auc', 'mean'])
        pipeline.fit_pipeline(pipeline_config=pipeline_config)

        selected_train_metric = selector.fit_output['train_metric']
        selected_additional_metrics = selector.fit_output['additional_metrics']

        self.assertEqual(selected_train_metric, accuracy)
        self.assertSetEqual(set(selected_additional_metrics), set([auc_metric, mean_distance]))



