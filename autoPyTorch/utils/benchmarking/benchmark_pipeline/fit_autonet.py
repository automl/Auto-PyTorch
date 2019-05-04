import time
import logging
from autoPyTorch.pipeline.base.pipeline_node import PipelineNode

class FitAutoNet(PipelineNode):

    def fit(self, autonet, data_manager, **kwargs):
        start_time = time.time()

        logging.getLogger('benchmark').debug("Fit autonet")

        fit_result = autonet.fit(
            data_manager.X_train, data_manager.Y_train, 
            data_manager.X_valid, data_manager.Y_valid, 
            refit=False)

        return { 'fit_duration': int(time.time() - start_time), 
                 'fit_result': fit_result }