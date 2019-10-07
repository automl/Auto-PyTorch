import os
import json
import time
import logging

from hpbandster.core.result import logged_results_to_HBS_result
from autoPyTorch.pipeline.base.pipeline_node import PipelineNode

class SaveResults(PipelineNode):

    def fit(self, result_dir, fit_duration, test_score, fit_result, autonet, task_id):
        if (task_id not in [-1, 1]):
            time.sleep(60)
            return dict()

        logging.getLogger('benchmark').info("Create and save summary")

        summary = {
            "incumbent_config": fit_result["optimized_hyperparameter_config"],
            "budget": fit_result["budget"],
            "loss": fit_result["loss"],
            "test_score": test_score,
            "incumbent_config" : incumbent_config,
            "info": fit_result["info"],
            "duration": fit_duration,
            }

        if "ensemble_configs" in fit_result:
            summary["ensemble_configs"] = list(fit_result["ensemble_configs"].values())

        # write as json
        with open(os.path.join(result_dir, "summary.json"), "w") as f:
            json.dump(summary, f)

        return dict()
