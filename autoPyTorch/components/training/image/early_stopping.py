from autoPyTorch.components.training.image.base_training import BaseTrainingTechnique
from autoPyTorch.utils.config.config_option import ConfigOption, to_bool

class EarlyStopping(BaseTrainingTechnique):
    """ Stop training when there is no improvement on the validation set for a specified number of epochs.
    Is able to take a snapshot of the parameters, where the performance of the validation set is best.
    There is no further split of the data. Therefore the validation performance reported to BOHB will become an optimistic estimator.
    """

    # OVERRIDE
    def set_up(self, training_components, pipeline_config, logger):
        super(EarlyStopping, self).set_up(training_components, pipeline_config, logger)
        self.reset_parameters = pipeline_config["early_stopping_reset_parameters"]
        self.minimize = pipeline_config["minimize"]
        self.patience = pipeline_config["early_stopping_patience"]

        # does not work with e.g. cosine anealing with warm restarts
        if "lr_scheduler" in training_components and not training_components["lr_scheduler"].allows_early_stopping:
            self.patience = float("inf")

        # initialize current best performance to +/- infinity
        if training_components["network"].current_best_epoch_performance is None:
            training_components["network"].current_best_epoch_performance = float("inf")
            if not self.minimize:
                training_components["network"].current_best_epoch_performance = -float("inf")

        self.logger.debug("Using Early stopping with patience: " + str(self.patience))
        self.logger.debug("Reset Parameters to parameters with best validation performance: " + str(self.reset_parameters))
    
    # OVERRIDE
    def after_train_batches(self, training_components, log, epoch):
        if "val_" + training_components["train_metric_name"] not in log:
            if self.patience < float("inf"):
                self.logger.debug("No Early stopping because no validation set performance available")
            return False
        if self.reset_parameters and ("lr_scheduler" not in training_components or not training_components["lr_scheduler"].snapshot_before_restart):
            log["best_parameters"] = False
        current_performance = log["val_" + training_components["train_metric_name"]]

        # new best performance
        if ((self.minimize and current_performance < training_components["network"].current_best_epoch_performance) or
            (not self.minimize and current_performance > training_components["network"].current_best_epoch_performance)):
            training_components["network"].num_epochs_no_progress = 0
            training_components["network"].current_best_epoch_performance = current_performance
            self.logger.debug("New best performance!")

            if self.reset_parameters and ("lr_scheduler" not in training_components or not training_components["lr_scheduler"].snapshot_before_restart):
                self.logger.debug("Early stopping takes snapshot of current parameters")
                log["best_parameters"] = True
                training_components["network"].snapshot()

        # do early stopping
        elif training_components["network"].num_epochs_no_progress > self.patience:
            self.logger.debug("Early stopping patience exhausted. Stopping Early!")
            training_components["network"].stopped_early = True
            return True
        
        # no improvement
        else:
            self.logger.debug("No improvement")
            training_components["network"].num_epochs_no_progress += 1
        return False
    
    # OVERRIDE
    def select_log(self, logs, training_components):
        # select the log where a snapshot has been taken
        if self.reset_parameters and ("lr_scheduler" not in training_components or not training_components["lr_scheduler"].snapshot_before_restart):
            self.logger.debug("Using logs of parameters with best validation performance")
            logs = [log for log in logs if log["best_parameters"]] or logs
            logs = logs[-1]
            return logs
        return False
    
    def needs_eval_on_valid_each_epoch(self):
        return self.reset_parameters or self.patience < float("inf")
    
    # OVERRIDE
    @staticmethod
    def get_pipeline_config_options():
        options = [
            ConfigOption("early_stopping_patience", default=float("inf"), type=float),
            ConfigOption("early_stopping_reset_parameters", default=False, type=to_bool)
        ]
        return options
