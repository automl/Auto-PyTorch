from autoPyTorch.components.training.base_training import BaseTrainingTechnique
from autoPyTorch.utils.config.config_option import ConfigOption, to_bool

class EarlyStopping(BaseTrainingTechnique):
    """ Stop training when there is no improvement on the validation set for a specified number of epochs.
    Is able to take a snapshot of the parameters, where the performance of the validation set is best.
    There is no further split of the data. Therefore the validation performance reported to BOHB will become an optimistic estimator.
    """

    # OVERRIDE
    def set_up(self, trainer, pipeline_config, **kwargs):
        super(EarlyStopping, self).set_up(trainer, pipeline_config)
        self.reset_parameters = pipeline_config["early_stopping_reset_parameters"]
        self.minimize = pipeline_config["minimize"]
        self.patience = pipeline_config["early_stopping_patience"]

        # does not work with e.g. cosine anealing with warm restarts
        if hasattr(trainer, "lr_scheduler") and not trainer.lr_scheduler.allows_early_stopping:
            self.patience = float("inf")

        # initialize current best performance to +/- infinity
        if trainer.model.current_best_epoch_performance is None:
            trainer.model.current_best_epoch_performance = float("inf")
            if not self.minimize:
                trainer.model.current_best_epoch_performance = -float("inf")

        trainer.logger.debug("Using Early stopping with patience: " + str(self.patience))
        trainer.logger.debug("Reset Parameters to parameters with best validation performance: " + str(self.reset_parameters))
    
    # OVERRIDE
    def on_epoch_end(self, trainer, log, **kwargs):
        if "val_" + trainer.metrics[0] not in log:
            if self.patience < float("inf"):
                trainer.logger.debug("No Early stopping because no validation set performance available")
            return False
        if self.reset_parameters and (not hasattr(trainer, "lr_scheduler") or not trainer.lr_scheduler.snapshot_before_restart):
            log["best_parameters"] = False
        current_performance = log["val_" + trainer.metrics[0]]

        # new best performance
        if ((self.minimize and current_performance < trainer.model.current_best_epoch_performance) or
            (not self.minimize and current_performance > trainer.model.current_best_epoch_performance)):
            trainer.model.num_epochs_no_progress = 0
            trainer.model.current_best_epoch_performance = current_performance
            trainer.logger.debug("New best performance!")

            if self.reset_parameters and (not hasattr(trainer, "lr_scheduler") or not trainer.lr_scheduler.snapshot_before_restart):
                trainer.logger.debug("Early stopping takes snapshot of current parameters")
                log["best_parameters"] = True
                trainer.model.snapshot()

        # do early stopping
        elif trainer.model.num_epochs_no_progress > self.patience:
            trainer.logger.debug("Early stopping patience exhausted. Stopping Early!")
            trainer.model.stopped_early = True
            return True
        
        # no improvement
        else:
            trainer.logger.debug("No improvement")
            trainer.model.num_epochs_no_progress += 1
        return False
    
    # OVERRIDE
    def select_log(self, logs, trainer, **kwargs):
        # select the log where a snapshot has been taken
        if self.reset_parameters and (not hasattr(trainer, "lr_scheduler") or not trainer.lr_scheduler.snapshot_before_restart):
            trainer.logger.debug("Using logs of parameters with best validation performance")
            logs = [log for log in logs if log["best_parameters"]] or logs
            logs = logs[-1]
            return logs
        return False
    
    def requires_eval_each_epoch(self):
        return self.reset_parameters or self.patience < float("inf")
    
    # OVERRIDE
    @staticmethod
    def get_pipeline_config_options():
        options = [
            ConfigOption("early_stopping_patience", default=float("inf"), type=float),
            ConfigOption("early_stopping_reset_parameters", default=False, type=to_bool)
        ]
        return options