from autoPyTorch.training.base_training import BaseTrainingTechnique

class LrScheduling(BaseTrainingTechnique):
    """Schedule the learning rate with given learning rate scheduler.
    The learning rate scheduler is usually set in a LrSchedulerSelector pipeline node.
    """

    # OVERRIDE
    def after_train_batches(self, training_components, log, epoch):

        # do one step of lr scheduling
        if callable(getattr(training_components["lr_scheduler"], "get_lr", None)):
            log['lr'] = training_components["lr_scheduler"].get_lr()[0]
        try:
            training_components["lr_scheduler"].step(epoch=(epoch + 1), metrics=log['loss'])
        except:
            training_components["lr_scheduler"].step(epoch=(epoch + 1))
        self.logger.debug("Perform learning rate scheduling")

        # check if lr scheduler has converged, if possible
        if not training_components["lr_scheduler"].snapshot_before_restart:
            return False
        training_components["lr_scheduler"].get_lr()
        log["lr_scheduler_converged"] = False
        if training_components["lr_scheduler"].restarted_at == (epoch + 1):
            self.logger.debug("Learning rate scheduler converged. Taking Snapshot of models parameters.")
            training_components["network"].snapshot()
            log["lr_scheduler_converged"] = True
        return False
    
    def select_log(self, logs, training_components):

        # select the log where the lr scheduler has converged, if possible.
        if training_components["lr_scheduler"].snapshot_before_restart:
            self.logger.debug("Using logs where lr scheduler converged")
            logs = [log for log in logs if log["lr_scheduler_converged"]] or logs
            logs = logs[-1]
            return logs
        return False
