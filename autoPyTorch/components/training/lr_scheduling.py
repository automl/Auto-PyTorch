from autoPyTorch.components.training.base_training import BaseTrainingTechnique
import time

class LrScheduling(BaseTrainingTechnique):
    """Schedule the learning rate with given learning rate scheduler.
    The learning rate scheduler is usually set in a LrSchedulerSelector pipeline node.
    """
    def __init__(self, training_components, lr_step_after_batch, lr_step_with_time, allow_snapshot):
        super(LrScheduling, self).__init__(training_components=training_components)
        self.lr_step_after_batch = lr_step_after_batch
        self.lr_step_with_time = lr_step_with_time
        self.allow_snapshot = allow_snapshot

    # OVERRIDE
    def on_batch_end(self, batch_loss, trainer, epoch, step, num_steps, **kwargs):
        if not self.lr_step_after_batch:
            return

        if self.lr_step_with_time:
            self.perform_scheduling(trainer, time.time() - trainer.fit_start_time, batch_loss)
        else:
            self.perform_scheduling(trainer, (epoch - 1) + ((step + 1) / num_steps), batch_loss)

    # OVERRIDE
    def on_epoch_end(self, trainer, epoch, log, **kwargs):
        log["lr_scheduler_converged"] = False
        if callable(getattr(trainer.lr_scheduler, "get_lr", None)):
            log['lr'] = trainer.lr_scheduler.get_lr()[0]

        if self.lr_step_after_batch:
            return

        if self.lr_step_with_time:
            if isinstance(log['loss'], list):
                log["lr_scheduler_converged"] = self.perform_scheduling(trainer, time.time() - trainer.fit_start_time, log['loss'][-1])
            else:
                log["lr_scheduler_converged"] = self.perform_scheduling(trainer, time.time() - trainer.fit_start_time, log['loss'])
        else:
            if isinstance(log['loss'], list):
                log["lr_scheduler_converged"]  = self.perform_scheduling(trainer, epoch, log['loss'][-1])
            else:
                log["lr_scheduler_converged"] = self.perform_scheduling(trainer, epoch, log['loss'])
        return False
    
    def perform_scheduling(self, trainer, epoch, metric, **kwargs):
        try:
            trainer.lr_scheduler.step(epoch=epoch, metrics=metric)
        except:
            trainer.lr_scheduler.step(epoch=epoch)
        trainer.logger.debug("Perform learning rate scheduling")

        # check if lr scheduler has converged, if possible
        if not trainer.lr_scheduler.snapshot_before_restart:
            return False
        trainer.lr_scheduler.get_lr()
        if trainer.lr_scheduler.restarted_at == epoch:
            if self.allow_snapshot:
                trainer.logger.debug("Learning rate scheduler converged. Taking Snapshot of models parameters.")
                trainer.model.snapshot()
            return True
        return False

    def select_log(self, logs, trainer, **kwargs):
        # select the log where the lr scheduler has converged, if possible.
        if trainer.lr_scheduler.snapshot_before_restart:
            trainer.logger.debug("Using logs where lr scheduler converged")
            logs = [log for log in logs if log["lr_scheduler_converged"]] or logs
            logs = logs[-1]
            return logs
        return False
