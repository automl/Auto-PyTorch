from autoPyTorch.components.training.base_training import BaseTrainingTechnique
import time

class BudgetTypeTime(BaseTrainingTechnique):
    default_min_budget = 120
    default_max_budget = 6000
    compensate = 10 # will be modified by cv

    # OVERRIDE
    def set_up(self, trainer, pipeline_config, **kwargs):
        super(BudgetTypeTime, self).set_up(trainer, pipeline_config)
        self.end_time = trainer.budget - self.compensate + trainer.fit_start_time
        self.start_time = time.time()
        
        if self.start_time >= self.end_time:
            raise Exception("Budget exhausted before training started")
    
    # OVERRIDE
    def on_batch_end(self, **kwargs):
        return time.time() >= self.end_time
    
    # OVERRIDE
    def on_epoch_end(self, trainer, **kwargs):
        elapsed = time.time() - self.start_time
        trainer.model.budget_trained = elapsed
        trainer.logger.debug("Budget used: " + str(elapsed) + "/" + str(self.end_time - self.start_time))

        if time.time() >= self.end_time:
            trainer.logger.debug("Budget exhausted!")
            return True
        return False

class BudgetTypeEpochs(BaseTrainingTechnique):
    default_min_budget = 5
    default_max_budget = 150
    
    # OVERRIDE
    def set_up(self, trainer, pipeline_config, **kwargs):
        super(BudgetTypeEpochs, self).set_up(trainer, pipeline_config)
        self.target = trainer.budget
    
    # OVERRIDE
    def on_epoch_end(self, trainer, epoch, **kwargs):
        trainer.model.budget_trained = epoch
        trainer.logger.debug("Budget used: " + str(epoch) + "/" + str(self.target))

        if epoch >= self.target:
            trainer.logger.debug("Budget exhausted!")
            return True
        return False