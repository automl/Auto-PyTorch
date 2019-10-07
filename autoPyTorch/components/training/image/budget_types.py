from autoPyTorch.components.training.image.base_training import BaseTrainingTechnique
import time

class BudgetTypeTime(BaseTrainingTechnique):
    default_min_budget = 120
    default_max_budget = 6000
    compensate = 10 # will be modified by cv

    # OVERRIDE
    def set_up(self, training_components, pipeline_config, logger):
        super(BudgetTypeTime, self).set_up(training_components, pipeline_config, logger)
        self.end_time = training_components["budget"] - self.compensate + training_components["fit_start_time"]
        self.start_time = time.time()
        
        if self.start_time >= self.end_time:
            raise Exception("Budget exhausted before training started")
    
    # OVERRIDE
    def during_train_batches(self, batch_loss, training_components):
        return time.time() >= self.end_time
    
    # OVERRIDE
    def after_train_batches(self, training_components, log, epoch):
        elapsed = time.time() - self.start_time
        training_components["network"].budget_trained = elapsed
        self.logger.debug("Budget used: " + str(elapsed) + "/" + str(self.end_time - self.start_time))

        if time.time() >= self.end_time:
            self.logger.debug("Budget exhausted!")
            return True
        return False

class BudgetTypeEpochs(BaseTrainingTechnique):
    default_min_budget = 5
    default_max_budget = 150
    
    # OVERRIDE
    def set_up(self, training_components, pipeline_config, logger):
        super(BudgetTypeEpochs, self).set_up(training_components, pipeline_config, logger)
        self.target = training_components["budget"]
    
    # OVERRIDE
    def after_train_batches(self, training_components, log, epoch):
        training_components["network"].budget_trained = epoch
        self.logger.debug("Budget used: " + str(epoch) + "/" + str(self.target))

        if epoch >= self.target:
            self.logger.debug("Budget exhausted!")
            return True
        return False
