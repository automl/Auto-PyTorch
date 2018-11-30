from autonet.training.base_training import BaseTrainingTechnique
import time

class BudgetTypeTime(BaseTrainingTechnique):
    default_min_budget = 120
    default_max_budget = 6000
    compensate = 10 # will be modified by cv

    # OVERRIDE
    def set_up(self, training_components, pipeline_config, logger):
        super(BudgetTypeTime, self).set_up(training_components, pipeline_config, logger)
        self.t = training_components["initial_budget"]
        if self.t == 0:
            self.t = (time.time() - training_components["fit_start_time"])
        
        if self.t >= training_components["budget"] - self.compensate:
            raise Exception("Budget exhausted before training started")


    # OVERRIDE
    def before_train_batches(self, training_components, log, epoch):
        self.epoch_start_time = time.time()
    
    # OVERRIDE
    def during_train_batches(self, batch_loss, training_components):
        delta = (time.time() - self.epoch_start_time)
        return self.t + delta > training_components["budget"] - self.compensate
    
    # OVERRIDE
    def after_train_batches(self, training_components, log, epoch):
        delta = (time.time() - self.epoch_start_time)
        self.t += delta
        training_components["network"].budget_trained = self.t
        self.logger.debug("Budget used: " + str(self.t) + "/" + str(training_components["budget"]))

        if self.t >= training_components["budget"] - self.compensate:
            self.logger.debug("Budget exhausted!")
            return True
        return False

class BudgetTypeEpochs(BaseTrainingTechnique):
    default_min_budget = 5
    default_max_budget = 150
    
    # OVERRIDE
    def set_up(self, training_components, pipeline_config, logger):
        super(BudgetTypeEpochs, self).set_up(training_components, pipeline_config, logger)
        self.t = training_components["initial_budget"]
    
    # OVERRIDE
    def after_train_batches(self, training_components, log, epoch):
        self.t += 1
        training_components["network"].budget_trained = self.t
        self.logger.debug("Budget used: " + str(self.t) + "/" + str(training_components["budget"]))

        if self.t >= training_components["budget"]:
            self.logger.debug("Budget exhausted!")
            return True
        return False