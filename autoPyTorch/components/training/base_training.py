import ConfigSpace

class BaseTrainingTechnique():
    def __init__(self, training_components=None):
        """Initialize the training technique. Should be called in a fit Method of a Pipeline node.
        
        Keyword Arguments:
            training_components {dict} -- Maps a names to a training components necessary for this training technique (default: {None})
        """
        self.training_components = training_components or dict()

    # VIRTUAL
    def set_up(self, trainer, pipeline_config):
        """Set up the training component
        
        Arguments:
            trainer {Trainer} -- The trainer object used for training.
            pipeline_config {dict} -- Configuration of the Pipeline.
            logger {Logger} -- Logger.
        """

        pass
    
    # VIRTUAL
    def on_epoch_start(self, trainer, log, epoch):
        """Function that gets called before the train_batches method of each epoch in training.
        
        Arguments:
            trainer {Trainer} -- The trainer object used for training.
            log {dict} -- The log of the current epoch.
            epoch {int} -- The current epoch of training.
        """

        pass

    # VIRTUAL
    def on_epoch_end(self, trainer, log, epoch):
        """Function that gets called after the train_batches method of each epoch in training.
        Is able to stop training by returning True.
        
        Arguments:
            trainer {Trainer} -- The trainer object used for training.
            log {dict} -- The log of the current epoch.
            epoch {int} -- The current epoch of training.
        
        Returns:
            bool -- If training should be stopped.
        """

        return False

    # VIRTUAL
    def on_batch_start(self, trainer, epoch, step, num_steps):
        """Function that gets called in the train_batches method of training.
        Is able to cancel the current epoch by returning True.
        
        Arguments:
            batch_loss {tensor} -- The batch loss of the current batch.
            trainer {Trainer} -- The trainer object used for training
        
        Returns:
            bool -- If the current epoch should be canceled.
        """

        return False
    
        # VIRTUAL
    def on_batch_end(self, batch_loss, trainer, epoch, step, num_steps):
        """Function that gets called in the train_batches method of training.
        Is able to cancel the current epoch by returning True.
        
        Arguments:
            batch_loss {tensor} -- The batch loss of the current batch.
            trainer {Trainer} -- The trainer object used for training
        
        Returns:
            bool -- If the current epoch should be canceled.
        """

        return False
    
    # VIRTUAL
    def select_log(self, logs, trainer):
        """Select one log from the list of all epoch logs.
        
        Arguments:
            logs {list} -- A list of log. For each epoch of training there is one entry.
            trainer {Trainer} -- The trainer object used for training
        
        Returns:
            log -- The selected log. Return None if undecided.
        """

        return False
    
        # VIRTUAL
    def requires_eval_each_epoch(self):
        """ Specify if the training technique needs the network to be evaluated on a snapshot after training.
        
        Return:
            bool -- If the training technique needs the network to be evaluated on a snapshot after training
        """

        return False


    # VIRTUAL
    @staticmethod
    def get_pipeline_config_options():
        """Return a list of ConfigOption used for this training technique.
        
        Returns:
            list -- A list of ConfigOptions.
        """

        return []


class BaseBatchLossComputationTechnique():

    # VIRTUAL 
    def set_up(self, pipeline_config, hyperparameter_config, logger):
        """Initialize the batch loss computation technique.
        
        Arguments:
            pipeline_config {dict} -- The configuration of the pipeline.
            hyperparameter_config {dict} -- The hyperparameter config sampled by BOHB.
            logger {Logger} -- Logger.
        """
        self.logger = logger
    
    # VIRTUAL
    def prepare_data(self, X_batch, y_batch):
        """Method that gets called, before batch is but into network.
        
        Arguments:
            X_batch {tensor} -- The features of the batch.
            y_batch {tensor} -- The targets of the batch.
        """

        return X_batch, {'y_batch' : y_batch}
    
    # VIRTUAL
    def criterion(self, y_batch):
        return lambda criterion, pred: criterion(pred, y_batch)
    
    # VIRTUAL
    @staticmethod
    def get_hyperparameter_search_space(**pipeline_config):
        """Get the hyperparameter config space for this technique.
        
        Returns:
            ConfigurationSpace -- The hyperparameter config space for this technique
        """

        return ConfigSpace.ConfigurationSpace()


        