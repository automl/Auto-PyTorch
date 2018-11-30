import ConfigSpace

class BaseTrainingTechnique():
    def __init__(self, training_components=None):
        """Initialize the training technique. Should be called in a fit Method of a Pipeline node.
        
        Keyword Arguments:
            training_components {dict} -- Maps a names to a training components necessary for this training technique (default: {None})
        """

        self.training_components = training_components or dict()
    
    # VIRTUAL
    def set_up(self, training_components, pipeline_config, logger):
        """Set up the training component
        
        Arguments:
            training_components {dict} -- All training components of training.
            pipeline_config {dict} -- Configuration of the Pipeline.
            logger {Logger} -- Logger.
        """

        self.logger = logger
    
    # VIRTUAL
    def before_train_batches(self, training_components, log, epoch):
        """Function that gets called before the train_batches method of each epoch in training.
        
        Arguments:
            training_components {dict} -- All training components used in training.
            log {dict} -- The log of the current epoch.
            epoch {int} -- The current epoch of training.
        """

        pass

    # VIRTUAL
    def after_train_batches(self, training_components, log, epoch):
        """Function that gets called after the train_batches method of each epoch in training.
        Is able to stop training by returning True.
        
        Arguments:
            training_components {dict} -- All training components used in training.
            log {dict} -- The log of the current epoch.
            epoch {int} -- The current epoch of training.
        
        Returns:
            bool -- If training should be stopped.
        """

        return False

    # VIRTUAL
    def during_train_batches(self, batch_loss, training_components):
        """Function that gets called in the train_batches method of training.
        Is able to cancel the current epoch by returning True.
        
        Arguments:
            batch_loss {tensor} -- The batch loss of the current batch.
            training_components {dict} -- All training components used in training.
        
        Returns:
            bool -- If the current epoch should be canceled.
        """

        return False
    
    # VIRTUAL
    def select_log(self, logs, training_components):
        """Select one log from the list of all epoch logs.
        
        Arguments:
            logs {list} -- A list of log. For each epoch of training there is one entry.
            training_components {dict} -- All training components used in training.
        
        Returns:
            log -- The selected log. Return None if undecided.
        """

        return False

    # VIRTUAL
    def needs_eval_on_valid_each_epoch(self):
        """Specify if the training technique needs the network to be evaluated on the validation set.
        
        Returns:
            bool -- If the network should be evaluated on the validation set.
        """

        return False
    
    # VIRTUAL
    def needs_eval_on_train_each_epoch(self):
        """Specify if the training technique needs the network to be evaluated on the training set.
        
        Returns:
            bool -- If the network should be evaluated on the training set.
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
    def prepare_batch_data(self, X_batch, y_batch):
        """Method that gets called, before batch is but into network.
        
        Arguments:
            X_batch {tensor} -- The features of the batch.
            y_batch {tensor} -- The targets of the batch.
        """

        self.y_batch = y_batch
    
    # VIRTUAL
    def compute_batch_loss(self, loss_function, y_batch_pred):
        """Method that computes the batch loss.
        
        Arguments:
            loss_function {torch.nn._Loss} -- Loss function.
            y_batch_pred {tensor} -- The prediction of the network for the batch.
        
        Returns:
            tensor -- The batch loss.
        """

        result = loss_function(y_batch_pred, self.y_batch)
        self.y_batch = None
        return result
    
    # VIRTUAL
    @staticmethod
    def get_pipeline_config_options():
        """A list of ConfigOptions used for this technique.
        
        Returns:
            list -- A list of ConfigOptions for this technique.
        """

        return []
    
    # VIRTUAL
    @staticmethod
    def get_hyperparameter_search_space(**pipeline_config):
        """Get the hyperparameter config space for this technique.
        
        Returns:
            ConfigurationSpace -- The hyperparameter config space for this technique
        """

        return ConfigSpace.ConfigurationSpace()


        