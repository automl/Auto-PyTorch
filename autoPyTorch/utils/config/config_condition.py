class ConfigCondition():
    def __init__(self, name, check):
        """Initialize the Condition
        
        Arguments:
            name {str} -- Name of the condition. Will be displayed if condition is violated.
            check {callable} -- takes a pipeline config and returns False, if config violates the condition, else True.
        """

        self.name = name
        self.check = check
    
    def __call__(self, config):
        if not self.check(config):
            raise ValueError("Pipeline configuration condition violated: %s" % self.name)
    
    @staticmethod
    def get_larger_condition(name, config_option_name1, config_option_name2):
        def check(config):
            return config[config_option_name1] > config[config_option_name2]
        return ConfigCondition(name, check)
    
    @staticmethod
    def get_larger_equals_condition(name, config_option_name1, config_option_name2):
        def check(config):
            return config[config_option_name1] >= config[config_option_name2]
        return ConfigCondition(name, check)
