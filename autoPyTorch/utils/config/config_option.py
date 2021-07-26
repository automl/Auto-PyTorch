__author__ = "Max Dippel, Michael Burkart and Matthias Urban"
__version__ = "0.0.1"
__license__ = "BSD"


# Transform a config to a bool.
def to_bool(value):
    return value.lower() in ["1", "true", "yes", "y"]

class ConfigOption():
    """ Options in a config file. A config file specifies values for ConfigOptions. """

    def __init__(self, name, default=None, type=str, list=False, depends=False, required=False, choices=None, info=None):
        """
        Initialize the ConfigOption.
        
        Parameters:
            name: The name of the option.
            default: The default value.
                If it depends on the value of other options, you can provide a function,
                which maps a dictionary of the other values to the default value.
            type: The type of the option.
                Might be the string "directory", if the option asks for a directoy.
                Might be a dictionary or function, which maps strings to accepted values.
                Might be a list, if multiple transformations need to be applied.
            list: Whether the option expects a list of values.
            depends: Whether the default depends on other values.
            required: Whether this option must be set.
            choices: possible values if string or bounds for numerical - None => no restrictions
        """
            
        self.name = name
        self.default = default
        self.type = type
        self.list = list
        self.depends = depends
        self.required = required
        self.choices = choices
        self.info = info

    def __str__(self):
        return str(self.name) + " \t Default: " + str(self.default) + " \t Choices: " + str(self.choices) + " \t Type: " + str(self.type)