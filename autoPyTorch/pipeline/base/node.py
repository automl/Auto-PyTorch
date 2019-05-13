__author__ = "Max Dippel, Michael Burkart and Matthias Urban"
__version__ = "0.0.1"
__license__ = "BSD"


import gc
import inspect


class Node():
    def __init__(self):
        self.child_node = None
        self.fit_output = None
        self.predict_output = None

    def fit(self, **kwargs):
        """Fit pipeline node.
        Each node computes its fit function in linear order by fit_traverse()..
        All args have to be specified in a parent node fit output.

        Returns:
            dict -- output values that will be passed to child nodes, if required
        """
        return dict()

    def predict(self, **kwargs):
        """Predict pipeline node.
        Each node computes its predict function in linear order by predict_traverse().
        All args have to be specified in a parent node predict output or in the fit output of this node

        Returns:
            dict -- output values that will be passed to child nodes, if required
        """
        return dict()
    
    def get_fit_argspec(self):
        """Get the necessary keywords of the fit method for this node
        
        Returns:
            tuple -- The keywords and their defaults
        """
        possible_keywords, _, _, defaults, _, _, _ = inspect.getfullargspec(self.fit)
        possible_keywords = [k for k in possible_keywords if k != 'self']
        return possible_keywords, defaults

    def get_predict_argspec(self):
        """Get the necessary keywords of the predict method for this node
        
        Returns:
            tuple -- The keywords and their defaults
        """
        possible_keywords, _, _, defaults, _, _, _ = inspect.getfullargspec(self.predict)
        possible_keywords = [k for k in possible_keywords if k != 'self']
        return possible_keywords, defaults

    def clean_fit_data(self):
        node = self
        
        # clear outputs
        while (node is not None):
            node.fit_output = None
            node.predict_output = None
            node = node.child_node

    def fit_traverse(self, **kwargs):
        """
        Calls fit function of child nodes.
        The fit function can have different keyword arguments.
        All keywords have to be either defined in kwargs or in an fit output of a parent node.

        The fit method of each node specifies a list of keyword arguments.
        The fit method of each node returns a dictionary of values for keywords of follwing nodes.

        This method collects the results of each fit method call and calls the fit methods with the collected values.
        """

        self.clean_fit_data()
        gc.collect()

        base = Node()
        base.fit_output = kwargs

        # map all collected kwargs to node whose result the kwarg was
        available_kwargs = {key: base for key in kwargs.keys()}

        node = self
        prev_node = base

        while (node is not None):
            prev_node = node
            # get necessary kwargs of current node
            possible_keywords, defaults = node.get_fit_argspec()

            last_required_keyword_index = len(possible_keywords) - len(defaults or [])
            required_kwargs = dict()

            # get the values to the necessary keywords if available. Use default if not.
            for index, keyword in enumerate(possible_keywords):
                if (keyword in available_kwargs):
                    required_kwargs[keyword] = available_kwargs[keyword].fit_output[keyword]

                elif index >= last_required_keyword_index:
                    required_kwargs[keyword] = defaults[index - last_required_keyword_index]

                else:  # Neither default specified nor keyword available
                    print ("Available keywords:", sorted(available_kwargs.keys()))
                    raise ValueError('Node ' + str(type(node)) + ' requires keyword ' + str(keyword) + ' which is not available.')
            
            # call fit method
            node.fit_output = node.fit(**required_kwargs)
            if (not isinstance(node.fit_output, dict)):
                raise ValueError('Node ' + str(type(node)) + ' does not return a dictionary.')

            # collect resulting keyword-value pairs
            for keyword in node.fit_output.keys():
                if keyword in available_kwargs:
                    # delete old values
                    if (keyword not in available_kwargs[keyword].get_predict_argspec()[0]):
                        del available_kwargs[keyword].fit_output[keyword]
                available_kwargs[keyword] = node
            node = node.child_node

        gc.collect()

        return prev_node.fit_output

    def predict_traverse(self, **kwargs):
        """Calls predict function of child nodes.
        The predict function can have different keyword arguments.
        All keywords have to be either defined in kwargs, in a predict output of a parent node or in the nodes own fit output.

        The predict method of each node specifies a list of keyword arguments.
        The predict method of each node returns a dictionary of values for keywords of follwing nodes.

        This method collects the results of each predict method call and calls the predict methods with the collected values.
        For each node, the results of the fit call can also be passed to the predict method
        
        """
        
        base = Node()
        base.predict_output = kwargs

        # map all collected kwargs to node whose whose result the kwarg was
        available_kwargs = {key: base for key in kwargs.keys()}

        node = self

        # clear outputs
        while (node is not None):
            node.predict_output = None
            node = node.child_node

        gc.collect()

        node = self
        prev_node = base

        while (node is not None):
            prev_node = node
            # get necessary kwargs of current node
            possible_keywords, defaults = node.get_predict_argspec()

            last_required_keyword_index = len(possible_keywords) - len(defaults or [])
            required_kwargs = dict()

            # get the values to the necessary keywords if available. Use fit result or default if not.
            for index, keyword in enumerate(possible_keywords):
                if (keyword in available_kwargs):
                    if (available_kwargs[keyword].predict_output is None):
                        print(str(type(available_kwargs[keyword])))
                    required_kwargs[keyword] = available_kwargs[keyword].predict_output[keyword]
                
                elif (node.fit_output is not None and keyword in node.fit_output):
                    required_kwargs[keyword] = node.fit_output[keyword]

                elif index >= last_required_keyword_index:
                    required_kwargs[keyword] = defaults[index - last_required_keyword_index]

                else:  # Neither default specified nor keyword available nor available in fit result of the node
                    raise ValueError('Node ' + str(type(node)) + ' requires keyword ' + keyword + ' which is not available.')
            
            node.predict_output = node.predict(**required_kwargs)
            if (not isinstance(node.predict_output, dict)):
                raise ValueError('Node ' + str(type(node)) + ' does not return a dictionary.')

            # collect keyword arguments
            for keyword in node.predict_output.keys():
                if keyword in available_kwargs:
                    # delete old values
                    if (available_kwargs[keyword].predict_output[keyword] is not None):
                        del available_kwargs[keyword].predict_output[keyword]
                available_kwargs[keyword] = node
            node = node.child_node
            
        gc.collect()

        return prev_node.predict_output









