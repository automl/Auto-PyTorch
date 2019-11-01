
from autoPyTorch.pipeline.base.pipeline_node import PipelineNode
from autoPyTorch.pipeline.base.pipeline import Pipeline

class SubPipelineNode(PipelineNode):
    """A Pipeline node that contains a sub-pipeline"""
    def __init__(self, sub_pipeline_nodes):
        """Construct the node and the sub pipeline
        
        Arguments:
            sub_pipeline_nodes {list} -- A list of nodes of the sub-pipeline
        """
        super(SubPipelineNode, self).__init__()
        
        self.sub_pipeline = Pipeline(sub_pipeline_nodes)

    def set_pipeline(self, pipeline):
        super(SubPipelineNode, self).set_pipeline(pipeline)
        self.sub_pipeline.set_parent_pipeline(pipeline)

    def fit(self, **kwargs):
        return self.sub_pipeline.fit_pipeline(**kwargs)

    def predict(self, **kwargs):
        return self.sub_pipeline.predict_pipeline(**kwargs)
    
    def clone(self):
        sub_pipeline = self.sub_pipeline.clone()
        new_node = super().clone(skip=("pipeline", "fit_output", "predict_output", "child_node", "sub_pipeline"))
        new_node.sub_pipeline = sub_pipeline
        return new_node

