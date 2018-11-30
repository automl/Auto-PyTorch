
from autoPyTorch.pipeline.base.pipeline_node import PipelineNode
from autoPyTorch.pipeline.base.pipeline import Pipeline

class SubPipelineNode(PipelineNode):
    def __init__(self, sub_pipeline_nodes):
        super(SubPipelineNode, self).__init__()
        
        self.sub_pipeline = Pipeline(sub_pipeline_nodes)

    def set_pipeline(self, pipeline):
        super(SubPipelineNode, self).set_pipeline(pipeline)
        self.sub_pipeline.set_parent_pipeline(pipeline)

    def fit(self, **kwargs):
        return self.sub_pipeline.fit_pipeline(**kwargs)

    def predict(self, **kwargs):
        return self.sub_pipeline.predict_pipeline(**kwargs)

