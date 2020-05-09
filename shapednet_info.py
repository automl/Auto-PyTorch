from IPython import embed

from autoPyTorch.components.networks.feature.shapedmlpnet import *
from autoPyTorch.components.networks.feature.shapedresnet import *
from autoPyTorch.components.networks.feature.embedding import NoEmbedding


resnet_config = {"activation": "relu", 
                 "blocks_per_group": 2, 
                 "max_units": 100,
                 "num_groups": 2, 
                 "resnet_shape": "funnel", 
                 "use_dropout": True, 
                 "use_shake_drop": True,
                 "use_shake_shake": True, 
                 "max_dropout": 0.1, 
                 "max_shake_drop_probability": 0.1}

mlp_config = {"max_units":100,
              "num_layers":4,
              "mlp_shape":"funnel",
              "use_dropout":True,
              "max_dropout":0.1,
              "activation":"relu"}

embedding = NoEmbedding(config={}, in_features=5, one_hot_encoder=None)

resnet = ShapedResNet(config=resnet_config, in_features=5, out_features=10, embedding=embedding, final_activation=None)
mlnet = ShapedMlpNet(config=mlp_config, in_features=5, out_features=10, embedding=embedding, final_activation=None)


print(get_shaped_neuron_counts(shape="funnel", in_feat=5, out_feat=10, max_neurons=100, layer_count=5)   )

embed()
