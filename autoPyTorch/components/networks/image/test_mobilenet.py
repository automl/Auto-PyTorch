import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from mobilenet import MobileNet

test_config = {'nr_main_blocks':4,
               'initial_filters':32,
               'nr_sub_blocks_1':1,
               'nr_sub_blocks_2':2,
               'nr_sub_blocks_3':2,
               'nr_sub_blocks_4':3,
               'op_type_1':'dwise_sep_conv',
               'op_type_2':'inverted_residual',
               'op_type_3':'inverted_residual',
               'op_type_4':'inverted_residual',
               'kernel_size_1':3,
               'kernel_size_2':3,
               'kernel_size_3':5,
               'kernel_size_4':3,
               'stride_1':1,
               'stride_2':2,
               'stride_3':2,
               'stride_4':2,
               'out_filters_1':16,
               'out_filters_2':24,
               'out_filters_3':40,
               'out_filters_4':80,
               'skip_con_1':True,
               'skip_con_2':True,
               'skip_con_3':True,
               'skip_con_4':True,
               'se_ratio_1':0.25,
               'se_ratio_2':0.25,
               'se_ratio_3':0.25,
               'se_ratio_4':0.25,
               'dropout_rate':0.2,
               'drop_connect_rate':0.2}


def get_layers(network, layer_list):
    for layer in network.children():
        if type(layer) == nn.Sequential:
            get_layers(layer, layer_list)
        else:
            layer_list.append(layer)


class ParallelMobileNet(MobileNet):

    def __init__(self, parent_net, n_gpus=torch.cuda.device_count()):
        super(ParallelMobileNet, self).__init__(parent_net.config, parent_net.n_feats, parent_net.n_classes, parent_net.final_activation)

        self.model = parent_net.model
        self.n_gpus = n_gpus
        self.layer_list = []
        
        get_layers(self.model, self.layer_list)
        self.parallelize_model()

    def auto_forward_layer(self, x, layer_ind_start, layer_ind_end=None):
        if layer_ind_end is None:
            layer_ind_end = len(self.layer_list)
        if layer_ind_start > -1:
            increment = layer_ind_start
        else:
            increment = len(self.layer_list) + layer_ind_start
        print(layer_ind_start, layer_ind_end, increment)
        for ind,layer in enumerate(self.layer_list[layer_ind_start:layer_ind_end]):
            x = layer(x.to("cuda:"+str(int((ind+increment) // self.chunk_size))))
        return x

    def forward_features(self, x, pool=True):
        x = self.model.conv_stem(x)
        x = self.model.bn1(x)
        x = self.model.act_fn(x, inplace=True)
        x = self.auto_forward_layer(x, 2, -4)
        if self.model.efficient_head:
            # efficient head, currently only mobilenet-v3 performs pool before last 1x1 conv
            x = self.model.global_pool(x)  # always need to pool here regardless of flag
            x = self.model.conv_head(x)
            # no BN
            x = self.model.act_fn(x, inplace=True)
            if pool:
                # expect flattened output if pool is true, otherwise keep dim
                x = x.view(x.size(0), -1)
        else:
            if self.model.conv_head is not None:
                x = self.model.conv_head(x)
                x = self.model.bn2(x)
            x = self.model.act_fn(x, inplace=True)
            if pool:
                x = self.model.global_pool(x)
                x = x.view(x.size(0), -1)
        return x

    def forward(self, x):
        x = self.forward_features(x)
        if self.model.drop_rate > 0.:
            x = F.dropout(x, p=self.drop_rate, training=self.training)

        x = self.model.classifier(x)

        if not self.training and self.final_activation is not None:
            x = self.final_activation(x)
        return x


    def parallelize_model(self):
        # TODO
        # This should just slice the mobilenet blocks, not the conv head and so on. conv stem... should be placed on first gpu,  conv head ... on last gpu
        n_layers = len(self.layer_list)

        self.chunk_size = np.ceil(n_layers/self.n_gpus)

        for ind, layer in enumerate(self.layer_list):
            print("cuda:" + str(int(ind // self.chunk_size)))
            layer.to("cuda:" + str(int(ind // self.chunk_size)))


if __name__=="__main__":
    from mobilenet import MobileNet
    import torch
    import torch.nn as nn

    test_mobilenet = MobileNet(config=test_config,
                               in_features=(3,64,64),
                               out_features=11,
                               final_activation=None)

    test_mobilenet = ParallelMobileNet(test_mobilenet)

    #test_data = torch.zeros((10,3,64,64))
    test_data = torch.zeros((10,3,64,64)).to("cuda:0")

    print("MobileNet default cfg: ", test_mobilenet.model.default_cfg)
    print("Test predicitons", test_mobilenet.forward(test_data))
    #print("Test predicitons", test_mobilenet.model.forward(test_data))

    n_modules = 0
    module_names = []
    for module in test_mobilenet.model.modules():
        n_modules += 1
        module_names.append(module)

    n_children = 0
    children = []
    for child in test_mobilenet.model.children():
        n_children += 1
        children.append(child)

    all_layers = []
    get_layers(test_mobilenet.model, all_layers)
    n_layers = len(all_layers)
    
    #print("Modules: ", module_names)
    #print("Number of modules: ", n_modules)

    #print("Children: ", children)
    #print("Number of children: ", n_children)

    #print("Layers: ", all_layers)
    #print("Layer 1 :", all_layers[0])
    print("Number of layers: ", n_layers)

    print("type(test_mobilenet.model.conv_stem): ", type(test_mobilenet.model.conv_stem))
    print("type(test_mobilenet.model.bn1): ", type(test_mobilenet.model.bn1))
    print("type(test_mobilenet.model.act_fn): ", type(test_mobilenet.model.act_fn))
    print("type(test_mobilenet.model.blocks): ", type(test_mobilenet.model.blocks))
    print("type(test_mobilenet.model.conv_head): ", type(test_mobilenet.model.conv_head))
    print("type(test_mobilenet.model.bn2): ", type(test_mobilenet.model.bn2))
    print("type(test_mobilenet.model.global_pool): ", type(test_mobilenet.model.global_pool))
    print("type(test_mobilenet.model.classifier): ", type(test_mobilenet.model.classifier))

    print("Model blocks:", test_mobilenet.model.blocks)
