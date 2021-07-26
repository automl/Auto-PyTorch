import os
import logging
import time
from ConfigSpace.read_and_write import json
from autoPyTorch.utils.config.config_option import ConfigOption, to_bool
from autoPyTorch.pipeline.base.pipeline_node import PipelineNode
from autoPyTorch.utils.hyperparameter_search_space_update import HyperparameterSearchSpaceUpdates

class CreateDatasetOverview(PipelineNode):

    def fit(self, pipeline_config, autonet, data_manager, result_dir):
        if not pipeline_config['create_dataset_overview']:
            return dict()

        overview_path = os.path.join(result_dir, pipeline_config['overview_name'])

        if pipeline_config['task_id'] not in [-1, 1]:
            while not os.path.exists(overview_path):
                logging.getLogger('benchmark').info('Waiting for overview.pdf')
                time.sleep(2)
            logging.getLogger('benchmark').info('overview.pdf exists - continue')
            return dict()

        autonet_config = autonet.get_current_autonet_config()
       
        from autoPyTorch.pipeline.nodes.image.create_dataset_info import CreateDatasetInfo
        from autoPyTorch.pipeline.nodes.image.multiple_datasets import MultipleDatasets
        from autoPyTorch.pipeline.nodes.image.image_dataset_reader import ImageDatasetReader

        from autoPyTorch.pipeline.base.pipeline import Pipeline
        from autoPyTorch.pipeline.base.pipeline_node import PipelineNode

        class PrintNode(PipelineNode):
            def fit(self, X_train, Y_train):
                import numpy as np
                import random
                labels = np.unique(Y_train)
                paths = []
                for label in labels:
                    indices = (Y_train==label).nonzero()[0]
                    rand = random.randint(0, len(indices)-1)
                    path = X_train[indices[rand]]
                    paths.append(path)
                return {'loss': 0, 'info': {'sample_images': paths}}

        pipe = Pipeline([
            MultipleDatasets([
                ImageDatasetReader(),
                CreateDatasetInfo(),
                PrintNode()
            ])
        ])

        hyperparameter_config = pipe.get_hyperparameter_search_space().sample_configuration()

        autonet_config['use_tensorboard_logger'] = False
        autonet_config['max_budget'] = autonet_config['min_budget'] = 1

        # fig, big_axes = plt.subplots(figsize=(sample_images * scale, classes * scale), nrows=classes, ncols=1, sharey=True)
        res = pipe.fit_pipeline(
            pipeline_config=autonet_config, hyperparameter_config=hyperparameter_config, budget=1, budget_type='epochs',
            X_train=data_manager.X_train, Y_train=data_manager.Y_train, X_valid=None, Y_valid=None,
            config_id=[0,0,0], working_directory=None)
        
        infos = res['info']
        max_classes = max([len(val['sample_images']) for val in infos])
        n_datasets = len(infos)

        import matplotlib.pyplot as plt

        scale = 3
        fig, big_axes = plt.subplots(figsize=(max_classes * scale, n_datasets * scale), nrows=n_datasets, ncols=1, sharey=True)
        for i, info in enumerate(infos):
            dataset = info['dataset_path']
            n = i * max_classes + 1
            axes = big_axes[i] if n_datasets > 1 else big_axes
            axes.set_title(os.path.basename(dataset), fontweight='bold', size=40)
            axes._frameon = False
            axes.tick_params(labelcolor=(1.,1.,1., 0.0), top=False, bottom=False, left=False, right=False)
            class_paths = info['sample_images']
            n += (max_classes - len(class_paths)) // 2
            for img_path in class_paths:
                img = plt.imread(img_path)
                fig.add_subplot(n_datasets, max_classes, n)
                plt.imshow(img, interpolation='nearest')
                plt.xticks([])
                plt.yticks([])
                n += 1

        fig.set_facecolor('w')
        plt.tight_layout()
        fig.savefig(overview_path, bbox_inches='tight')

        return dict()
        

    def get_pipeline_config_options(self):
        options = [
            ConfigOption('create_dataset_overview', default=False, type=to_bool),
            ConfigOption('overview_name', default='overview.pdf', type=str)
        ]
        return options
