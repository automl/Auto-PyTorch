"""
======================
Image Classification
======================
"""
import numpy as np

import sklearn.model_selection

import torchvision.datasets

from autoPyTorch.pipeline.image_classification import ImageClassificationPipeline

# Get the training data for tabular classification
trainset = torchvision.datasets.FashionMNIST(root='../datasets/', train=True, download=True)
data = trainset.data.numpy()
data = np.expand_dims(data, axis=3)
# Create a proof of concept pipeline!
dataset_properties = dict()
pipeline = ImageClassificationPipeline(dataset_properties=dataset_properties)

# Train and test split
train_indices, val_indices = sklearn.model_selection.train_test_split(
    list(range(data.shape[0])),
    random_state=1,
    test_size=0.25,
)

# Configuration space
pipeline_cs = pipeline.get_hyperparameter_search_space()
print("Pipeline CS:\n", '_' * 40, f"\n{pipeline_cs}")
config = pipeline_cs.sample_configuration()
print("Pipeline Random Config:\n", '_' * 40, f"\n{config}")
pipeline.set_hyperparameters(config)

# Fit the pipeline
print("Fitting the pipeline...")

pipeline.fit(X=dict(X_train=data,
                    is_small_preprocess=True,
                    dataset_properties=dict(mean=np.array([np.mean(data[:, :, :, i]) for i in range(1)]),
                                            std=np.array([np.std(data[:, :, :, i]) for i in range(1)]),
                                            num_classes=10,
                                            num_features=data.shape[1] * data.shape[2],
                                            image_height=data.shape[1],
                                            image_width=data.shape[2],
                                            is_small_preprocess=True),
                    train_indices=train_indices,
                    val_indices=val_indices,
                    )
             )

# Showcase some components of the pipeline
print(pipeline)
