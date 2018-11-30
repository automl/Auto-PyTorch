import os
import setuptools

requirements = []
with open('requirements.txt', 'r') as f:
    for line in f:
        requirements.append(line.strip())

optional_requirements = []
with open('optional-requirements.txt', 'r') as f:
    for line in f:
        optional_requirements.append(line.strip())

setuptools.setup(
    name="autoPyTorch",
    version="0.0.1",
    author="AutoML Freiburg",
    author_email="urbanm@informatik.uni-freiburg.de",
    description=("Auto-PyTorch searches neural architectures using BO-HB"),
    license="3-clause BSD",
    keywords="machine learning algorithm configuration hyperparameter "
             "optimization tuning neural architecture deep learning",
    url="",
    packages=setuptools.find_packages(),
    classifiers=[
        "Development Status :: 3 - Alpha",
        "Topic :: Utilities",
        "Topic :: Scientific/Engineering",
        "Topic :: Scientific/Engineering :: Artificial Intelligence",
        "License :: OSI Approved :: 3-clause BSD",
    ],
	python_requires='>=3',
    platforms=['Linux'],
    install_requires=requirements,
#    extras_require=optional_requirements
)
