import os
import setuptools

with open("README.md", "r") as f:
    long_description = f.read()

requirements = []
with open('requirements.txt', 'r') as f:
    for line in f:
        requirements.append(line.strip())

optional_requirements = []
with open('optional-requirements.txt', 'r') as f:
    for line in f:
        optional_requirements.append(line.strip())

add_presets = []
for dirname, subdirs, files in os.walk(os.path.join('autoPyTorch', 'core', 'presets')):
    add_presets.extend([os.path.join(dirname, f) for f in files])

# noinspection PyInterpreter
setuptools.setup(
    name="autoPyTorch",
    version="0.0.2",
    author="AutoML Freiburg",
    author_email="zimmerl@informatik.uni-freiburg.de",
    description=("Auto-PyTorch searches neural architectures using BO-HB"),
    long_description=long_description,
    url="https://github.com/automl/Auto-PyTorch",
    long_description_content_type="text/markdown",
    license="3-clause BSD",
    keywords="machine learning algorithm configuration hyperparameter "
             "optimization tuning neural architecture deep learning",
    packages=setuptools.find_packages(),
    classifiers=[
        "Development Status :: 3 - Alpha",
        "Topic :: Utilities",
        "Topic :: Scientific/Engineering",
        "Topic :: Scientific/Engineering :: Artificial Intelligence",
        "Programming Language :: Python :: 3",
        "License :: OSI Approved :: BSD License",
    ],
	python_requires='>=3',
    platforms=['Linux'],
    install_requires=requirements,
    data_files=[('', add_presets)],
    include_package_data=True,
#    extras_require=optional_requirements
)
