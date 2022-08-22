import setuptools
import sys


if sys.version_info < (3, 7):
    raise ValueError(
        'Auto-Pytorch requires Python 3.7 or higher, but found version {}.{}.{}'.format(
            sys.version_info.major, sys.version_info.minor, sys.version_info.micro
        )
    )

with open("README.md", "r") as f:
    long_description = f.read()

requirements = []
with open("requirements.txt", "r") as f:
    for line in f:
        requirements.append(line.strip())


# noinspection PyInterpreter
setuptools.setup(
    name="autoPyTorch",
    version="0.2.1",
    author="AutoML Freiburg",
    author_email="eddiebergmanhs@gmail.com",
    description=("Auto-PyTorch searches neural architectures using smac"),
    long_description=long_description,
    url="https://github.com/automl/Auto-PyTorch",
    long_description_content_type="text/markdown",
    license="3-clause BSD",
    keywords="machine learning algorithm configuration hyperparameter"
             "optimization tuning neural architecture deep learning",
    packages=setuptools.find_packages(),
    package_data={"autoPyTorch": ['py.typed']},
    classifiers=[
        "Development Status :: 3 - Alpha",
        "Topic :: Utilities",
        "Topic :: Scientific/Engineering",
        "Topic :: Scientific/Engineering :: Artificial Intelligence",
        'Programming Language :: Python :: 3.7',
        'Programming Language :: Python :: 3.8',
        'Programming Language :: Python :: 3.9',
        "License :: OSI Approved :: BSD License",
    ],
    python_requires='>=3.7',
    platforms=['Linux'],
    install_requires=requirements,
    include_package_data=True,
    extras_require={
        "forecasting": [
            "gluonts>=0.10.0",
            "sktime",
            "pytorch-forecasting",
        ],
        "test": [
            "matplotlib",
            "pytest",
            "pytest-xdist",
            "pytest-timeout",
            "flaky",
            "pyarrow",
            "pre-commit",
            "pytest-cov",
            'pytest-forked',
            'pytest-subtests',
            "codecov",
            "pep8",
            "mypy",
            "openml",
            "emcee",
            "scikit-optimize",
            "pyDOE",
        ],
        "examples": [
            "matplotlib",
            "jupyter",
            "notebook",
            "seaborn",
            "openml"
        ],
        "docs": ["sphinx", "sphinx-gallery", "sphinx_bootstrap_theme", "numpydoc"],
    },
    test_suite="pytest",
    data_files=[('configs', ['autoPyTorch/configs/default_pipeline_options.json']),
                ('portfolio', ['autoPyTorch/configs/greedy_portfolio.json']),
                ('forecasting_init', ['autoPyTorch/configs/forecasting_init_cfgs.json'])],
    dependency_links=['https://github.com/automl/automl_common.git/tarball/autoPyTorch#egg=package-0.0.1']
)
