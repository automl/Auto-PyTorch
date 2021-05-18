import setuptools

with open("README.md", "r") as f:
    long_description = f.read()

requirements = []
with open("requirements.txt", "r") as f:
    for line in f:
        requirements.append(line.strip())

# noinspection PyInterpreter
setuptools.setup(
    name="autoPyTorch",
    version="0.1.0",
    author="AutoML Freiburg",
    author_email="zimmerl@informatik.uni-freiburg.de",
    description=("Auto-PyTorch searches neural architectures using smac"),
    long_description=long_description,
    url="https://github.com/automl/Auto-PyTorch",
    long_description_content_type="text/markdown",
    license="3-clause BSD",
    keywords="machine learning algorithm configuration hyperparameter"
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
    include_package_data=True,
    extras_require={
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
        ],
        "docs": ["sphinx", "sphinx-gallery", "sphinx_bootstrap_theme", "numpydoc"],
    },
    test_suite="pytest",
    data_files=[('configs', ['autoPyTorch/configs/default_pipeline_options.json']),
                ('portfolio', ['autoPyTorch/configs/greedy_portfolio.json'])]
)
