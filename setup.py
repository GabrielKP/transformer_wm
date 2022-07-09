"""Sentential functional short term memory in language models.
"""

import pathlib

# Always prefer setuptools over distutils
from setuptools import find_packages, setup

here = pathlib.Path(__file__).parent.resolve()

# Get the long description from the README file
long_description = (here / "README.md").read_text(encoding="utf-8")

# Arguments marked as "Required" below must be included for upload to PyPI.
# Fields marked as "Optional" may be commented out.

setup(
    name="transformer_wm",
    version="0.0.1",
    description="Analyze sentential functional short term memory in language models.",
    long_description=long_description,
    long_description_content_type="text/markdown",
    url="https://github.com/GabrielKP/transformer_wm",
    author="Gabriel Kressin Palacios",
    author_email="gabriel.kressin@fu-berlin.de",
    license="MIT Licence",
    # For a list of valid classifiers, see https://pypi.org/classifiers/
    classifiers=[
        "Development Status :: 3 - Alpha",
        "Intended Audience :: Science/Research",
        # "License :: OSI Approved :: MIT License",
        "Programming Language :: Python :: 3",
        "Programming Language :: Python :: 3.7",
        "Programming Language :: Python :: 3 :: Only",
    ],
    keywords="lm, memory, senteces, working memory, wm, transformers,",
    packages=find_packages(),
    python_requires=">=3.7",
    install_requires=[
        "dill",
        "nltk",
        "numpy",
        "pandas",
        "tqdm",
        "transformers",
        "matplotlib",
        "seaborn>=0.11.2",
        "wordfreq",
        "ptitprince",
        "seaborn",
        "ptitprince",
    ],
    extras_require={
        "jupyter": ["jupyterlab", "jupytext"],
    },
    project_urls={
        "Source": "https://github.com/KristijanArmeni",
    },
)
