from setuptools import setup, find_packages
from setuptools import setup
from os import path

# Read the contents of the README.md file
this_directory = path.abspath(path.dirname(__file__))
with open(path.join(this_directory, 'README.md'), encoding='utf-8') as f:
    long_description = f.read()


setup(
    name = "om_quant",
    version = "0.1.0",
    packages = find_packages(),
    install_requires = [
        "yfinance",
        "pandas",
        "scikit-learn",
        "numpy"
    ],
    author = "Outspoken Market",
    author_email = "info@outspokenmarket.com",
    description = "A simple quantitative trading library for the OMNP Class",
    long_description = long_description,
    long_description_content_type = 'text/markdown',
    classifiers = [
        "Development Status :: 3 - Alpha",
        "License :: OSI Approved :: MIT License",
        "Programming Language :: Python :: 3",
        "Programming Language :: Python :: 3.6",
        "Programming Language :: Python :: 3.7",
        "Programming Language :: Python :: 3.8",
    ],
)
