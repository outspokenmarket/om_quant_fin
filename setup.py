from setuptools import setup, find_packages

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
    description="A simple quantitative trading library for the OMNP Class",
    classifiers=[
        "Development Status :: 3 - Alpha",
        "License :: OSI Approved :: MIT License",
        "Programming Language :: Python :: 3",
        "Programming Language :: Python :: 3.6",
        "Programming Language :: Python :: 3.7",
        "Programming Language :: Python :: 3.8",
    ],
)

