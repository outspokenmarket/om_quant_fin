# OM Quant Fin

OM Quant Fin is a modern Python library for quantitative trading analysis. Our mission is to make your quant life easier and more accurate.

## Project Structure

```python
om_quant_fin/               # Root directory of the project
├── om_quant_fin/           # Python package containing the library's code
│   ├── __init__.py         # Marks the directory as a package and can contain package-level code or imports
│   └── om_quant_fin.py     # Contains library's functions
├── setup.py                # Provides package metadata and dependencies for packaging and distribution
├── .gitignore              # Lists files and folders that should not be tracked by Git
└── README.md               # Markdown file with a description of the project, usage instructions, and other information
```

## What's new in version 1.0.0

- OM Quant Fin is now beta!
- Pain Index - by Dr. Thomas Becker and Aaron Moore of Zephyr Associates
- Bootstrapping methodology for model robustness evaluation
- Plot funcions for the Pain Index and Bootstrapping
- QCutTransformer is a custom transformer class that extends the BaseEstimator and TransformerMixin classes from sklearn.
    - This is a fit and transform for the qcut method!
    - This transformer performs quantile-based discretization on the input data.
    - The constructor function initializes the transformer with the number of quantiles (default is 10)
    and optional labels for the bins.
- Removal of the logistic regression model; the package will focus on attributes, indicators and relevant metrics
    - for modelling please use the very well known Python packages

## Features

- Download stock data from Yahoo Finance
- Calculate rolling Z-scores
- Calculate rolling ratio of adjusted close and its mean
- Evaluates the model with AUC and Gini for classification models and respective plots
- QCut fit method for proper binning in unseen data
- Robustness test with bootstrapping
- Calculate the Pain Index that is a measure of portfolio risk that takes into account both the depth and duration of drawdowns

## Installation

Install the library using pip:

```python
pip install om_quant_fin
```

## Some sample usage

```python
import om_quant_fin as mql

#Download stock data:
  data = mql.download_data("AAPL", "2020-01-01", "2022-12-31")

#Calculate rolling Z-score:
  z_score = mql.rolling_z_score(data["Adj Close"], window = 20)

#Calculate rolling ratio:
  ratio = mql.rolling_ratio(data["Adj Close"], window = 20)

#Calculate returns:
  returns = mql.calculate_returns(data["Adj Close"], period = 1)

#Pain index
  window_size = 52 #rolling window for the pain index
  data["Pain_index"] = data["Adj Close"].rolling(window_size).apply(mql.pain_index, raw = True)
  mql.plot_pain_index(ticker1, data.index, data["Adj Close"], data["Pain_index"])

#Bootstrapping
  bs = mql.model_bs(model, x_train, y_train, x_test, y_test, n_iterations = 1000, range_bs = 0.1)
  mql.plot_bs(bs["accuracy_train"])
  mql.plot_bs(bs["accuracy_test"])

#qcut fit and transformer
labels = ["bin1", "bin2", "bin3", "bin4", "bin5", "bin6", "bin7", "bin8", "bin9", "bin10"
          ,"bin11", "bin12", "bin13", "bin14", "bin15", "bin16", "bin17", "bin18", "bin19", "bin20"]
qcut_transformer = mql.QCutTransformer(q = 20, labels = labels)
qcut_transformer.fit(data["column"])
qcut_transformer.transform(data["column"])
```
## License

This project is licensed under the MIT License.
