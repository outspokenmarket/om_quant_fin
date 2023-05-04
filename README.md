# OM Quant lib

OM Quant is a simple Python library for quantitative trading analysis. It provides functionality for downloading stock data, calculating various indicators, and running a logistic regression model with AUC and Gini metrics for model evaluation.

## Project Structure

```python
om_quant_fin/                   # Diretório raiz do projeto
├── om_quant_fin/              # Python package
│   ├── __init__.py         # Marca o diretório como um package
│   └── om_quant_fin.py         # Contém as funções da sua lib
├── setup.py                # Fornece os metadados do package bem como suas dependências
├── .gitignore              # Lista os arquivos e pastas que não precisam ser registradas no git
└── README.md               # Arquivo Markdown com a descrição do projeto, exemplos e outras informações
```

## Features

- Download stock data from Yahoo Finance
- Calculate rolling Z-scores
- Calculate rolling ratio of adjusted close and its mean
- Calculate returns
- Split data into training and test sets
- Run a logistic regression model
- Evaluates the model with AUC and Gini

## Installation

Install the library using pip:

```python
pip install om_quant_fin
```

## Usage

```python
import om_quant_fin as mql
import om_quant_fin as mql

#Download stock data:
  data = mql.download_data("AAPL", "2020-01-01", "2022-12-31")

#Calculate rolling Z-score:
  z_score = mql.rolling_z_score(data["Adj Close"], window=20)

#Calculate rolling ratio:
  ratio = mql.rolling_ratio(data["Adj Close"], window=20)

#Calculate returns:
  returns = mql.calculate_returns(data["Adj Close"])

#Split data into training and test sets:
  train, test = mql.split_data(data, "2020-01-01", "2021-12-31", "2022-01-01", "2022-12-31")

#Run a logistic regression model:
  predictions = mql.run_logistic_regression(X_train, y_train, X_test)

```
## License

This project is licensed under the MIT License.

This markdown file provides an overview of your project, its structure, features, installation, and usage instructions. It's a good starting point for users who want to learn about your library and how to use it. You can also include additional information, such as code examples, detailed explanations of the functions, and any other relevant information that you'd like to share.
