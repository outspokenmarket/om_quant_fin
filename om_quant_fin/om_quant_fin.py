import yfinance as yf
import pandas as pd
import numpy as np
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import roc_curve, roc_auc_score, auc
import matplotlib.pyplot as plt


def download_data(ticker, start_date, end_date):
    """Download stock data from Yahoo Finance.
    
    Args:
        ticker (str): Stock symbol.
        start_date (str): Start date for data in the format "YYYY-MM-DD".
        end_date (str): End date for data in the format "YYYY-MM-DD".
        
    Returns:
        DataFrame: Stock data.
    """
    stock_data = yf.download(ticker, start = start_date, end = end_date)
    return stock_data


def rolling_z_score(data, window):
    """Calculate the rolling Z-score.
    
    Args:
        data (Series): Input data.
        window (int): Rolling window size.
        
    Returns:
        Series: Rolling Z-score.
    """
    mean = data.rolling(window = window).mean()
    std_dev = data.rolling(window = window).std()
    z_score = (data - mean) / std_dev
    return z_score


def rolling_ratio(data, window):
    """Calculate the rolling ratio of data and its mean.
    
    Args:
        data (Series): Input data.
        window (int): Rolling window size.
        
    Returns:
        Series: Rolling ratio.
    """
    mean = data.rolling(window = window).mean()
    ratio = np.round((data / mean - 1), 3)*100
    return ratio


def calculate_returns(data):
    """Calculate returns from price data.
    
    Args:
        data (Series): Input price data.
        
    Returns:
        Series: Returns.
    """
    returns = data.pct_change()
    return returns


def split_data(data, train_start, train_end, test_start, test_end):
    """Split data into training and test sets based on date ranges.
    
    Args:
        data (DataFrame): Input data.
        train_start (str): Start date for training data in the format "YYYY-MM-DD".
        train_end (str): End date for training data in the format "YYYY-MM-DD".
        test_start (str): Start date for test data in the format "YYYY-MM-DD".
        test_end (str): End date for test data in the format "YYYY-MM-DD".
        
    Returns:
        tuple: A tuple containing the training and test data (DataFrames).
    """
    train = data[train_start:train_end]
    test = data[test_start:test_end]
    return train, test


def run_logistic_regression(X_train, y_train, X_test):
    """Train a logistic regression model and make predictions.
    
    Args:
        X_train (DataFrame): Input features for training data.
        y_train (Series): Target variable for training data.
        X_test (DataFrame): Input features for test data.
        
    Returns:
        Series: Predicted values for test data.
    """
    model = LogisticRegression()
    model.fit(X_train, y_train)
    predictions = model.predict(X_test)
    return predictions

def calculate_plot_auc(y_true, y_pred, dataset_type = "Unknown"):
    """Calculate and plot the AUC for given true labels and predictions.
    
    Args:
        y_true (Series): True labels.
        y_pred (Series): Predicted probabilities.
        dataset_type (str): Name of the dataset type ("Training" or "Test").
        
    Returns:
        float: AUC value.
    """
    fpr, tpr, _ = roc_curve(y_true, y_pred)
    roc_auc = auc(fpr, tpr)
    
    plt.figure()
    plt.plot(fpr, tpr, color = "darkorange", lw = 2, label = "ROC curve (area = %0.2f)" % roc_auc)
    plt.plot([0, 1], [0, 1], color = "navy", lw = 2, linestyle = "--")
    plt.xlim([0.0, 1.0])
    plt.ylim([0.0, 1.05])
    plt.xlabel("False Positive Rate")
    plt.ylabel("True Positive Rate")
    plt.title(f"{dataset_type} Set ROC")
    plt.legend(loc = "lower right")
    plt.show()
    
    return roc_auc

def calculate_gini(auc_score):
    """Calculate the Gini coefficient based on the AUC.
    
    Args:
        auc_score (float): AUC value.
        
    Returns:
        float: Gini coefficient.
    """
    gini = 2 * auc_score - 1
    return gini
