import yfinance as yf
import pandas as pd
import numpy as np
from numpy import mean, absolute
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import roc_curve, roc_auc_score, auc, accuracy_score, mean_squared_error, mean_absolute_error
from sklearn.utils import resample
from sklearn.base import BaseEstimator, TransformerMixin
import matplotlib.pyplot as plt
import plotly.graph_objects as go
from plotly.subplots import make_subplots
from statsmodels.tsa.stattools import pacf, acf
from numpy_ext import rolling_apply as rolling_apply_ext
from datetime import datetime, timedelta  
from time import sleep
import sys
import warnings
warnings.filterwarnings("ignore")


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


def calculate_returns(data, period = 1):
    """Calculate returns from price data.
    
    Args:
        data (Series): Input price data.
        period (Integer): number of periods; 1 as default
        
    Returns:
        Series: Returns.
    """
    returns = data.pct_change(period)
    return returns


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
    
    fig = go.Figure()
    fig.add_trace(go.Scatter(x = fpr, y = tpr, mode = "lines", name = f"ROC curve (area = {roc_auc:.2f})"))
    fig.add_trace(go.Scatter(x = [0, 1], y = [0, 1], mode = "lines", name = "Random Classifier"))
    
    fig.update_layout(
        title = f"{dataset_type} Set ROC"
        , xaxis_title = "False Positive Rate"
        , yaxis_title = "True Positive Rate"
        , autosize = False
        , width = 800
        , height = 600
        , legend = dict( x = 0.8, y = 0.0
                        , bgcolor = "rgba(255, 255, 255, 0)"
                        bordercolor = "rgba(255, 255, 255, 0)"
                        )
        , template = "plotly_white")
    
    fig.show()
    
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


def pain_index(prices_series, freq = 252):
    """Calculate the Pain Index for a given price series
    
    Args:
        prices_series (float): Price series
        freq (integer): frequency for the pain index estimation
        
    Returns:
        float: Pain index
    """

    # Calculate drawdowns
    peak_value = pd.Series(prices_series).expanding(min_periods = 1).max()
    drawdowns = (prices_series / peak_value) - 1
    
    # Calculate drawdown durations
    drawdown_durations = (drawdowns < 0).cumsum()
    drawdown_durations[drawdowns >= 0] = 0
    
    # Calculate the product of drawdowns and their durations
    pain_products = drawdowns * drawdown_durations

    # Calculate the average pain per period
    average_pain = pain_products.sum() / freq

    # Calculate the Pain Index
    pain_index = -average_pain

    return pain_index


def plot_pain_index(ticker, index, close, pain_index):
    """Plot the Pain Index and the price series for the given tickers
    
    Args:
        ticker (string): Ticker symbol from Yahoo finance
        index (datetime): index from the main data frame in a date time format
        close (data frame column): price series
        pain_index (data frame column): the pain index
        
    Returns:
        float: Pain index
    """

    fig = make_subplots(rows = 2, cols = 1
                        , shared_xaxes = True
                        , vertical_spacing = 0.05)

    fig.add_trace(go.Scatter(x = index, y = close
                                    , name = ticker
                                    , line = dict(color = "blue"))
                  , row = 1, col = 1)

    fig.add_trace(go.Scatter(x = index, y = pain_index
                                    , name = "Pain index"
                                    , line = dict(color = "red"))
                  , row = 2, col = 1)

    fig.update_layout(height = 800, width = 800
                      , title = "Pain Index: " + ticker
                      , font_color = "blue"
                      , title_font_color = "black"
                      , xaxis_title = "Time"
                      , yaxis_title = ticker
                      , yaxis2_title = "Pain Index"
                      , font = dict(size = 15, color = "Black")
                  )

    fig.update_layout(hovermode = "x")

    # Code to exclude empty dates from the chart
    dt_all = pd.date_range(start = index[0]
                           , end = index[-1]
                           , freq = "D")
    dt_all_py = [d.to_pydatetime() for d in dt_all]
    dt_obs_py = [d.to_pydatetime() for d in index]

    dt_breaks = [d for d in dt_all_py if d not in dt_obs_py]

    fig.update_xaxes(
        rangebreaks = [dict(values = dt_breaks)]
    )
    return fig.show()


def model_bs(model, X_train, y_train, X_test, y_test, n_iterations = 1000, range_bs = 0.1):
    """Calculates a bootstrapping simulation for model accuracy
    
    Args:
        model (model object): configured classification model
        X_train (DataFrame): training features
        y_train (DataFrame): training target
        X_test (DataFrame): testing features
        y_test (DataFrame): testing target
        n_iterations (integer): number of interations for the BS calculation; 1000 is default
        range (float): percentage of the mean for robustness validation
        
    Returns:
        DataFrame: with accuracies for y_train and y_test
    """
    n_iterations = n_iterations
    accuracy_df = pd.DataFrame()
    model_bs = model

    for i in range(n_iterations):
        sys.stdout.write("\r")
        sys.stdout.write(str(np.round(i/n_iterations*100, 2)) + " %")
        sys.stdout.flush()
        sleep(0.005)
        
        X_train_resample, y_train_resample = resample(X_train, y_train)
    
        model.fit(X_train_resample, y_train_resample)
        predictions_train = model_bs.predict(X_train)
        predictions_test = model_bs.predict(X_test)
        accuracy_train = accuracy_score(y_train, predictions_train)
        accuracy_test = accuracy_score(y_test, predictions_test)

    # Armazena os resultados no data frame
        accuracy_df.loc[i, "accuracy_train"] = accuracy_train
        accuracy_df.loc[i, "accuracy_test"] = accuracy_test
    
    print("")
    print(70*"-")
    print("Average - training set: " + str(np.round(accuracy_df["accuracy_train"].mean(), 3)))
    print("Standard Deviation - training set: " + str(np.round(accuracy_df["accuracy_train"].std(), 3)))
    print("Average - testing set: " + str(np.round(accuracy_df["accuracy_test"].mean(), 3)))
    print("Standard Deviation - testing set: " + str(np.round(accuracy_df["accuracy_test"].std(), 3)))
    print(70*"-")

    bs_trigger = range_bs
    p2_train = accuracy_df["accuracy_train"].quantile(0.02)  # 2nd percentile
    p98_train = accuracy_df["accuracy_train"].quantile(0.98)  # 98th percentile
    bs_range_train = p98_train-p2_train
    bs_mean_train = accuracy_df["accuracy_train"].mean()

    p2_test = accuracy_df["accuracy_test"].quantile(0.02)  # 2nd percentile
    p98_test = accuracy_df["accuracy_test"].quantile(0.98)  # 98th percentile
    bs_range_test = p98_test-p2_test
    bs_mean_test = accuracy_df["accuracy_test"].mean()

    if (bs_range_train < bs_mean_train*bs_trigger):
        print("Training: Model is robust")
    else:
        print("Training: Model is not robust")
    print(70*"-")
    print(70*"-")

    if (bs_range_test < bs_mean_test*bs_trigger):
        print("Test: Model is robust")
    else:
        print("Test: Model is not robust")
    print(70*"-")
    
    return accuracy_df


def plot_bs(metric_bs):
    """Plot the distribution of accuracies from the bootstrapping simulation
    
    Args:
        metric_bs (DataFrame): data frame from the model_bs function
        
    Returns:
        histogram with the distribution of the metric of choice
    """
    fig = go.Figure(data = [go.Histogram(x = metric_bs)])
    fig.update_layout(
        title_text = "Bootstrapping Results - Histogram"
        , width = 600
        , height = 600 
        )
    fig.show()


class QCutTransformer(BaseEstimator, TransformerMixin):
    """QCutTransformer is a custom transformer class that extends the BaseEstimator and TransformerMixin classes from sklearn.
    This transformer performs quantile-based discretization on the input data.
    The constructor function initializes the transformer with the number of quantiles (default is 10)
    and optional labels for the bins.
    """
    def __init__(self, q = 10, labels = None):
        """Init for the QCurTransformer class
    
        Args:
            q (integer): number of bins; default is 10
            labels (string list): list with the labels in the same lenght of q    
        """
        self.q = q  # Number of quantiles
        self.labels = labels  # Labels for the bins
        self.bins = None  # Will hold the bin edges after fit is called

    def fit(self, X, y = None):
        """Calculates the bin edges using pandas" qcut function
        The `retbins` parameter is set to True so that qcut returns the bin edges in addition to the binned data.
    
        Args:
            q (integer): number of bins
            labels (string list): list with the labels in the same lenght of q    
        """
        _, self.bins = pd.qcut(X, q = self.q, retbins = True, duplicates="drop")
        
        return self

    def transform(self, X, y = None):
        """The transform function bins the input data using the bin edges calculated in fit.
        If labels were provided, they are used to label the bins; otherwise, the bins are labeled with their numeric range.
        We add -inf and inf to the bin edges to ensure that all data points are included in a bin.
        If labels were provided, we add "below" and "above" labels for the -inf and inf bins.
        
        Args:
            X (DataFrame column): values to be binned
            
        Returns:
            the bins values of the input data
        """
        
        extended_bins = np.hstack([[-np.inf], self.bins, [np.inf]])
        
        
        if self.labels is not None:
            extended_labels = np.hstack([["below"], self.labels, ["above"]])
        else:
            extended_labels = None

        # We use pandas" cut function to bin the data using the extended bin edges and labels.
        return pd.cut(X, bins = extended_bins, labels = extended_labels, include_lowest = True, duplicates = "drop")

def get_acf1(x):
    """Gets the first ACF component for a given time series
        
        Args:
            x (DataFrame column): time series
            
        Returns:
            first ACF component for a given time series
            """
    return acf(x, alpha = 0.05, nlags = 5)[0][1]

def z_score(x):
    """Calculates a traditional normalization with z-score
        
        Args:
            x (DataFrame column): time series
            
        Returns:
            Normalized z-score serie
        """
    return ((x - x.mean())/np.std(x))[-1]

def next_business_day(date):
    """Given a a date, it returns the next business day
        
        Args:
            date (string): date in the string format
            
        Returns:
            Next business day
        """
    if isinstance(date, str):
        date = pd.to_datetime(date)
    business_days = pd.date_range(start=date, end=date + pd.DateOffset(weeks=1), freq = "B")

    return business_days.min()

def z_score_med(x):
    """Calculate the  z-score with a median
    
    Args:
        data (Series): Input data.
        
    Returns:
        Series: z-score calculated with the median
    """
    return ((x - x.median())/np.std(x))[-1]

def create_vars(ticker1, start_date, end_date, p = 10):
    """Creates a dataframe with a generic set of variables for volatility forecasting
    
    Args:
        ticker1 (string): ticker name from Yahoo Finance
        start_date (string): start date for the data collection
        end_date (string): end date for the data collection
        p (integer): value of the periods for the volatility forecasting; 10 is the default
        
    Returns:
        df: dataframe with all variables and target
        forecast_df: dataframe with the last p observations
        last_vol: last volatility measuread in the p periods
        today_vol: last volatility from the dataframe

    """
    # Get the data
    df1 = download_data(ticker1, start_date, end_date)
    df1["Returns"] = df1["Adj Close"].pct_change(1)
    df1["Vol"] = np.round(df1["Returns"].rolling(20).std()*np.sqrt(252)*100, 3)
    df1.dropna(axis = 0, inplace = True) 

    # Some generic vars - users are welcome to create new ones
    df1["f1"] = rolling_apply_ext(get_acf1, 20, df1["Returns"])
    df1["f2"] = rolling_apply_ext(z_score, 10, df1["Returns"])

    df1["f3"] = df1["Adj Close"]/df1["Adj Close"].rolling(5).mean()-1
    df1["f4"] = df1["Adj Close"]/df1["Adj Close"].rolling(10).mean()-1
    df1["f5"] = df1["Adj Close"]/df1["Adj Close"].rolling(20).mean()-1
    df1["f6"] = df1["Adj Close"]/df1["Adj Close"].rolling(52).mean()-1

    df1["f7"] = df1["Returns"].rolling(5).std()
    df1["f8"] = df1["Returns"].rolling(10).std()
    df1["f9"] = df1["Returns"].rolling(20).std()
    df1["f10"] = df1["Returns"].rolling(52).std()

    df1["f11"] = df1["Returns"].shift(1)
    df1["f12"] = df1["Returns"].shift(2)
    df1["f13"] = df1["Returns"].shift(3)
    df1["f14"] = df1["Returns"].shift(4)
    df1["f15"] = df1["Returns"].shift(5)

    df1["f16"] = df1["Adj Close"]/df1["MA200"]-1
    df1["f_Z16"] = rolling_apply_ext(z_score_med, 200, df1["f16"])
    df1["f_Z16"] = np.where((df1["f_Z16"] > 2), 2
                     , np.where(df1["f_Z16"] <-2, -2, df1["f_Z16"]))

    df1["f17"] = df1["Vol"].shift(1)
    df1["f18"] = df1["Vol"].shift(2)
    df1["f19"] = df1["Vol"].shift(3)
    df1["f20"] = df1["Vol"].shift(4)
    df1["f21"] = df1["Vol"].shift(5)
    df1["f22"] = df1["Vol"].shift(6)
    df1["f23"] = df1["Vol"].shift(7)
    df1["f24"] = df1["Vol"].shift(8)
    df1["f25"] = df1["Vol"].shift(9)
    df1["f26"] = df1["Vol"].shift(10)

    # Calculate drawdown
    df1["Drawdown"] = (1 + df1["Returns"]).cumprod() - 1

    # Calculate moving drawdowns for specified windows
    windows = [15, 20, 30, 52]
    for window in windows:
        df1[f"DD_{window}d"] = df1["Drawdown"].rolling(window=window).min()


    df1["f27"] = rolling_apply_ext(z_score, 200, df1["Adj Close"].rolling(20).apply(pain_index, raw = True))

    forecast_df = df1.iloc[(-1-p):].copy()
    last_vol = df1["Vol"].iloc[(-1-p):-p]
    today_vol = df1["Vol"][-1]

    # Target: volatility p days ahead
    df1["Target1"] = df1["Vol"].shift(-p)
    df1.dropna(inplace = True)

    return df1, forecast_df, last_vol, today_vol

def regression_metrics(model, x_train, x_test, y_train, y_test, stability = 0.10):
    """Generates a simple report with main regression metrics: RMSE and MAE
    
    Args:
        model (regression model object): fitted regression model
        x_train (DataFrame): dataframe with the training set
        x_test (DataFrame): dataframe with the testing set
        y_train (DataFrame): dataframe with the target variable from the training set
        y_test (DataFrame): dataframe with the target variable from the testing set
        stability (float): stability cut-off for the MAE
        
    Returns:
        Regression metrics report

    """

    # Generate predictions for both training and testing sets
    y_train_pred = model.predict(x_train)
    y_test_pred = model.predict(x_test)

    # Calculate MAE for the training set
    mae_train = np.round(mean_absolute_error(y_train, y_train_pred), 2)
    print(f"Training Set MAE: {mae_train}")

    # Calculate MAE for the testing set
    mae_test = np.round(mean_absolute_error(y_test, y_test_pred), 2)
    print(f"Testing Set MAE: {mae_test}")
    print("-"*70)
    print("")
    if np.abs(mae_test-mae_train) < mae_train*stability:
        print("Model is stable according to MAE")
    else:
        print("Model is not stable according to MAE")

def prediction_report(model, ticker1, forecast_df, last_vol, today_vol, p):
    """Generates a report with predicted value and actions to take for a given volatility model
    
    Args:
        model (model object): fitted regression model
        ticker1 (string): Yahoo Finance ticker
        forecast_df (DataFrame): dataframe to forecast the volatility
        last_vol (float): last predicted volatility
        today_vol (float): today measured volatility
        p (integer): period for the prediction
        
    Returns:
        Regression metrics report

    """
    print("-"*70)
    print("Last volatility measured before a prediction on the day " + str(forecast_df.index[0].strftime("%Y-%m-%d")))
    print("%.2f" % last_vol)
    print("Predicted volatility for " + ticker1 + " on that day was")
    print("%.2f" % (model.predict(forecast_df[vars].iloc[(-1-p):-p].values.reshape(1, -1))[0]))
    
    print("Volatility today " + str(datetime.today().strftime("%Y-%m-%d")))
    print("%.2f" % today_vol)
    print("-"*70)
    print("")
    print("-"*70)
    print("Next volatility prediction for " + ticker1 + " on the day " + str(next_business_day((datetime.today() + timedelta(days = p))).strftime("%Y-%m-%d")))
    print("%.2f" % (model.predict(forecast_df[vars].iloc[-1].values.reshape(1, -1))[0]))
    print("-"*70)

def mad_calc(data, axis = None):
    """Calculates the Mean Absolute Deviation - MAD
    
    Args:
        data (list or data frame column): series for MAD's calculation
        
    Returns:
        Mean Absolute Deviation
    """
    return mean(absolute(data - mean(data, axis)), axis)

def ifat(returns, p = 67):
    """Calculates the iFat - Fata Tail Index for indentification of fat tails
    
    Args:
        returns (data frame column): asset returns
        
    Returns:
        The iFat and it's moving standard deviation(mstd)
    """
    p = p
    ifat = returns.rolling(p).apply(mad_calc)/returns.rolling(p).std()
    mstd = ifat.rolling(20).mean() - ifat.rolling(252*2).std()
    return ifat, mstd

#