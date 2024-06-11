""" Module to predict financial time series such as USDMXN, S and P 500"""
import os
import datetime as dttm
import json
import logging
from datetime import timedelta
from pathlib import Path

import numpy as np
import pandas as pd
from sklearn.preprocessing import StandardScaler
from sklearn.model_selection import KFold
from statsmodels.tsa.stattools import adfuller
import matplotlib.pyplot as plt
import seaborn as sns
import torch

from data.handle_data import get_data
from model.symb_model import SymbolModel

sns.set()


def create_lags(values, n_steps, pred_interval):
    """
    Create lags to predict with nn, after this process an array of size [n_examples, n_steps + 1,
    n_cols] is created. For example given array [1, 2, ... 10], if n_steps = 1 and
    pred_interval = 5 the result would be:
    [[1,6], [2,7], [3,8], [4,9], [5,10]]

    However usually we pass a 2D array having several columns which stand for the different
    variables. For example, given array [[1, 2, ..., 10], [11, 12, ... 20]] the result is:
    [[  [1, 11], [6, 16]  ], ... , [  [5, 15], [10, 20] ]]

    :param values: np.array, series values
    :param n_steps: int, number of steps that lags will have
    :param pred_interval: int, length of each of these steps
    :return: np.array, size [values.shape[0] - n_steps * pred_interval, n_steps + 1]
    """
    no_ex = values.shape[0] - n_steps * pred_interval
    if no_ex <= n_steps * pred_interval:
        logging.warning(
            "Cannot create examples with given time series, there are: %s "
            "values, and number of steps is %s and prediction interval is %s"
            , values.shape[0], n_steps, pred_interval
        )
        n_steps = (values.shape[0] // 2) // pred_interval
        no_ex = values.shape[0] - n_steps * pred_interval
        logging.warning("Number of steps updated to %s", n_steps)
    start = 0
    end = no_ex
    lags = [values[start:end]]
    for _ in range(n_steps):
        start += pred_interval
        end += pred_interval
        lags.append(values[start:end])

    # stack lags in middle axis, so that final shape is examples, steps/ lags, columns
    res = np.stack(lags, axis=1)

    return res, n_steps


def stationary(series, pred_interval):
    """
    Makes series stationary if not stationary
    :param series: pd.Series, series to make stationary
    :param pred_interval: int, skip interval
    :return: modified series, modification type (no change, difference, difference of logs)
    """
    try:
        pval0 = adfuller(series)[1]
    except Exception as e:
        print(f"Attempted to make stationary series {series.name}, but failed")
        return series, None

    if np.isnan(pval0) or pval0 < 0.1:
        return series[pred_interval:], None

    new_series_dif = series[pred_interval:].values - series[:-pred_interval].values
    pval1 = adfuller(new_series_dif)[1]
    if np.isnan(pval1) or pval1 < 0.1:
        return new_series_dif, "dif"

    try:
        new_series_log = np.log(series[pred_interval:].values) - np.log(
            series[:-pred_interval].values
        )
        pval2 = adfuller(new_series_log)[1]
    except Exception as e:
        print(f"Attempted to make log stationary series {series.name}, but failed")
        return series, None

    if pval2 > 0.1:
        print(
            f"Unable to make stationary series, p value remains greater than 0.1 for {series.name}"
        )
        min_val = np.argmin(np.array([pval0, pval1, pval2]))
        return [series, new_series_dif, new_series_log][min_val], [
            None,
            "dif",
            "log_dif",
        ][min_val]

    return new_series_log, "log_dif"


def create_data(df_symbol, symbol, n_steps, pred_interval, path):
    new_data = df_symbol[pred_interval:].copy()
    col2stat = {}
    pathf = Path(path) / "col2stat.json"

    if Path.exists(pathf):
        with open(str(pathf), "r") as f:
            col2stat = json.load(f)

        # Filter columns to use
        for col in df_symbol:
            if col not in col2stat:
                new_data[col], type_stationary = stationary(
                    df_symbol[col], pred_interval
                )
                col2stat[col] = type_stationary
            else:
                if col2stat[col] == "dif":
                    new_data[col] = (
                        df_symbol[col][pred_interval:].values
                        - df_symbol[col][:-pred_interval].values
                    )
                elif col2stat[col] == "logdif":
                    new_data[col] = np.log(
                        df_symbol[col][pred_interval:].values
                    ) - np.log(df_symbol[col][:-pred_interval].values)
    else:
        for col in df_symbol:
            # Make series stationary
            new_data[col], type_stationary = stationary(df_symbol[col], pred_interval)
            col2stat[col] = type_stationary

    if not Path(path).exists():
        os.makedirs(path)

    with open(str(pathf), "w") as f:
        json.dump(col2stat, f)

    scaler_x = StandardScaler()
    scaler_y = StandardScaler()
    scaled_data_x = scaler_x.fit_transform(new_data.drop(symbol, axis=1))
    scaled_data_y = scaler_y.fit_transform(new_data[[symbol]])

    scaled_data = np.c_[scaled_data_y, scaled_data_x]

    res, n_steps = create_lags(scaled_data, n_steps, pred_interval)

    X = res[:, :-1, :]
    y = res[:, 1:, 0:1]
    Xs = res[-1:, 1:, :]

    return X, y, Xs, n_steps, scaler_x, scaler_y, col2stat, len(res)


def split_sets(X, y, train_size=0.9):
    """
    Split sets X and y by the given
    :param X:
    :param y:
    :param Xs:
    :param train_size:
    :return:
    """
    cut = int(train_size * X.shape[0])
    X_train, y_train = X[:cut], y[:cut]
    X_test, y_test = X[cut:], y[cut:]

    return X_train, y_train, X_test, y_test


def score(preds, scaler_y, adj_value, type):
    """
    Score given dataset with given model.
    :param X_score:
    :param symb_model:
    :param scaler_y:
    :param adj_value:
    :param type:
    :return:
    """
    ypredsc = scaler_y.inverse_transform(preds)

    if type == "dif":
        ypredsc = ypredsc + adj_value
    elif type == "log_dif":
        ypredsc = np.exp(ypredsc + np.log(ypredsc))

    return ypredsc


def errors_dist(
    X,
    y,
    symb_model,
    cv_loops,
    cv_splits,
    n_epochs,
    symbol,
    pred_interval,
    train_all=False,
):
    """
    Creates a probability distribution from the data, using "pred interval" trading days.

    """
    print("Calculating errors: ")
    errors = []

    kf = KFold(n_splits=cv_splits)
    kf.get_n_splits(X, y)

    for i in range(cv_loops):
        print("Loop " + str(i + 1) + " of " + str(cv_loops))
        j = 1
        for train_index, test_index in kf.split(X):
            print(f"Split j {j}")
            X_train, X_test = (
                torch.tensor(X[train_index]).float(),
                torch.tensor(X[test_index]).float(),
            )
            y_train, y_test = (
                torch.tensor(y[train_index]).float(),
                torch.tensor(y[test_index]).float(),
            )

            symb_model.get_model(
                n_epochs,
                X_train,
                y_train,
                path=f"trained/{symbol}/{pred_interval}/cv",
                train_all=train_all,
            )

            y_pred = symb_model(X_test).detach().numpy()
            error = y_test[:, -1, 0] - y_pred[:, -1, 0]
            errors.extend(error.tolist())
            j += 1

    return np.array(errors)


def pred_dist(
    X,
    y,
    X_score,
    n_epochs,
    errors,
    symb_model,
    no_pred,
    no_samples,
    symbol,
    pred_interval,
):
    preds = np.zeros(no_pred * no_samples)
    X, y = torch.tensor(X).float(), torch.tensor(y).float()
    for i in range(no_pred):
        print("Prediction " + str(i + 1) + " of " + str(no_pred))
        symb_model.get_model(
            n_epochs, X, y, path=f"trained/{symbol}/{pred_interval}/train"
        )
        y_pred = symb_model(X_score).detach().numpy()

        pred = y_pred[0, -1, 0]
        preds[i * no_samples : (i + 1) * no_samples] = pred + np.random.choice(
            errors, size=no_samples
        )

    return preds.reshape(-1, 1)


def predict(params, df_symbol, path):
    symb = params["symbol"]
    interval = params["pred_interval"]

    X, y, Xs, n_steps, scaler_x, scaler_y, col2stat, r_cut = create_data(
        df_symbol, params["symbol"], params["n_steps"], interval, path
    )
    splitted = split_sets(X, y)
    X_train, y_train, X_test, y_test = [torch.tensor(x).float() for x in splitted]
    X_score = torch.tensor(Xs).float()

    symb_model = SymbolModel(*X_train.shape[1:])

    n_epochs = params["n_epochs"]
    cv_loops = params["cv_loops"]
    cv_splits = params["cv_splits"]
    no_preds = params["no_preds"]
    no_samples = params["no_samples"]

    errors = errors_dist(
        X, y, symb_model, cv_loops, cv_splits, n_epochs, symb, interval
    )
    print(f"RMSE of cross val: {np.sqrt((errors ** 2.0).mean())}")

    preds = pred_dist(
        X,
        y,
        X_score,
        n_epochs,
        errors,
        symb_model,
        no_preds,
        no_samples,
        symb,
        interval,
    )

    idx_score = n_steps * interval + r_cut
    preds = score(preds, scaler_y, df_symbol[symb].iloc[idx_score], col2stat[symb])

    return preds


def quantiles(values, size=99):
    """
    Calculates quantiles splitting them by the size specified. For example size=99, would
    calculate quantiles 0.01, 0.02, ... 0.99
    :param values: np.array, with samples from which to calculate quantiles
    :param size: int, number of quantiles to calculate
    :return: np.array, of size "size" with quantiles
    """
    qts = np.zeros(size)
    for i in range(1, size + 1):
        qts[i - 1] = np.quantile(values, q=i / (size + 1))
    return qts


def plot_predictions(df_symbol, symbol, predictions, intervals):
    df_symbol.index = pd.to_datetime(df_symbol.index)

    fig, ax = plt.subplots()
    df_symbol[symbol][-intervals[-1] :].plot(ax=ax, color="orange")

    sz = predictions[0].shape[0]
    preds = np.zeros((sz, len(intervals) + 1))
    preds[:, 0] = df_symbol[symbol].iloc[-1]
    for i in range(1, len(intervals) + 1):
        preds[:, i] = predictions[i - 1]

    intervals_ext = [0]
    intervals_ext.extend(intervals)
    dates = [
        df_symbol.index[-1] + timedelta(days=int(interval * 7.0 / 5.0))
        for interval in intervals_ext
    ]
    hsz = sz // 2
    for i in range(sz - 1):
        alpha = (hsz - abs(hsz - i) ** 0.9) / hsz
        ax.fill_between(
            dates, preds[i], preds[i + 1], color="b", alpha=alpha, linewidth=0.0
        )

    plt.show()


def main():
    symbol = "SPX"

    df_symbol = get_data(symbol)
    df_symbol.to_csv("last_run.csv")

    # filter provisionally before 14th of feb
    df_symbol.index = pd.to_datetime(df_symbol.index)
    # df_symbol = df_symbol[df_symbol.index < dttm(2023, 2, 14)]

    # FROM MOST RECENT RUN WHERE RMSE was 0.47 with only ta vs 0.48 with all vars @25 days
    # seems only TA has any predictive power in the short term. TA should be used when
    # predicting 50 days or less, afterwards it should be dumped

    predictions = []
    intervals = [
        30, 40, 50, 60, 70, 80,
        90,
        100,
        110,
        120,
        130,
        140,
        150,
        200,
        250,
        300,
        400,
    ]
    for i in intervals:
        parameters = {
            "symbol": symbol,
            "n_steps": 20,
            "pred_interval": i,
            "n_epochs": 25,
            "cv_loops": 3,
            "cv_splits": 5,
            "no_preds": 10,
            "no_samples": 1000,
        }

        print(f"Predicting for interval {i} of {symbol}")
        path = f"trained/{symbol}/{i}"

        # I patched the code to work with tech indicators only if interval is 45 or less;
        # needs to be refactored
        if i > 45:
            # if interval is longer drop technical indicators
            try:
                tech_start = list(df_symbol.columns).index(f"{symbol}_LOW")
                tech_vars = df_symbol.columns[tech_start:]
                df_notech = df_symbol.drop(tech_vars, axis=1)
                preds = predict(parameters, df_notech, path).flatten()
            except ValueError as e:
                print(e)
                preds = predict(parameters, df_symbol, path).flatten()
        else:
            preds = predict(parameters, df_symbol, path).flatten()

        now = dttm.datetime.now().strftime("%Y%m%d")
        np.savetxt(
            "outputs/preds_" + symbol + "_" + str(i) + "_" + str(now) + ".csv",
            preds,
            delimiter=",",
        )
        predictions.append(quantiles(preds))

    plot_predictions(df_symbol, symbol, predictions, intervals)

    # TOOD
    # Try conv1D + RNN arch
    # Try self attention decoder
    # Try adding other series EURUSD, ...
    # Move this to AWS Batch / create docker container


if __name__ == "__main__":
    main()
