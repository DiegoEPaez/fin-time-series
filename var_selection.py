import numpy as np
import random
import torch

from model.symb_model import SymbolModel
from data.handle_data import get_data
from pred import errors_dist, split_sets, create_data


def error(params, df_symbol, path):
    symb = params['symbol']
    interval = params['pred_interval']

    X, y, Xs, n_steps, scaler_x, scaler_y, col2stat, r_cut = create_data(df_symbol, params['symbol'], params['n_steps'],
                                                                         interval, path)
    splitted = split_sets(X, y)
    X_train, y_train, X_test, y_test = [torch.tensor(x).float() for x in splitted]

    symb_model = SymbolModel(*X_train.shape[1:])

    n_epochs = params['n_epochs']
    cv_loops = params['cv_loops']
    cv_splits = params['cv_splits']

    errors = errors_dist(X, y, symb_model, cv_loops, cv_splits, n_epochs, symb, interval, train_all=True)
    return np.sqrt((errors ** 2.0).mean())


def main():
    symbol = 'USDMXN'

    df_symbol = get_data(symbol)
    df_symbol.to_csv('last_run.csv')

    tries_per_interval = 70
    intervals = [25, 50, 75, 125, 150]
    for i in intervals:
        parameters = {
            'symbol': symbol,
            'n_steps': 20,
            'pred_interval': i,
            'n_epochs': 25,
            'cv_loops': 1,
            'cv_splits': 3,
        }

        print(f"Predicting for interval {i} of {symbol}")
        path = f'trained/{symbol}/{i}'
        best_rmse = float('inf')
        vars = []
        cols = list(set(df_symbol.columns) - {symbol})

        for k in range(tries_per_interval):
            # TEST
            selected_cols = [symbol]
            n_cols = np.random.randint(1, len(cols))
            others = random.sample(cols, n_cols)
            selected_cols.extend(others)
            df_symbol_red = df_symbol[selected_cols]

            rmse = error(parameters, df_symbol_red, path)
            print(f"Try {k} of {tries_per_interval} for interval {i}, RMSE of {selected_cols}: {rmse}")
            if rmse < best_rmse:
                best_rmse = rmse
                vars = selected_cols
        print(f"Best for interval {i}: ", best_rmse, vars)


if __name__ == "__main__":
    main()