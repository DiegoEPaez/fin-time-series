import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
import torch.utils.data as data
import os
from datetime import datetime as dttm
from pathlib import Path


class SymbolModel(nn.Module):
    def __init__(self, n_steps, no_features, dropout_rate=0.7, hidden=100):
        super().__init__()
        self.hidden_size = hidden
        self.n_steps = n_steps

        self.lstm = nn.LSTM(input_size=no_features, hidden_size=hidden, num_layers=3, dropout=dropout_rate, batch_first=True)
        self.dropout = nn.Dropout(dropout_rate)
        self.linear = nn.Linear(hidden, 1)

    def forward(self, x):
        x, _ = self.lstm(x)
        x = self.dropout(x)
        x = self.linear(x.reshape(-1, self.hidden_size))
        x = x.reshape(-1, self.n_steps, 1)
        return x

    def eval_performance(self, epoch, loss_fn, X_train, y_train, X_test, y_test):
        self.eval()
        with torch.no_grad():
            y_pred = self(X_train)
            train_rmse = np.sqrt(loss_fn(y_pred, y_train))
            y_pred = self(X_test)
            test_rmse = np.sqrt(loss_fn(y_pred, y_test))

        print(f"Epoch {epoch}: train RMSE {train_rmse:.4f}, test RMSE {test_rmse:.4f}")

    def train_model(self, n_epochs, X_train, y_train, X_test=None, y_test=None, verbose=1):
        optimizer = optim.Adam(self.parameters(), weight_decay=1e-4)
        loss_fn = nn.MSELoss()
        loader = data.DataLoader(data.TensorDataset(X_train, y_train), shuffle=True, batch_size=8)

        for epoch in range(n_epochs):
            if epoch == 0:
                if X_test is not None and verbose >= 2:
                    self.eval_performance(epoch, loss_fn, X_train, y_train, X_test, y_test)
            self.train()
            for X_batch, y_batch in loader:
                y_pred = self(X_batch)
                loss = loss_fn(y_pred, y_batch)
                optimizer.zero_grad()
                loss.backward()
                optimizer.step()

            if X_test is not None and verbose >= 2:
                self.eval_performance(epoch + 1, loss_fn, X_train, y_train, X_test, y_test)
            elif verbose == 1:
                print(f"{float(epoch) / n_epochs * 100.0:4.0f}%", end="\n")

    def get_model(self, n_epochs, X_train, y_train, path, train_all=False):
        if not os.path.exists(path):
            os.makedirs(path)

        # Get all saved models which will be loaded according to how old they are (the older the less likely to be chosen)
        models = os.listdir(path)
        dates = [dttm.strptime(m.split("_")[0], "%Y%m%d") for m in models]
        probs = [(dttm.now() - d).days for d in dates]

        # Add a last entry to train a new model
        probs.append(0)
        probs = np.array(probs)
        if probs.shape[0] > 1:
            probs[:-1] = probs[:-1] + max(probs[:-1]) + int((probs.shape[0]) ** (1 / 3.))

        if probs.sum() == 0:
            probs = probs + 1
        else:
            probs = (max(probs) + 1) - probs

        idx = np.random.choice(probs.shape[0], p=probs/probs.sum())

        if idx == probs.shape[0] - 1 or train_all:
            self.train_model(n_epochs, X_train, y_train)
            curr_date = dttm.strftime(dttm.now(), "%Y%m%d")
            torch.save(self.state_dict(), str(Path(path) / f"{curr_date}_{np.random.randint(1000000)}"))
        else:
            self.load_state_dict(torch.load(str(Path(path) / models[idx])))


