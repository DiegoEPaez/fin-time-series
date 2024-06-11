from unittest import TestCase
import pandas as pd

from technicals import oscillators


class Test(TestCase):

    def setUp(self) -> None:
        self.spx23 = pd.read_csv('SPX23.csv', index_col='Date', parse_dates=True)

    def test_rsi(self):
        # According to tradingview.com RSI is 48.32, small difference due to either closing prices or ewma algorithm
        rsi = oscillators.rsi(self.spx23, 'Adj_Close')
        rsi_correct = 48.50353
        self.assertAlmostEqual(rsi.values[-1], rsi_correct, places=4)

