from unittest import TestCase
import numpy as np
import pandas as pd

from pred import stationary, create_lags, quantiles


class Test(TestCase):
    def test_stationary1(self):
        data = pd.Series(data=np.arange(100), name="Linear")
        expected = data[1:].values - data[:-1].values
        result, type = stationary(data, 1)

        np.array_equal(expected, result)
        self.assertEqual("dif", type)

    def test_stationary2(self):
        data = pd.Series(data=np.arange(100), name="Squared with zero") ** 2
        expected = data
        result, type = stationary(data, 1)

        np.array_equal(expected, result)
        self.assertEqual(type, None)

    def test_stationary3(self):
        data = pd.Series(data=np.arange(1, 1001), name="Squared") ** 2
        expected = data[1:].values - data[:-1].values
        result, type = stationary(data, 1)

        np.array_equal(expected, result)
        self.assertEqual(type, "dif")

    def test_stationary4(self):
        data = pd.Series(data=np.arange(1, 1001), name="Cubed") ** 3
        expected = np.log(data[1:].values) - np.log(data[:-1].values)
        result, type = stationary(data, 1)

        np.array_equal(expected, result)
        self.assertEqual(type, "log_dif")

    def test_create_lags1(self):
        data = np.arange(1, 11)
        n_steps = 1
        pred_interval = 5
        expected = np.array([[1, 6], [2, 7], [3, 8], [4, 9], [5, 10]])
        result, n_steps_res = create_lags(data, n_steps, pred_interval)

        np.array_equal(expected, result)
        self.assertEqual(n_steps, n_steps_res)

    def test_create_lags2(self):
        data = np.arange(1, 11)
        n_steps = 2
        pred_interval = 5
        expected = np.array([[1, 6], [2, 7], [3, 8], [4, 9], [5, 10]])
        result, n_steps_res = create_lags(data, n_steps, pred_interval)

        np.array_equal(expected, result)
        self.assertEqual(1, n_steps_res)

    def test_create_lags3(self):
        data = np.c_[np.arange(1, 11), np.arange(11, 21)]
        n_steps = 1
        pred_interval = 5
        expected = np.array(
            [[[1, 11], [6, 16]], [[2, 12], [7, 17]], [[3, 13], [8, 18]], [[4, 14], [9, 19]], [[5, 15], [10, 20]]])
        result, n_steps_res = create_lags(data, n_steps, pred_interval)

        np.array_equal(expected, result)
        self.assertEqual(1, n_steps_res)

    def test_quantiles(self):
        data = np.array([0, 1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11])
        expected = np.array([1., 2., 3., 4., 5., 6., 7., 8., 9., 10.])
        qs = quantiles(data, 10)
        np.array_equal(expected, qs)