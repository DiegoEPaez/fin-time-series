from unittest import TestCase

from data.multpl_data import query_multpl


class Test(TestCase):
    def test_query_multpl(self):
        res = query_multpl('s-p-500-pe-ratio')
        self.assertAlmostEquals(res.iloc[-1, 0], 11.1)
