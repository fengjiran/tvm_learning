import unittest
import numpy as np
import tvm
from tvm import te


class TestTE(unittest.TestCase):
    def test_tensor(self):
        m = te.size_var('m')
        n = te.size_var('n')
        l = te.size_var('l')


if __name__ == '__main__':
    unittest.main()
