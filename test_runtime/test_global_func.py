import unittest
from tvm import get_global_func


class TestGlobalFunc(unittest.TestCase):
    def test_get_global_func(self):
        func = get_global_func("relay._transform.FoldConstantExpr")
        self.assertNotEqual(func, None)


if __name__ == '__main__':
    unittest.main()
