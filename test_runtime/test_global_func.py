import unittest
from tvm import get_global_func
from tvm._ffi.registry import list_global_func_names


class TestGlobalFunc(unittest.TestCase):
    def test_get_global_func(self):
        func = get_global_func("relay._transform.FoldConstantExpr")
        self.assertNotEqual(func, None)

    def test_list_global_func_names(self):
        func_names = list_global_func_names()
        self.assertNotEqual(len(func_names), 0)


if __name__ == '__main__':
    unittest.main()
