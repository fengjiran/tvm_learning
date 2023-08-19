import unittest

import tvm
from tvm import get_global_func
from tvm._ffi.registry import list_global_func_names


class TestGlobalFunc(unittest.TestCase):
    def test_get_global_func(self):
        func = get_global_func("relay._transform.FoldConstantExpr")
        self.assertNotEqual(func, None)
        self.assertIsInstance(func, tvm.runtime.PackedFunc)

    @unittest.skip('There are so many global functions.')
    def test_list_global_func_names(self):
        func_names = list_global_func_names()
        self.assertNotEqual(len(func_names), 0)
        print('List all global functions:')
        for name in func_names:
            print(name)
        print('\n')


if __name__ == '__main__':
    unittest.main()
