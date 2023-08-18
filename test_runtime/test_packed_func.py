import unittest
import tvm


class TestPackedFunc(unittest.TestCase):
    def test_get_global_func(self):
        targs = (10, 10.0, "hello")

        # register into global function table
        @tvm.register_func
        def my_packed_func(*args):
            assert tuple(args) == targs
            return 10

        # get it from global function table
        f = tvm.get_global_func('my_packed_func')
        self.assertIsInstance(f, tvm.runtime.PackedFunc)
        y = f(*targs)
        self.assertEqual(y, 10)


if __name__ == '__main__':
    unittest.main()
