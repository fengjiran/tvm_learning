import unittest
import tvm


class TestPackedFunc(unittest.TestCase):
    def test_convert(self):
        targs = (10, 10.0, "hello")

        def myfunc(*args):
            assert tuple(args) == targs

        f = tvm.runtime.convert(myfunc)
        self.assertIsInstance(f, tvm.runtime.PackedFunc)

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

    def test_get_callback_with_node(self):
        x = tvm.runtime.convert(10)

        def test(y):
            assert y.handle != x.handle
            return y

        f2 = tvm.runtime.convert(test)
        self.assertIsInstance(f2, tvm.runtime.PackedFunc)

        @tvm.register_func
        def my_callback_with_node(y, f):
            assert y == x
            return f(y)

        f = tvm.get_global_func('my_callback_with_node')
        self.assertIsInstance(f, tvm.runtime.PackedFunc)
        y = f(x, f2)
        self.assertEqual(y.value, 10)


if __name__ == '__main__':
    unittest.main()
