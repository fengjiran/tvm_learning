import unittest
import tvm
from tvm import te


class TestTESchedule(unittest.TestCase):
    def test_split(self):
        n = te.size_var("n") # 1024
        k = te.reduce_axis((0, n), name='k')
        A = te.placeholder((n,), name="A")
        T = te.compute((1,), lambda i: te.sum(A[k], axis=k), name='T')
        s = te.create_schedule(T.op)
        ko, ki = s[T].split(T.op.reduce_axis[0], factor=16)
        print(tvm.lower(s, [A, T], simple_mode=True))

    def test_fuse(self):
        pass

    def test_tile(self):
        m = te.size_var("m")
        n = te.size_var("n")
        k = te.size_var("k")
        A = te.placeholder((m, k), name="A")
        B = te.placeholder((k, n), name="B")


if __name__ == '__main__':
    unittest.main()
