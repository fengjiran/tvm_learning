import unittest
import tvm
from tvm import te


class TestTESchedule(unittest.TestCase):
    def test_split(self):
        n = 1024
        k = te.reduce_axis((0, n), name='k')
        A = te.placeholder((n,), name="A")
        T = te.compute((1,), lambda i: te.sum(A[k], axis=k), name='T')
        s = te.create_schedule(T.op)
        ko, ki = s[T].split(T.op.reduce_axis[0], factor=16)


if __name__ == '__main__':
    unittest.main()