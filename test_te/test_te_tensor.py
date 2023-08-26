import unittest
import numpy as np
import tvm
from tvm import te


class TestTE(unittest.TestCase):
    def test_tensor(self):
        m = te.size_var('m')
        n = te.size_var('n')
        l = te.size_var('l')
        A = te.placeholder((m, l), name='A')
        B = te.placeholder((n, l), name='B')
        T = te.compute((m, n, l), lambda i, j, k: A[i, k] * B[j, k])
        print(T)
        print(T.op.body)
        self.assertTrue(tuple(T.shape) == (m, n, l))
        self.assertIsInstance(A.op, te.PlaceholderOp)
        self.assertTrue(T.op.output(0) == T)
        self.assertTrue(T.op.output(0).__hash__() == T.__hash__())
        d = {T.op.output(0): 1}
        self.assertTrue(d[T] == 1)
        self.assertTrue(T[0][0][0].astype("float16").dtype == "float16")

    def test_rank_zero(self):
        m = te.size_var('m')
        A = te.placeholder((m,), name='A')
        scale = te.placeholder((), name='scale')
        k = te.reduce_axis((0, m), name='k')
        T = te.compute((), lambda: te.sum(A[k] * scale(), axis=k))
        print(T)
        print(T.op.body)
        self.assertTrue(tuple(T.shape) == ())

    def test_common_reducer(self):
        m = te.size_var("m")
        n = te.size_var("n")
        A = te.placeholder((m, n), name="A")
        k = te.reduce_axis((0, n), "k")
        mysum = te.comm_reducer(lambda x, y: x + y, lambda t: tvm.tir.const(0, dtype=t))
        C = te.compute((m,), lambda i: mysum(A[i, k], axis=k))


if __name__ == '__main__':
    unittest.main()
