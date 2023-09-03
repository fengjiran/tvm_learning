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
        fcombine = lambda x, y: x + y
        fidentity = lambda t: tvm.tir.const(0, dtype=t)
        mysum = te.comm_reducer(fcombine, fidentity)
        fcompute = lambda i: mysum(A[i, k], axis=k)
        C = te.compute((m,), fcompute)
        self.assertTrue(tuple(C.shape) == (m,))

    def test_conv1d(self):
        n = te.size_var("n")
        A = te.placeholder((n + 2), name="A")
        B = te.compute((n,), lambda i: A[i] + A[i + 1] + A[i + 2])
        self.assertTrue(tuple(B.shape) == (n,))

    def test_tensor_slice(self):
        n = te.size_var('n')
        A = te.compute((n, n), lambda i, j: 1)
        B = te.compute((n,), lambda i: A[0][i] + A[0][i])
        self.assertTrue(tuple(A.shape) == (n, n))
        self.assertTrue(tuple(B.shape) == (n,))

    def test_tensor_reduce_multi_axis(self):
        m = te.size_var("m")
        n = te.size_var("n")
        A = te.placeholder((m, n), name="A")
        k1 = te.reduce_axis((0, n), "k1")
        k2 = te.reduce_axis((0, m), "k2")
        B = te.compute((1,), lambda _: te.sum(A[k1, k2], axis=(k1, k2)))
        self.assertTrue(tuple(B.shape) == (1,))

    def test_tensor_scan(self):
        m = te.size_var("m")
        n = te.size_var("n")
        x = te.placeholder((m, n))
        state = te.placeholder((m, n))
        init = te.compute((1, n), lambda _, i: x[0, i])
        update = te.compute((m, n), lambda t, i: state[t - 1, i] + x[t, i])
        scan = te.scan(init, update, state)
        self.assertTrue(tuple(scan.shape) == (m, n))


if __name__ == '__main__':
    unittest.main()
