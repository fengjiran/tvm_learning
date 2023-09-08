import unittest
import tvm
from tvm import te


class TestTESchedule(unittest.TestCase):
    def test_split(self):
        n = te.size_var("n")  # 1024
        k = te.reduce_axis((0, n), name='k')
        A = te.placeholder((n,), name="A")
        T = te.compute((1,), lambda i: te.sum(A[k], axis=k), name='T')
        s = te.create_schedule(T.op)
        ko, ki = s[T].split(T.op.reduce_axis[0], factor=16)
        print(tvm.lower(s, [A, T], simple_mode=True))

    def test_fuse(self):
        m = te.size_var("m")
        n = te.size_var("n")
        A = te.placeholder((m, n), name="A")
        T = te.compute((m, n), lambda i, j: A[i, j])

        s = te.create_schedule(T.op)
        xo, yo, xi, yi = s[T].tile(T.op.axis[0], T.op.axis[1], x_factor=10, y_factor=5)
        fused = s[T].fuse(xo, yo)
        self.assertTrue(any(isinstance(x, te.schedule.Fuse) for x in s[T].relations))
        self.assertTrue(tuple(s[T].leaf_iter_vars) == (fused, xi, yi))

    def test_reorder(self):
        m = te.size_var("m")
        n = te.size_var("n")
        A = te.placeholder((m, n), name="A")
        B = te.placeholder((m, n), name="B")
        C = te.compute((m, n), lambda i, j: A[i, j] + B[i, j], name="C")
        s = te.create_schedule(C.op)
        self.assertEqual(len(s.stages), 3)
        print(tvm.lower(s, [A, B, C], simple_mode=True))
        xo, xi = s[C].split(C.op.axis[0], factor=32)
        yo, yi = s[C].split(C.op.axis[1], factor=32)
        s[C].reorder(xo, yo, yi, xi)
        print(tvm.lower(s, [A, B, C], simple_mode=True))

    def test_tile(self):
        m = te.size_var("m")
        n = te.size_var("n")
        k = te.size_var("k")
        red_k = te.reduce_axis((0, k), name="red_k")

        A = te.placeholder((m, k), name="A")
        B = te.placeholder((k, n), name="B")
        T = te.compute((m, n), lambda i, j: te.sum(A[i, red_k] * B[red_k, j], axis=red_k))
        s = te.create_schedule(T.op)
        xo, yo, xi, yi = s[T].tile(T.op.axis[0], T.op.axis[1], x_factor=10, y_factor=5)
        self.assertTrue(tuple(s[T].leaf_iter_vars[:4]) == (xo, yo, xi, yi))

    def test_cache_read(self):
        m = te.size_var("m")
        k = te.reduce_axis((0, m), name="k")
        A = te.placeholder((m, m), name="A")
        T = te.compute((m,), lambda i: te.sum(A[i, k], axis=k), name="T")
        s = te.create_schedule(T.op)
        print(tvm.lower(s, [A, T], simple_mode=True))


if __name__ == '__main__':
    unittest.main()
