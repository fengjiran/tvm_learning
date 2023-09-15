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
        s.cache_read(A, "shared", [T])
        print(tvm.lower(s, [A, T], simple_mode=True))

    def test_cache_write(self):
        n = te.size_var("n")
        k = te.reduce_axis((0, n), name="k")
        A = te.placeholder((n, n), name="A")
        T = te.compute((n,), lambda i: te.sum(A[i, k], axis=k), name="T")
        s = te.create_schedule(T.op)
        print(tvm.lower(s, [A, T], simple_mode=True))
        print('----------------------------cut line-------------------------------')
        s.cache_write(T, "local")
        print(tvm.lower(s, [A, T], simple_mode=True))

    def test_set_scope(self):
        n = 1024
        dtype = 'float32'
        A = te.placeholder((n, n), dtype=dtype, name='A')
        k = te.reduce_axis((0, n), name='k')
        B = te.compute((n,), lambda i: te.sum(A[i, k], axis=k), name='B')
        C = te.compute((n,), lambda i: B[i] + 10, name='C')
        s = te.create_schedule(C.op)

        print(tvm.lower(s, [A, C], simple_mode=True))
        print('----------------------------cut line-------------------------------')
        s[B].set_scope('shared')
        print(tvm.lower(s, [A, C], simple_mode=True))

    def test_compute_at(self):
        m = te.size_var("m")
        A = te.placeholder((m,), name='A')
        B = te.compute((m,), lambda i: A[i] + 1, name='B')
        C = te.compute((m,), lambda i: B[i] * 2, name='C')
        s = te.create_schedule(C.op)
        print(tvm.lower(s, [A, B, C], simple_mode=True))
        print('----------------------------cut line-------------------------------')
        s[B].compute_at(s[C], C.op.axis[0])
        print(tvm.lower(s, [A, B, C], simple_mode=True))

    def test_compute_inline(self):
        m = te.size_var("m")
        A = te.placeholder((m,), name='A')
        B = te.compute((m,), lambda i: A[i] + 1, name='B')
        C = te.compute((m,), lambda i: B[i] * 2, name='C')
        s = te.create_schedule(C.op)
        s[B].compute_inline()
        print(tvm.lower(s, [A, B, C], simple_mode=True))

    def test_vectorize(self):
        m = te.size_var("m")
        n = te.size_var("n")
        A = te.placeholder((m, n), name='A')
        B = te.placeholder((m, n), name='B')
        C = te.compute((m, n), lambda i, j: A[i, j] + B[i, j], name='C')

        s = te.create_schedule(C.op)
        xo, yo, xi, yi = s[C].tile(C.op.axis[0], C.op.axis[1], 32, 32)
        print(tvm.lower(s, [A, B, C], simple_mode=True))
        print('----------------------------cut line-------------------------------')
        s[C].vectorize(yi)
        print(tvm.lower(s, [A, B, C], simple_mode=True))

    def test_unroll(self):
        n = 1024
        A = te.placeholder((n, n), name='A')
        B = te.placeholder((n, n), name='B')
        C = te.compute((n, n), lambda i, j: A[i, j] + B[i, j], name='C')

        s = te.create_schedule(C.op)

        xo, xi = s[C].split(s[C].op.axis[0], factor=4)
        print(tvm.lower(s, [A, B, C], simple_mode=True))
        print('----------------------------cut line-------------------------------')

        s[C].unroll(xi)
        print(tvm.lower(s, [A, B, C], simple_mode=True))

    def test_bind(self):
        n = 1024
        A = te.placeholder((n,), name='A')
        k = te.reduce_axis((0, n), name='k')
        B = te.compute((1,), lambda i: te.sum(A[k], axis=k), name='B')
        s = te.create_schedule(B.op)
        ko, ki = s[B].split(s[B].op.axis[0], factor=32)
        print(tvm.lower(s, [A, B], simple_mode=True))
        print('----------------------------cut line-------------------------------')
        s[B].bind(ko, te.thread_axis("blockIdx.x"))
        s[B].bind(ki, te.thread_axis("threadIdx.x"))
        print(tvm.lower(s, [A, B], simple_mode=True))

    def test_parallel(self):
        m = 1024
        n = 1024
        A = te.placeholder((n, m), name='A')
        k = te.reduce_axis((0, m), name='k')
        B = te.compute((n,), lambda i: te.sum(A[i, k], axis=k), name='B')

        s = te.create_schedule(B.op)
        print(tvm.lower(s, [A, B], simple_mode=True))
        print('----------------------------cut line-------------------------------')
        s[B].parallel(B.op.reduce_axis[0])
        print(tvm.lower(s, [A, B], simple_mode=True))

    def test_rfactor(self):
        n = 1024
        k = te.reduce_axis((0, n), name='k')
        A = te.placeholder((n,), name='A')
        B = te.compute((1,), lambda i: te.sum(A[k], axis=k), name='B')

        s = te.create_schedule(B.op)
        ko, ki = s[B].split(s[B].op.reduce_axis[0], 32)
        print(tvm.lower(s, [A, B], simple_mode=True))

        print('----------------------------cut line-------------------------------')

        s.rfactor(B, ki)
        print(tvm.lower(s, [A, B], simple_mode=True))

    def test_naive_conv(self):
        pass


if __name__ == '__main__':
    unittest.main()
