import tvm
from tvm import te

n = 1024
A = te.placeholder((n, n), name='A')
B = te.placeholder((n, n), name='B')
k = te.reduce_axis((0, n), name='k')
C = te.compute((n, n), lambda i, j: te.sum(A[i, k] * B[k, j], axis=k), name='C')

s = te.create_schedule(C.op)
print(tvm.lower(s, [A, B, C], simple_mode=True))
print('----------------------------cut line-------------------------------')
xo, yo, xi, yi = s[C].tile(C.op.axis[0], C.op.axis[1], 32, 32)
print(tvm.lower(s, [A, B, C], simple_mode=True))
