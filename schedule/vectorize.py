import tvm
from tvm import te

n = 1024
A = te.placeholder((n, n), name='A')
B = te.placeholder((n, n), name='B')
C = te.compute((n, n), lambda i, j: A[i, j] + B[i, j], name='C')

s = te.create_schedule(C.op)
xo, yo, xi, yi = s[C].tile(C.op.axis[0], C.op.axis[1], 32, 32)
print(tvm.lower(s, [A, B, C], simple_mode=True))
print('----------------------------cut line-------------------------------')
s[C].vectorize(yi)
print(tvm.lower(s, [A, B, C], simple_mode=True))
