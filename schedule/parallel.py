import tvm
from tvm import te

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
