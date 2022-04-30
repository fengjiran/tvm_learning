import tvm
from tvm import te

n = 1024
k = te.reduce_axis((0, n), name='k')
A = te.placeholder((n,), name='A')
B = te.compute((1,), lambda i: te.sum(A[k], axis=k), name='B')

s = te.create_schedule(B.op)
ko, ki = s[B].split(B.op.reduce_axis[0], factor=32)
print(tvm.lower(s, [A, B], simple_mode=True))
print('----------------------------cut line-------------------------------')

s[B].fuse(ko, ki)
print(tvm.lower(s, [A, B], simple_mode=True))
