import numpy as np
import tvm
from tvm import te

m = te.var('m')
n = te.var('n')
A = te.placeholder((n, m), name='A')
k = te.reduce_axis((0, m), name='k')
B = te.compute((n,), lambda i: te.sum(A[i, k], axis=k), name='B')

s = te.create_schedule(B.op)
print(tvm.lower(s, [A, B], simple_mode=True))

ko, ki = s[B].split(B.op.reduce_axis[0], factor=16)
BF = s.rfactor(B, ki)
print(tvm.lower(s, [A, B], simple_mode=True))
print('done')