import tvm
from tvm import te

n = 1024

A = te.placeholder((n,), name='A')
B = te.compute((n,), lambda i: A[i] + 1, name='B')
C = te.compute((n,), lambda i: B[i] * 2, name='C')
s = te.create_schedule(C.op)
s[B].compute_at(s[C], C.op.axis[0])
s[B].compute_root()
print(tvm.lower(s, [A, B, C], simple_mode=True))
