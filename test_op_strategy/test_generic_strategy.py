import os
import unittest
import numpy as np
import torch
from torch import nn
import tvm
from tvm import te
from tvm import relay
from tvm import topi
from tvm.relay.testing import run_infer_type

os.environ['TVM_NUM_THREADS'] = str(1)


class TestGenericStrategy(unittest.TestCase):
    def test_relu_schedule(self):
        dshape = (10, 3, 256, 256)
        dtype = "float32"
        target = "llvm"
        tvm_dev = tvm.device(target, 0)

        # get schedule
        A = te.placeholder(dshape, name="A", dtype=dtype)
        T = topi.nn.relu(A)

        if target == "llvm":
            with tvm.target.Target(target):
                s = topi.x86.schedule_injective(T)
            torch_dev = torch.device("cpu")
        elif target == "cuda":
            with tvm.target.Target(target):
                s = topi.cuda.schedule_injective(T)
            torch_dev = torch.device("cuda:0")
        else:
            raise ValueError("No target of {}".format(target))

        with tvm.transform.PassContext(3):
            func = tvm.build(s, [A, T], target=target, name="relu")

        # get test data
        inps = np.random.rand(*dshape).astype(dtype)
        a = tvm.nd.array(inps, tvm_dev)
        t = tvm.nd.array(np.zeros(dshape, dtype=dtype), tvm_dev)
        func(a, t)

        # verify correctness
        ans = nn.ReLU(True)(torch.from_numpy(inps).to(torch_dev)).to("cpu").numpy()
        np.testing.assert_allclose(t.numpy(), ans)

        # evaluate run time
        evaluator = func.time_evaluator(func.entry_name, tvm_dev, number=50)
        tvm_time = evaluator(a, t).mean
        GFLOPS = np.prod(dshape) * 1e-9
        print("\nTVM without tune time: {}s".format(tvm_time))
        print("TVM without tune GFLOPS: {}".format(GFLOPS / tvm_time))
        print("done")

    def test_matmul_strategy(self):
        m = 1024
        n = 1024
        k = 1024
        dtype = "float32"
        target = "llvm"
        tvm_dev = tvm.device(target, 0)
        A = te.placeholder((m, k), name="A", dtype=dtype)
        B = te.placeholder((k, n), name="B", dtype=dtype)
        red_k = te.reduce_axis((0, k), name="k")
        C = te.compute((m, n), lambda i, j: te.sum(A[i, red_k] * B[red_k, j], axis=red_k), name="C")
        sch = te.create_schedule(C.op)
        mo, no, mi, ni = sch[C].tile(C.op.axis[0], C.op.axis[1], 32, 32)
        kaxis = C.op.reduce_axis[0]
        ko, ki = sch[C].split(kaxis, 4)
        sch[C].reorder(mo, no, ko, mi, ki, ni)
        # sch[C].reorder(kaxis, mo, no, mi, ni)

        with tvm.transform.PassContext(3):
            func = tvm.build(sch, [A, B, C], target=target, name="matmul")

        # get test data
        a = tvm.nd.array(np.random.rand(m, k).astype(dtype), tvm_dev)
        b = tvm.nd.array(np.random.rand(k, n).astype(dtype), tvm_dev)
        c = tvm.nd.array(np.zeros((m, n), dtype=dtype), tvm_dev)
        ans = np.dot(a.numpy(), b.numpy())
        func(a, b, c)
        np.testing.assert_allclose(c.numpy(), ans, rtol=1e-5)

        # evaluate run time and gflops
        evaluator = func.time_evaluator(func.entry_name, tvm_dev, number=10)
        tvm_time = evaluator(a, b, c).mean
        GFLOPS = 2 * m * n * k * 1e-9
        print("\nTVM without tune time: {}s".format(tvm_time))
        print("TVM without tune GFLOPS: {}".format(GFLOPS / tvm_time))
        print("done")

    def test_relu_strategy(self):
        target = tvm.target.Target("llvm")
        dshape = (10, 3, 256, 256)
        op = relay.op.get("nn.relu")
        data = relay.var("data", shape=dshape)
        out = relay.nn.relu(data)
        out = run_infer_type(out)
        impl = relay.backend.te_compiler.get_valid_implementations(
            op,
            out.attrs,
            [te.placeholder(dshape)],
            out.checked_type,
            target
        )
        return impl

    def test_softmax_strategy(self):
        target = tvm.target.Target("llvm")
        shape = (10, 100)
        op = relay.op.get("nn.softmax")
        data = relay.var("data", shape=shape)
        out = relay.nn.softmax(data)
        out = run_infer_type(out)
        impl = relay.backend.te_compiler.get_valid_implementations(
            op,
            out.attrs,
            [te.placeholder(shape)],
            out.checked_type,
            target
        )
        return impl


if __name__ == '__main__':
    unittest.main()
