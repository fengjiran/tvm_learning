import unittest
import numpy as np
import torch
from torch import nn
import tvm
from tvm import te
from tvm import relay
from tvm import topi
from tvm.relay.testing import run_infer_type


class TestGenericStrategy(unittest.TestCase):
    def test_relu_schedule(self):
        dshape = (10, 3, 256, 256)
        dtype = "float32"
        target = "llvm"
        dev = tvm.device(target, 0)

        # get schedule
        A = te.placeholder(dshape, name="A", dtype=dtype)
        T = topi.nn.relu(A)
        s = te.create_schedule(T.op)
        with tvm.transform.PassContext(3):
            func = tvm.build(s, [A, T], target=target, name="relu")

        # get test data
        inps = np.random.rand(*dshape).astype(dtype)

        # verify correctness
        ans = nn.ReLU(True)(torch.from_numpy(inps)).numpy()
        a = tvm.nd.array(inps, dev)
        t = tvm.nd.array(np.zeros(dshape, dtype=dtype), dev)
        func(a, t)
        np.testing.assert_allclose(t.numpy(), ans)

        evaluator = func.time_evaluator(func.entry_name, dev, number=500)
        tvm_time = evaluator(a, t).mean
        print("\nTime: {}s".format(tvm_time))
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
