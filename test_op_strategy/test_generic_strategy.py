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
        target = "cuda"
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
