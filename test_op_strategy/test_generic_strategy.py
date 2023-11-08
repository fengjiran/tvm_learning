import unittest
import numpy as np
import tvm
from tvm import te
from tvm import relay
from tvm.relay.testing import run_infer_type


class TestGenericStrategy(unittest.TestCase):
    def test_softmax_strategy(self):
        target = tvm.target.Target("cuda")
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
