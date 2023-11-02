import unittest
import numpy as np
import tvm
from tvm import relay
from test_fold_constant import run_opt_pass

opPatternDict = {
    0: "kElemWise",
    1: "kBroadcast",
    2: "kInjective",
    3: "kCommReduce",
    4: "kOutEWiseFusable",
    7: "kTuple",
    8: "kOpaque"
}


class TestFuseOps(unittest.TestCase):
    def test_fuse_simple(self):
        def before():
            x = relay.var("x", shape=(10, 20))
            y = relay.add(x, relay.const(1, "float32"))
            z = relay.exp(y)
            w = relay.squeeze(z)
            print(y.op.name + " Pattern: " + opPatternDict[y.op.get_attr("TOpPattern")])
            print(z.op.name + " Pattern: " + opPatternDict[z.op.get_attr("TOpPattern")])
            print(w.op.name + " Pattern: " + opPatternDict[w.op.get_attr("TOpPattern")])
            return relay.Function([x], w)

        def expected():
            x = relay.var("p", shape=(10, 20))
            y = relay.add(x, relay.const(1, "float32"))
            z = relay.exp(y)
            w = relay.squeeze(z)
            f1 = relay.Function([x], w)
            f1 = f1.with_attr("Primitive", tvm.tir.IntImm("int32", 1))
            x = relay.var("x", shape=(10, 20))
            y = relay.Call(f1, [x])
            return relay.Function([x], y)

        z = before()
        zz = run_opt_pass(z, relay.transform.FuseOps())
        after = run_opt_pass(expected(), relay.transform.InferType())
        self.assertTrue(tvm.ir.structural_equal(zz, after))

    def test_conv2d_fuse(self):
        pass


if __name__ == '__main__':
    unittest.main()
