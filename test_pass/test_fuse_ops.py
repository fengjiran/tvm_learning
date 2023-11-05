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
        def before(shape):
            x = relay.var("x", shape=shape)
            x = relay.add(x, relay.const(1, "float32"))
            y = relay.nn.conv2d(x, relay.var("w1"), kernel_size=(3, 3), padding=(1, 1), channels=16)
            y1 = relay.add(y, relay.const(1, "float32"))
            y = relay.add(y, y1)
            z2 = relay.nn.conv2d(y, relay.var("w2"), kernel_size=(1, 1), padding=(0, 0), channels=16)
            z3 = relay.nn.conv2d(y, relay.var("w2"), kernel_size=(3, 3), padding=(1, 1), channels=16)
            z = relay.add(z2, z3)
            return relay.Function(relay.analysis.free_vars(z), z)

        def expected(shape):
            # segment0
            x = relay.var("p0", shape=shape)
            y = relay.add(x, relay.const(1, "float32"))
            f0 = relay.Function([x], y)
            f0 = f0.with_attr("Primitive", tvm.tir.IntImm("int32", 1))

            # segment1
            x = relay.var("p0", shape=shape)
            w = relay.var("p1")
            y = relay.nn.conv2d(x, w, kernel_size=(3, 3), padding=(1, 1), channels=16)
            y1 = relay.add(y, relay.const(1, "float32"))
            y = relay.add(y, y1)
            f1 = relay.Function([x, w], y)
            f1 = f1.with_attr("Primitive", tvm.tir.IntImm("int32", 1))

            # segment 2
            x = relay.var("p0", shape=shape)
            w = relay.var("p1")
            z2 = relay.nn.conv2d(x, w, kernel_size=(3, 3), padding=(1, 1), channels=16)
            f2 = relay.Function([x, w], z2)
            f2 = f2.with_attr("Primitive", tvm.tir.IntImm("int32", 1))

            # segment 3
            x = relay.var("p0", shape=shape)
            w = relay.var("p1")
            offset = relay.var("p2", shape=shape)
            z3 = relay.nn.conv2d(x, w, kernel_size=(1, 1), padding=(0, 0), channels=16)
            z3 = relay.add(z3, offset)
            f3 = relay.Function([x, w, offset], z3)
            f3 = f3.with_attr("Primitive", tvm.tir.IntImm("int32", 1))

            # compose
            x = relay.var("x", shape=shape)
            y = relay.Call(f0, [x])
            y = relay.Call(f1, [y, relay.var("w1")])
            z2 = relay.Call(f2, [y, relay.var("w3")])
            z3 = relay.Call(f3, [y, relay.var("w2"), z2])
            return relay.Function(relay.analysis.free_vars(z3), z3)

        shape = (1, 16, 64, 64)
        z = before(shape)
        zz = run_opt_pass(z, relay.transform.FuseOps(2))
        print(zz)
        after = run_opt_pass(expected(shape), relay.transform.InferType())
        self.assertTrue(tvm.ir.structural_equal(zz, after))

    def test_concat(self):
        """Test fusion case involving cancat op and Tuple node."""

        def before(shape: tuple) -> relay.Function:
            x = relay.var("x", shape=shape)
            pooled = relay.nn.max_pool2d(x, pool_size=(2, 2), strides=(2, 2), padding=(0, 0))
            upsampled = relay.nn.upsampling(pooled, scale_h=2, scale_w=2, layout="NCHW")
            concat = relay.concatenate((upsampled, x), axis=1)
            out = relay.add(concat, relay.const(1, "float32"))
            return relay.Function(relay.analysis.free_vars(out), out)

        def expected(shape: tuple) -> relay.Function:
            # segment0
            x = relay.var("x", shape=shape)
            pooled = relay.nn.max_pool2d(x, pool_size=(2, 2), strides=(2, 2), padding=(0, 0))
            f0 = relay.Function([x], pooled)
            f0 = f0.with_attr("Primitive", tvm.tir.IntImm("int32", 1))

            # segment1
            p0 = relay.var("p0", shape=(shape[0], shape[1], shape[2] // 2, shape[3] // 2))
            p1 = relay.var("p1", shape=shape)
            upsampled = relay.nn.upsampling(p0, scale_h=2, scale_w=2, layout="NCHW")
            concat = relay.concatenate((upsampled, p1), axis=1)
            out = relay.add(concat, relay.const(1, "float32"))
            f1 = relay.Function([p0, p1], out)
            f1 = f1.with_attr("Primitive", tvm.tir.IntImm("int32", 1))

            x = relay.var("x", shape=shape)
            y = relay.Call(f0, [x])
            z = relay.Call(f1, [y, x])
            return relay.Function([x], z)

        shape = (1, 16, 64, 64)
        z = before(shape)
        zz = run_opt_pass(z, relay.transform.FuseOps(2))
        after = run_opt_pass(expected(shape), relay.transform.InferType())
        self.assertTrue(tvm.ir.structural_equal(zz, after))

    def test_root_tuple(self):
        """Test fusion case where Tuple node is the root in its group."""

        def before(shape: tuple) -> relay.Function:
            x = relay.var("x", shape=shape)
            pooled = relay.nn.max_pool2d(x, pool_size=(2, 2), strides=(2, 2), padding=(0, 0))
            upsampled = relay.nn.upsampling(pooled, scale_h=2, scale_w=2, layout="NCHW")
            out = relay.Tuple((upsampled, x))
            return relay.Function(relay.analysis.free_vars(out), out)

        def expected(shape: tuple) -> relay.Function:
            x = relay.var("x", shape=shape)
            pooled = relay.nn.max_pool2d(x, pool_size=(2, 2), strides=(2, 2), padding=(0, 0))
            f0 = relay.Function([x], pooled)
            f0 = f0.with_attr("Primitive", tvm.tir.IntImm("int32", 1))

            p0 = relay.var("p0", shape=(shape[0], shape[1], shape[2] // 2, shape[3] // 2))
            upsampled = relay.nn.upsampling(p0, scale_h=2, scale_w=2, layout="NCHW")
            f1 = relay.Function([p0], upsampled)
            f1 = f1.with_attr("Primitive", tvm.tir.IntImm("int32", 1))

            x = relay.var("x", shape=shape)
            y = relay.Call(f0, [x])
            z = relay.Call(f1, [y])
            tup = relay.Tuple((z, x))
            return relay.Function([x], tup)

        shape = (1, 16, 64, 64)
        z = before(shape)
        zz = run_opt_pass(z, relay.transform.FuseOps(2))
        after = run_opt_pass(expected(shape), relay.transform.InferType())
        self.assertTrue(tvm.ir.structural_equal(zz, after))


if __name__ == '__main__':
    unittest.main()
