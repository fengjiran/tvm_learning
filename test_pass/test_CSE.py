import unittest
import tvm
from tvm import relay
from tvm.ir import structural_equal
from tvm.relay.transform import InferType
from tvm.relay.transform import EliminateCommonSubexpr


class TestCommonSubexprEliminate(unittest.TestCase):
    def test_simple(self):
        def before():
            x = relay.var("x", shape=(1, 16))
            y1 = relay.nn.relu(x)
            y2 = relay.nn.relu(x)
            y1 = relay.add(y1, relay.const(1.0, "float32"))
            y2 = relay.add(y2, relay.const(1.0, "float32"))
            y = relay.add(y1, y2)
            return y

        def expected():
            x = relay.var("x", shape=(1, 16))
            y = relay.nn.relu(x)
            y = relay.add(y, relay.const(1.0, "float32"))
            y = relay.add(y, y)
            return y

        z = before()
        mod = tvm.IRModule.from_expr(z)
        mod = InferType()(mod)
        mod = EliminateCommonSubexpr()(mod)
        self.assertTrue(structural_equal(mod["main"].body, expected(), map_free_vars=True))

    def test_skip(self):
        def before():
            x = relay.var("x", shape=(1, 16))
            y1 = relay.nn.relu(x)
            y2 = relay.nn.relu(x)
            y1 = relay.add(y1, relay.const(1.0, "float32"))
            y2 = relay.add(y2, relay.const(1.0, "float32"))
            y = relay.add(y1, y2)
            return y

        def expected():
            x = relay.var("x", shape=(1, 16))
            y = relay.nn.relu(x)
            y1 = relay.add(y, relay.const(1.0, "float32"))
            y2 = relay.add(y, relay.const(1.0, "float32"))
            y = relay.add(y1, y2)
            return y

        def fskip(expr):
            if isinstance(expr, relay.expr.Call) and expr.op.name == "add":
                return True
            return False

        z = before()
        mod = tvm.IRModule.from_expr(z)
        mod = InferType()(mod)
        mod = EliminateCommonSubexpr(fskip)(mod)
        self.assertTrue(structural_equal(mod["main"].body, expected(), map_free_vars=True))


if __name__ == '__main__':
    unittest.main()
