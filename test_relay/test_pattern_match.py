import unittest
import tvm
from tvm import relay
from tvm.relay.dataflow_pattern import *


class TestPatternMatch(unittest.TestCase):
    def test_expr_pattern(self):
        expr = is_expr(relay.var("x", shape=(4, 1)))
        self.assertTrue(isinstance(expr, ExprPattern))
        self.assertTrue(isinstance(expr.expr, relay.Var))

    def test_var_pattern(self):
        v = is_var("x")
        self.assertTrue(isinstance(v, VarPattern))
        self.assertTrue(v.name == "x")

    def test_constant_pattern(self):
        c = is_constant()
        self.assertTrue(isinstance(c, ConstantPattern))


if __name__ == '__main__':
    unittest.main()
