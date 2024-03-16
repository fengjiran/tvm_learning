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

    def test_wildcard_pattern(self):
        wc = wildcard()
        self.assertTrue(isinstance(wc, WildcardPattern))

    def test_CallPattern(self):
        wc1 = wildcard()
        wc2 = wildcard()
        c = is_op("add")(wc1, wc2)
        assert isinstance(c, CallPattern)
        assert isinstance(c.args[0], WildcardPattern)
        assert isinstance(c.args[1], WildcardPattern)


if __name__ == '__main__':
    unittest.main()
