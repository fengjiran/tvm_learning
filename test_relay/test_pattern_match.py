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

    def test_FunctionPattern(self):
        wc1 = wildcard()
        wc2 = wildcard()
        c = is_op("add")(wc1, wc2)
        f = FunctionPattern([wc1, wc2], c)
        assert isinstance(f, FunctionPattern)
        assert isinstance(f.params[0], WildcardPattern)
        assert isinstance(f.params[1], WildcardPattern)
        assert isinstance(f.body, CallPattern)
        assert isinstance(f.body.args[0], WildcardPattern)
        assert isinstance(f.body.args[1], WildcardPattern)

    def test_TuplePattern(self):
        wc1 = wildcard()
        wc2 = wildcard()
        t = is_tuple([wc1, wc2])
        assert isinstance(t, TuplePattern)
        assert isinstance(t.fields[0], WildcardPattern)
        assert isinstance(t.fields[1], WildcardPattern)

    def test_TupleGetItemPattern(self):
        wc1 = wildcard()
        wc2 = wildcard()
        t = is_tuple([wc1, wc2])
        tgi = is_tuple_get_item(t, 1)
        assert isinstance(tgi, TupleGetItemPattern)
        assert isinstance(tgi.tuple, TuplePattern)
        assert isinstance(tgi.tuple.fields[0], WildcardPattern)
        assert isinstance(tgi.tuple.fields[1], WildcardPattern)

    def test_AltPattern(self):
        is_add_or_sub = is_op("add") | is_op("subtract")
        assert isinstance(is_add_or_sub, AltPattern)

    def test_TypePattern(self):
        ttype = relay.TensorType((10, 10), "float32")
        ty_pat = has_type(ttype)
        assert isinstance(ty_pat, TypePattern)
        assert ty_pat.type == ttype

    def test_DataTypePattern(self):
        dtype = "float16"
        pattern = has_dtype(dtype)
        assert isinstance(pattern, DataTypePattern)
        assert pattern.dtype == dtype

    def test_ShapePattern(self):
        shape = [10, 10]
        pattern = has_shape(shape)
        assert isinstance(pattern, ShapePattern)
        assert tvm.ir.structural_equal(pattern.shape, shape)


if __name__ == '__main__':
    unittest.main()
