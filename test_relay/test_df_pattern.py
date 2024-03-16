import unittest
import tvm
from tvm import relay
from tvm.relay.dataflow_pattern import *

# NB: 1 corresponds to the C++ enum that specicfies this
# we loose the type safety due to the Python/C++ calling
# convention.
K_ELEMWISE = 0
K_BROADCAST = 1


class TestDFPattern(unittest.TestCase):
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

    def test_AttrPattern(self):
        op = is_op("add").has_attr({"TOpPattern": K_ELEMWISE})
        assert isinstance(op, AttrPattern)
        assert op.attrs["TOpPattern"] == K_ELEMWISE

    def test_IfPattern(self):
        x = is_var("x")
        y = is_var("y")
        pat = is_if(is_op("less")(x, y), x, y)

        assert isinstance(pat, IfPattern)
        assert isinstance(pat.cond, CallPattern)
        assert isinstance(pat.true_branch, VarPattern)
        assert isinstance(pat.false_branch, VarPattern)

    def test_LetPattern(self):
        x = is_var("x")
        y = is_var("y")
        let_var = is_var("let")
        pat = is_let(let_var, is_op("less")(x, y), let_var)

        assert isinstance(pat, LetPattern)
        assert isinstance(pat.var, VarPattern)
        assert isinstance(pat.value, CallPattern)
        assert isinstance(pat.body, VarPattern)


if __name__ == '__main__':
    unittest.main()
