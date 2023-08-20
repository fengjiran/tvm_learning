import unittest
import numpy as np
import tvm
from tvm import relay


def run_opt_pass(expr, opt_pass):
    assert isinstance(opt_pass, tvm.transform.Pass)
    mod = tvm.IRModule.from_expr(expr)
    mod = relay.transform.InferType()(mod)
    mod = opt_pass(mod)
    entry = mod['main']
    return entry if isinstance(expr, relay.Function) else entry.body


def annot_expr(expr):
    return relay.op.annotation.on_device(expr, tvm.cpu(), constrain_result=True)


class TestFoldConstant(unittest.TestCase):
    def test_fold_constant(self):
        c_data = np.array([1, 2, 3]).astype('float32')
        t = relay.TensorType([1, 2, 3], 'float32')

        def befor():
            c = relay.const(c_data)
            x = relay.var('x', t)
            y = relay.add(c, c)
            y = relay.multiply(y, relay.const(2, 'float32'))
            y = relay.add(x, y)
            z = relay.add(y, c)
            return relay.Function([x], z)


if __name__ == '__main__':
    unittest.main()
