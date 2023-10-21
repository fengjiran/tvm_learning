import unittest
import numpy as np
import tvm
from tvm import relay
from tvm.ir import structural_equal
from tvm.relay.transform import InferType
from tvm.relay.transform import SimplifyInference


class TestSimplifyInference(unittest.TestCase):
    def test_simplify_batch_norm(self, dtype="float32"):
        self.check(2, 1, 1, dtype)
        self.check(4, 1, 1, dtype)
        self.check(4, 0, 3, dtype)

    def simple_batch_norm(self, x, gamma, beta, moving_mean, moving_var, axis=1, epsilon=1e-5, shape=None,
                          dtype="float32"):
        # expect = (x - moving_mean) / sqrt(moving_var + epsilon) * gamma + beta
        scale = relay.multiply(relay.const(1, dtype) / relay.sqrt(moving_var + relay.const(epsilon, dtype)), gamma)
        shift = relay.add(relay.multiply(relay.negative(moving_mean), scale), beta)
        num_newaxis = len(shape) - (axis + 1)
        if num_newaxis:
            scale = relay.expand_dims(scale, axis=1, num_newaxis=num_newaxis)
            shift = relay.expand_dims(shift, axis=1, num_newaxis=num_newaxis)
        return x * scale + shift

    def check(self, dim, axis, nstep, dtype):
        eps = 0.01
        ttype1 = relay.TensorType(tuple(10 for _ in range(dim)), dtype)
        ttype2 = relay.TensorType((10,), dtype)
        x = relay.var("x", ttype1)
        beta = relay.var("beta", ttype2)
        gamma = relay.var("gamma", ttype2)
        moving_mean = relay.var("moving_mean", ttype2)
        moving_var = relay.var("moving_var", ttype2)
        y1 = x
        y2 = x
        for _ in range(nstep):
            y1, _, _ = relay.nn.batch_norm(
                y1 + relay.const(1, dtype),
                gamma,
                beta,
                moving_mean,
                moving_var,
                epsilon=eps,
                axis=axis
            )
            y1 = relay.nn.dropout(y1)

            y2 = self.simple_batch_norm(
                y2 + relay.const(1, dtype),
                gamma,
                beta,
                moving_mean,
                moving_var,
                epsilon=eps,
                axis=axis,
                shape=ttype1.shape,
                dtype=dtype
            )

        mod = tvm.IRModule.from_expr(y1)
        mod = InferType()(mod)
        mod = SimplifyInference()(mod)
        y1 = mod["main"].body
        self.assertTrue(structural_equal(y1, y2, map_free_vars=True))


if __name__ == '__main__':
    unittest.main()
