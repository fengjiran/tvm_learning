import unittest
import numpy as np
import tvm
from tvm import relay


class TestSimplifyInference(unittest.TestCase):
    def test_simplify_batch_norm(self, dtype="float32"):
        pass

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

    def check(self, axis, nstep):
        eps = 0.01



if __name__ == '__main__':
    unittest.main()
