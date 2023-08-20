import unittest
import numpy as np
import tvm
from tvm import relay


def run_opt_pass(expr, opt_pass):
    pass


def annot_expr(expr):
    return relay.op.annotation.on_device(expr, tvm.cpu(), constrain_result=True)


class TestFoldConstant(unittest.TestCase):
    pass


if __name__ == '__main__':
    unittest.main()
