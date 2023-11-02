import unittest
import numpy as np
import tvm
from tvm import relay


class TestFuseOps(unittest.TestCase):
    def __init__(self):
        super().__init__()
        self.opPatternDict = {
            0: "kElemWise",
            1: "kBroadcast",
            2: "kInjective",
            3: "kCommReduce",
            4: "kOutEWiseFusable",
            7: "kTuple",
            8: "kOpaque"
        }

    def test_fuse_simple(self):
        def before():
            x = relay.var("x", shape=(10, 20))
            y = relay.add(x, relay.const(1, "float32"))
            z = relay.exp(y)
            w = relay.squeeze(z)
            return relay.Function([x], w)

        z = before()
        print(z)


if __name__ == '__main__':
    unittest.main()
