import unittest
import tvm
from tvm import relay
from tvm.relay.dataflow_pattern import *


class TestPatternMatch(unittest.TestCase):
    def test_match_op(self):
        assert is_op("add").match(relay.op.op.get("add"))


if __name__ == '__main__':
    unittest.main()
