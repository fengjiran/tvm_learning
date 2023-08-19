import unittest
import tvm
from tvm import relay
import numpy as np


def check_json_roundtrip(node):
    json_str = tvm.ir.save_json(node)
    back = tvm.ir.load_json(json_str)
    assert tvm.ir.structural_equal(back, node, map_free_vars=True)


class TestRelayIR(unittest.TestCase):
    def test_constant(self):
        arr = np.random.uniform(0, 10, size=(3, 4))
        const = relay.Constant(tvm.nd.array(arr))
        np.testing.assert_array_equal(const.data.numpy(), arr)
        self.assertIsNone(const.span)
        print(const)
        # str(const)
        check_json_roundtrip(const)


if __name__ == '__main__':
    unittest.main()
