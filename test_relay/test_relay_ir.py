import unittest
import tvm


def check_json_roundtrip(node):
    json_str = tvm.ir.save_json(node)
    back = tvm.ir.load_json(json_str)
    assert tvm.ir.structural_equal(back, node, map_free_vars=True)


class TestRelayIR(unittest.TestCase):
    pass
