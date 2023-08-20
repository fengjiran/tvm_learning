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
        arr = np.random.uniform(0, 10, size=(3, 4)).astype(np.float32)
        const1 = relay.Constant(tvm.nd.array(arr))
        const2 = relay.const(arr)
        np.testing.assert_array_equal(const1.data.numpy(), arr)
        np.testing.assert_array_equal(const2.data.numpy(), arr)
        self.assertIsNone(const1.span)
        print(const1)
        print(const2)
        # str(const)
        check_json_roundtrip(const1)

    def test_tuple(self):
        fields = tvm.runtime.convert([])
        tup = relay.Tuple(fields)
        self.assertEqual(tup.fields, fields)
        self.assertIsNone(tup.span)
        print(tup)
        str(tup)
        check_json_roundtrip(tup)

    def test_local_var(self):
        name_hint = 's'
        local_var = relay.Var(name_hint)
        self.assertEqual(local_var.name_hint, name_hint)
        self.assertIsNone(local_var.type_annotation)
        str(local_var)
        print(local_var)
        check_json_roundtrip(local_var)

    def test_global_var(self):
        name_hint = 'g'
        gv = relay.GlobalVar(name_hint)
        self.assertEqual(gv.name_hint, name_hint)
        str(gv)
        print(gv)
        check_json_roundtrip(gv)


if __name__ == '__main__':
    unittest.main()
