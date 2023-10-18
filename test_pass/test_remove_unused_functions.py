import unittest
import numpy as np
import tvm
from tvm import relay


class TestRemoveUnusedFunctions(unittest.TestCase):
    def test_remove_all_prelude_functions(self):
        mod = tvm.IRModule()
        x = relay.var("x", shape=(1, 16))
        mod["main"] = relay.Function([x], x)
        mod = relay.transform.RemoveUnusedFunctions()(mod)
        l = set([x[0].name_hint for x in mod.functions.items()])
        self.assertEqual(l, {"main"})


if __name__ == '__main__':
    unittest.main()
