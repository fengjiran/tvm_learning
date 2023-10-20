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
        gv_name = set([x[0].name_hint for x in mod.functions.items()])
        self.assertEqual(gv_name, {"main"})

    def test_remove_prelude_funcs_but_ref_funcs(self):
        mod = tvm.IRModule()
        x = relay.var("x", shape=(1, 16))
        id_func = relay.Function([x], x)
        id_name = relay.GlobalVar("id_func")
        mod[id_name] = id_func

        mod["main"] = relay.Function([x], id_name(x))
        mod = relay.transform.RemoveUnusedFunctions()(mod)
        gv_name = set([x[0].name_hint for x in mod.functions.items()])
        self.assertEqual(gv_name, {"id_func", "main"})


if __name__ == '__main__':
    unittest.main()
