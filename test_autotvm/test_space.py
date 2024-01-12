import unittest
from tvm import te
from tvm.autotvm.task.space import ConfigSpace


class TestAutotvmSpace(unittest.TestCase):
    def gemm_func(self, cfg, M, filter_x=None, filter_y=None):
        A = te.placeholder((M, M), name="A")
        B = te.placeholder((M, M), name="B")
        k = te.reduce_axis((0, M), name="k")
        C = te.compute((M, M), lambda i, j: te.sum(A[i, k] * B[k, j], axis=k), name="C")
        s = te.create_schedule(C.op)

        cfg.define_split("split_y", s[C].op.axis[0], num_outputs=2, filter=filter_y)
        cfg.define_split("split_x", s[C].op.axis[1], num_outputs=2, filter=filter_x)
        return s, [A, B, C]

    def test_split(self):
        cfg = ConfigSpace()
        self.gemm_func(cfg, 128)
        self.assertTrue(cfg.range_length == 64)
        self.assertTrue(len(cfg.space_map["split_y"]) == 8)

    def test_policy(self):
        cfg = ConfigSpace()
        cfg.define_split("tile_x", cfg.axis(256), policy="factors", num_outputs=3)
        self.assertTrue(len(cfg.space_map["tile_x"]) == 45)

        cfg.define_split("tile_y", cfg.axis(256), policy="power2", num_outputs=3)
        self.assertTrue(len(cfg.space_map["tile_y"]) == 45)

        cfg.define_split("tile_z", cfg.axis(256), policy="verbose", num_outputs=3)
        self.assertTrue(len(cfg.space_map["tile_z"]) == 45)

        cfg.define_split("tile_a", cfg.axis(224), policy="factors", num_outputs=3)
        self.assertTrue(len(cfg.space_map["tile_a"]) == 63)

        cfg.define_split("tile_b", cfg.axis(224), policy="power2", num_outputs=3)
        self.assertTrue(len(cfg.space_map["tile_b"]) == 36)

        cfg.define_split("tile_c", cfg.axis(224), policy="verbose", num_outputs=3)
        self.assertTrue(len(cfg.space_map["tile_c"]) == 84)


if __name__ == '__main__':
    unittest.main()
