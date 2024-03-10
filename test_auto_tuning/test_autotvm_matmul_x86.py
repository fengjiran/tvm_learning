import sys
import logging
import unittest
import numpy as np
import tvm
from tvm import te
from tvm import autotvm


# x86 matmul autotvm template
@autotvm.template("matmul_x86")
def matmul_x86(M, N, K, dtype):
    A = te.placeholder((M, K), name="A", dtype=dtype)
    B = te.placeholder((K, N), name="B", dtype=dtype)

    k = te.reduce_axis((0, K), name="k")
    C = te.compute((M, N), lambda i, j: te.sum(A[i, k] * B[k, j], axis=k), name="C")
    s = te.create_schedule(C.op)

    # schedule
    y, x = s[C].op.axis
    k = s[C].op.reduce_axis[0]

    ##### define space begin #####
    cfg = autotvm.get_config()
    cfg.define_split("tile_y", y, num_outputs=2)
    cfg.define_split("tile_x", x, num_outputs=2)
    ##### define space end #####

    # schedule according to config
    yo, yi = cfg["tile_y"].apply(s, C, y)
    xo, xi = cfg["tile_x"].apply(s, C, x)

    s[C].reorder(yo, xo, k, yi, xi)

    return s, [A, B, C]


class TestAutoTVM(unittest.TestCase):
    def test_matmul_x86(self):
        M, N, K, dtype = 512, 512, 512, "float32"
        task = autotvm.task.create("matmul_x86", args=(M, N, K, dtype), target="llvm")
        print(task.config_space)

        # logging config (for printing tuning log to the screen)
        logging.getLogger("autotvm").setLevel(logging.DEBUG)
        logging.getLogger("autotvm").addHandler(logging.StreamHandler(sys.stdout))

        measure_option = autotvm.measure_option(builder="local", runner=autotvm.LocalRunner(number=5))

        # Begin tuning with RandomTuner, log records to file `matmul.log`
        # You can use alternatives like XGBTuner.
        tuner = autotvm.tuner.RandomTuner(task)
        tuner.tune(
            n_trial=10,
            measure_option=measure_option,
            callbacks=[autotvm.callback.log_to_file("matmul.log")],
        )

        # apply history best from log file
        with autotvm.apply_history_best("matmul.log"):
            with tvm.target.Target("llvm"):
                s, arg_bufs = matmul_x86(M, N, K, dtype)
                func = tvm.build(s, arg_bufs)

        # check correctness
        a_np = np.random.uniform(size=(M, K)).astype(np.float32)
        b_np = np.random.uniform(size=(K, N)).astype(np.float32)
        c_np = a_np.dot(b_np)

        c_tvm = tvm.nd.empty(c_np.shape)
        func(tvm.nd.array(a_np), tvm.nd.array(b_np), c_tvm)

        np.testing.assert_allclose(c_np, c_tvm.numpy(), rtol=1e-4)


if __name__ == '__main__':
    unittest.main()
