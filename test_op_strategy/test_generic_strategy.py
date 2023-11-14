import os
import unittest
import numpy as np
import torch
from torch import nn
import tvm
from tvm import te
from tvm import relay
from tvm import topi
from tvm.relay.testing import run_infer_type

# os.environ['TVM_NUM_THREADS'] = str(1)


class TestGenericStrategy(unittest.TestCase):
    def test_relu_schedule(self):
        dshape = (10, 3, 256, 256)
        dtype = "float32"
        target = "llvm"
        tvm_dev = tvm.device(target, 0)

        # get schedule
        A = te.placeholder(dshape, name="A", dtype=dtype)
        T = topi.nn.relu(A)

        if target == "llvm":
            with tvm.target.Target(target):
                s = topi.x86.schedule_injective(T)
            torch_dev = torch.device("cpu")
        elif target == "cuda":
            with tvm.target.Target(target):
                s = topi.cuda.schedule_injective(T)
            torch_dev = torch.device("cuda:0")
        else:
            raise ValueError("No target of {}".format(target))

        with tvm.transform.PassContext(3):
            func = tvm.build(s, [A, T], target=target, name="relu")

        # get test data
        inps = np.random.rand(*dshape).astype(dtype)
        a = tvm.nd.array(inps, tvm_dev)
        t = tvm.nd.array(np.zeros(dshape, dtype=dtype), tvm_dev)
        func(a, t)

        # verify correctness
        ans = nn.ReLU(True)(torch.from_numpy(inps).to(torch_dev)).to("cpu").numpy()
        np.testing.assert_allclose(t.numpy(), ans)

        # evaluate run time
        evaluator = func.time_evaluator(func.entry_name, tvm_dev, number=50)
        tvm_time = evaluator(a, t).mean
        GFLOPS = np.prod(dshape) * 1e-9
        print("\nTVM without tune time: {}s".format(tvm_time))
        print("TVM without tune GFLOPS: {}".format(GFLOPS / tvm_time))
        print("done")

    def test_matmul_strategy(self):
        m = 1024
        n = 1024
        k = 1024
        bn = 32
        dtype = "float32"
        target = "llvm -mcpu=core-avx2"
        # target = "llvm"
        tvm_dev = tvm.device(target, 0)
        A = te.placeholder((m, k), name="A", dtype=dtype)
        B = te.placeholder((k, n), name="B", dtype=dtype)
        red_k = te.reduce_axis((0, k), name="k")
        C = te.compute((m, n), lambda i, j: te.sum(A[i, red_k] * B[red_k, j], axis=red_k), name="C")
        m_axis = C.op.axis[0]
        n_axis = C.op.axis[1]
        k_axis = C.op.reduce_axis[0]

        # packed B
        PackedB = te.compute((n / bn, k, bn), lambda bigN, k_, littleN: B[k_, bigN * bn + littleN], name="PackedB")
        CC = te.compute((m, n),
                        lambda i, j: te.sum(
                            A[i, red_k] * PackedB[j // bn, red_k, tvm.tir.indexmod(j, tvm.tir.IntImm("int32", bn))],
                            axis=red_k),
                        name="CC")

        def get_default_schedule():
            return te.create_schedule(C.op)

        def get_schedule_with_reorder_kmn():
            sch = te.create_schedule(C.op)
            sch[C].reorder(k_axis, m_axis, n_axis)
            return sch

        def get_schedule_with_redorder_mkn():
            sch = te.create_schedule(C.op)
            sch[C].reorder(m_axis, k_axis, n_axis)
            return sch

        def get_schedule_with_tile():
            sch = te.create_schedule(C.op)
            mo, no, mi, ni = sch[C].tile(m_axis, n_axis, bn, bn)
            ko, ki = sch[C].split(k_axis, 4)
            # sch[C].reorder(mo, no, ko, mi, ki, ni)
            sch[C].reorder(mo, ko, no, mi, ki, ni)
            return sch

        def get_schedule_with_tile_vectorize():
            sch = te.create_schedule(C.op)
            mo, no, mi, ni = sch[C].tile(m_axis, n_axis, bn, bn)
            ko, ki = sch[C].split(k_axis, 4)
            sch[C].reorder(mo, ko, no, mi, ki, ni)
            sch[C].vectorize(ni)
            return sch

        def get_schedule_with_cache_write():
            sch = te.create_schedule(C.op)
            # allocate write cache
            CC = sch.cache_write(C, "local")
            mo, no, mi, ni = sch[C].tile(m_axis, n_axis, bn, bn)
            # cache write is computed at no axis
            sch[CC].compute_at(sch[C], no)
            mc, nc = CC.op.axis
            kaxis = CC.op.reduce_axis[0]
            ko, ki = sch[CC].split(kaxis, 4)
            sch[CC].reorder(ko, mc, ki, nc)
            sch[C].parallel(mo)
            sch[CC].vectorize(nc)
            sch[CC].unroll(ki)
            return sch

        def get_schedule_with_packing():
            s = te.create_schedule(CC.op)
            mo, no, mi, ni = s[CC].tile(CC.op.axis[0], CC.op.axis[1], bn, bn)
            ko, ki = s[CC].split(s[CC].op.reduce_axis[0], 4)
            s[CC].reorder(mo, no, ko, mi, ki, ni)
            s[CC].vectorize(ni)
            # bigN, _, littleN = s[PackedB].op.axis
            # s[PackedB].vectorize(littleN)
            # s[PackedB].parallel(bigN)
            return s

        def get_schedule_with_packing_cache_write():
            s = te.create_schedule(CC.op)
            # allocate write cache
            CCC = s.cache_write(CC, "local")
            mo, no, mi, ni = s[CC].tile(CC.op.axis[0], CC.op.axis[1], bn, bn)

            s[CCC].compute_at(s[CC], no)
            mc, nc = CCC.op.axis
            kaxis = CCC.op.reduce_axis[0]
            ko, ki = s[CCC].split(kaxis, 4)
            s[CCC].reorder(ko, mc, ki, nc)
            s[CCC].vectorize(nc)
            s[CC].parallel(mo)
            return s

        # sch = get_default_schedule()
        # sch = get_schedule_with_reorder_kmn()
        # sch = get_schedule_with_redorder_mkn()
        # sch = get_schedule_with_tile()
        # sch = get_schedule_with_tile_vectorize()
        sch = get_schedule_with_cache_write()
        # sch = get_schedule_with_packing()
        # sch = get_schedule_with_packing_cache_write()
        with tvm.transform.PassContext(3):
            func = tvm.build(sch, [A, B, C], target=target, name="matmul")

        # get test data
        a = tvm.nd.array(np.random.rand(m, k).astype(dtype), tvm_dev)
        b = tvm.nd.array(np.random.rand(k, n).astype(dtype), tvm_dev)
        c = tvm.nd.array(np.zeros((m, n), dtype=dtype), tvm_dev)
        ans = np.dot(a.numpy(), b.numpy())
        func(a, b, c)
        np.testing.assert_allclose(c.numpy(), ans, rtol=1e-5)

        # evaluate run time and gflops
        evaluator = func.time_evaluator(func.entry_name, tvm_dev, number=10)
        tvm_time = evaluator(a, b, c).mean
        GFLOPS = 2 * m * n * k * 1e-9
        print("\nTVM without tune time: {}s".format(tvm_time))
        print("TVM without tune GFLOPS: {}".format(GFLOPS / tvm_time))
        print("done")

    def test_relu_strategy(self):
        target = tvm.target.Target("llvm")
        dshape = (10, 3, 256, 256)
        op = relay.op.get("nn.relu")
        data = relay.var("data", shape=dshape)
        out = relay.nn.relu(data)
        out = run_infer_type(out)
        impl = relay.backend.te_compiler.get_valid_implementations(
            op,
            out.attrs,
            [te.placeholder(dshape)],
            out.checked_type,
            target
        )
        return impl

    def test_softmax_strategy(self):
        target = tvm.target.Target("llvm")
        shape = (10, 100)
        op = relay.op.get("nn.softmax")
        data = relay.var("data", shape=shape)
        out = relay.nn.softmax(data)
        out = run_infer_type(out)
        impl = relay.backend.te_compiler.get_valid_implementations(
            op,
            out.attrs,
            [te.placeholder(shape)],
            out.checked_type,
            target
        )
        return impl


if __name__ == '__main__':
    unittest.main()
