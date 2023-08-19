import tvm
import tvm.testing
import numpy as np
import unittest


class TestNDArray(unittest.TestCase):
    @tvm.testing.uses_gpu
    def test_nd_create(self):
        for target, dev in tvm.testing.enabled_targets():
            for dtype in ["uint8", "int8", "uint16", "int16", "uint32", "int32", "float32"]:
                x = np.random.randint(0, 10, size=(3, 4))
                x = np.array(x, dtype=dtype)
                y = tvm.nd.array(x, device=dev)
                z = y.copyto(dev)
                self.assertEqual(y.dtype, x.dtype)
                self.assertEqual(y.shape, x.shape)
                self.assertIsInstance(y, tvm.nd.NDArray)
                np.testing.assert_equal(x, y.numpy())
                np.testing.assert_equal(x, z.numpy())
            dev.sync()


if __name__ == '__main__':
    unittest.main()
