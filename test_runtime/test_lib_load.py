import unittest
from tvm._ffi import libinfo


class TestTVMLibLoad(unittest.TestCase):
    def test_find_tvm_lib(self):
        lib_path = libinfo.find_lib_path()
        self.assertNotEqual(len(lib_path), 0, 'Not found tvm libs.')
        for path in lib_path:
            print(path)


# https://www.cnblogs.com/miki-peng/p/12501341.html
if __name__ == '__main__':
    unittest.main()