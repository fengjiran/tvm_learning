import unittest
from tvm._ffi import libinfo


def find_tvm_lib_path(name=None, search_path=None, optional=False):
    lib_path = libinfo.find_lib_path(name, search_path, optional)
    return lib_path


class TestTVMLibLoad(unittest.TestCase):
    def test_load_tvm_lib(self):
        lib_path = find_tvm_lib_path()
        self.assertNotEqual(len(lib_path), 0)


# https://www.cnblogs.com/miki-peng/p/12501341.html
if __name__ == '__main__':
    unittest.main()
