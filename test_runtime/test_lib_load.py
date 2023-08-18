import unittest
from tvm._ffi import libinfo


class TestTVMLibLoad(unittest.TestCase):
    def test_find_tvm_lib(self):
        lib_path = libinfo.find_lib_path()
        self.assertNotEqual(len(lib_path), 0, 'tvm libs not found.')
        print('List tvm libs path:')
        for path in lib_path:
            print(path)
        print('\n')

    def test_find_include_path(self):
        include_path = libinfo.find_include_path()
        self.assertNotEqual(len(include_path), 0, 'tvm header files for C compilation not found.')
        print('List tvm include path:')
        for path in include_path:
            print(path)
        print('\n')


# https://www.cnblogs.com/miki-peng/p/12501341.html
if __name__ == '__main__':
    unittest.main()
