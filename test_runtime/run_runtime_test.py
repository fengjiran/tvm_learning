import unittest
import test_lib_load
import test_global_func
import test_packed_func
import test_ndarray


def create_suite():
    suite = unittest.TestSuite()
    loader = unittest.TestLoader()
    suite.addTest(loader.loadTestsFromModule(test_lib_load))
    suite.addTest(loader.loadTestsFromModule(test_global_func))
    suite.addTest(loader.loadTestsFromModule(test_packed_func))
    suite.addTest(loader.loadTestsFromModule(test_ndarray))
    return suite


if __name__ == '__main__':
    runner = unittest.TextTestRunner()
    runner.run(create_suite())
