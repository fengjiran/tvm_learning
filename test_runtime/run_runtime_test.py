import unittest
import test_lib_load

# Create a test suite
suite = unittest.TestSuite()

# Create a loader obj
loader = unittest.TestLoader()
suite.addTest(loader.loadTestsFromModule(test_lib_load))
runner = unittest.TextTestRunner()
runner.run(suite)
