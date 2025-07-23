import os
import unittest
import numpy as np
from nalulib.nalu_input import NALUInputFile

class TestNALUInputFile(unittest.TestCase):
    def test_nalu_input_properties(self):
        # Path to example input file
        example_file = os.path.join(os.path.dirname(__file__), '../examples/input.yaml')
        yml = NALUInputFile(example_file)

        expected_velocity = [75.0, 0.0, 0.0]
        expected_density = 1.2
        expected_viscosity = 9e-6

        # Test velocity
        velocity = yml.velocity
        np.testing.assert_allclose(velocity, expected_velocity, rtol=1e-12, err_msg=f"Velocity mismatch: {velocity} != {expected_velocity}")

        # Test density
        density = yml.density
        self.assertAlmostEqual(density, expected_density, places=12)

        # Test viscosity
        viscosity = yml.viscosity
        self.assertAlmostEqual(viscosity, expected_viscosity, places=12)

if __name__ == '__main__':
    unittest.main()
