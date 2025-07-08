import unittest
import os
import numpy as np
from nalulib.airfoil_shapes_io import read_airfoil, write_airfoil, convert_airfoil

scriptDir = os.path.dirname(os.path.abspath(__file__))

def cleanup_files(file_list):
    """Delete files in file_list if they exist."""
    for f in file_list:
        if os.path.exists(f):
            os.remove(f)

class TestAirfoilShapesIO(unittest.TestCase):
        
    # test reading of airfoil shapes
    def test_read_airfoil(self):
        # Test reading a CSV file
        x, y, d = read_airfoil(os.path.join(scriptDir, '../examples/airfoils/ffa_w3_211_coords.csv'), format='csv')
        np.testing.assert_almost_equal(x[0], 1.0)
        np.testing.assert_almost_equal(x[-1], 1.0)
        np.testing.assert_almost_equal(y[-1], 2.85e-4)

        ## Test reading a Plot3D file
        x, y, d = read_airfoil(os.path.join(scriptDir, '../examples/airfoils/naca0012_sharp.fmt'), format='plot3d')
        np.testing.assert_almost_equal(x[0], 1.0)
        np.testing.assert_almost_equal(x[-1], 1.0)
        np.testing.assert_almost_equal(y[-1], 0.0)

        ## Test reading a Pointwise file
        x, y, d = read_airfoil(os.path.join(scriptDir, '../examples/airfoils/ffa_w3_211_coords.pwise'), format='pointwise')
        np.testing.assert_almost_equal(x[-1], 1.0)
        np.testing.assert_almost_equal(x[-2], 1.0)
        np.testing.assert_almost_equal(x[-3], 1.0)
        np.testing.assert_array_almost_equal(y[-3:], [9.4000000e-04, 2.8500000e-04,-3.7000000e-04])

    def test_convert_airfoil_csv_to_plot3d_and_back(self):
        input_file = os.path.join(scriptDir, '../examples/airfoils/ffa_w3_211_coords.csv')
        temp_fmt = os.path.join(scriptDir, '_test_convert.fmt')
        temp_csv = os.path.join(scriptDir, '_test_convert.csv')

        # Clean up before
        cleanup_files([temp_fmt, temp_csv])

        # Convert CSV -> Plot3D
        convert_airfoil(input_file, output_file=temp_fmt, out_format='plot3d')
        self.assertTrue(os.path.exists(temp_fmt))
        # Read back and compare
        x0, y0, _ = read_airfoil(input_file, format='csv')
        x1, y1, _ = read_airfoil(temp_fmt, format='plot3d')
        np.testing.assert_allclose(x0, x1, rtol=1e-7, atol=1e-10)
        np.testing.assert_allclose(y0, y1, rtol=1e-7, atol=1e-10)

        # Convert Plot3D -> CSV
        convert_airfoil(temp_fmt, output_file=temp_csv, out_format='csv')
        self.assertTrue(os.path.exists(temp_csv))
        x2, y2, _ = read_airfoil(temp_csv, format='csv')
        np.testing.assert_allclose(x1, x2, rtol=1e-7, atol=1e-10)
        np.testing.assert_allclose(y1, y2, rtol=1e-7, atol=1e-10)

        # Clean up after
        cleanup_files([temp_fmt, temp_csv])

    def test_convert_airfoil_csv_to_pointwise_and_back(self):
        input_file = os.path.join(scriptDir, '../examples/airfoils/ffa_w3_211_coords.csv')
        temp_pwise = os.path.join(scriptDir, '_test_convert.pwise')
        temp_csv = os.path.join(scriptDir, '_test_convert2.csv')

        # Clean up before
        cleanup_files([temp_pwise, temp_csv])

        # Convert CSV -> Pointwise
        convert_airfoil(input_file, output_file=temp_pwise, out_format='pointwise')
        self.assertTrue(os.path.exists(temp_pwise))
        # Read back and compare
        x0, y0, _ = read_airfoil(input_file, format='csv')
        x1, y1, _ = read_airfoil(temp_pwise, format='pointwise')
        print('\n[TODO]Pointwise conversion not ready for unittest\n')
        # print length
        #print(f"Length of x0: {len(x0)}, y0: {len(y0)}")
        #print(f"Length of x1: {len(x1)}, y1: {len(y1)}")
        #print(f"x0: {x0}, y0: {y0}")
        #print(f"x1: {x1}, y1: {y1}")
        #np.testing.assert_allclose(x0, x1, rtol=1e-7, atol=1e-10)
        #np.testing.assert_allclose(y0, y1, rtol=1e-7, atol=1e-10)

        # Convert Pointwise -> CSV
        #convert_airfoil(temp_pwise, output_file=temp_csv, out_format='csv')
        #self.assertTrue(os.path.exists(temp_csv))
        #x2, y2, _ = read_airfoil(temp_csv, format='csv')
        #np.testing.assert_allclose(x1, x2, rtol=1e-7, atol=1e-10)
        #np.testing.assert_allclose(y1, y2, rtol=1e-7, atol=1e-10)

        # Clean up after
        cleanup_files([temp_pwise, temp_csv])

if __name__ == "__main__":
    unittest.main()