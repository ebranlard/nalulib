import unittest
import numpy as np
from nalulib.airfoillib import *
from nalulib.airfoil_shapes_naca import naca_shape
from nalulib.curves import curve_check_superset

scriptDir = os.path.dirname(os.path.abspath(__file__))

class TestAirfoilLib(unittest.TestCase):

    def test_diamond_shape(self):
        # Generate diamond shape
        x, y = airfoil_get_xy('diamond')
        self.assertEqual(len(x), 5)
        self.assertTrue(np.allclose(x[0], x[-1]))
        self.assertTrue(np.allclose(y[0], y[-1]))

        # Split surfaces
        IUpper, ILower, ITE, iLE = airfoil_split_surfaces(x, y)
        self.assertTrue(len(IUpper) > 0)
        self.assertTrue(len(ILower) > 0)
        self.assertTrue(len(ITE) > 0)

    def test_resample_refine(self):
        def test(*args, factor_surf=3, **kwargs):
            x, y = airfoil_get_xy(*args, **kwargs)
            x, y = normalize_airfoil_coords(x, y)
            IUpper, ILower, ITE, iLE = airfoil_split_surfaces(x, y)
            # Refine grid
            x_new, y_new = resample_airfoil_refine(x, y, IUpper, ILower, ITE, factor_surf=factor_surf, factor_te=2)
            # All original points should be present in the refined grid (within tolerance)
            #ax = plot_normalized(x, y, label='Original', sty='ro', simple=True)
            #ax = plot_normalized(x_new, y_new, label='Refined', sty='k.', simple=True, ax=ax)
            curve_check_superset(x, y, x_new, y_new, raiseError=True, verbose=True) #, reltol=0.01)
            #plt.show()

        test('diamond')
        test('naca0012')
        test(os.path.join(scriptDir, '../data/airfoils/ffa_w3_211_coords.pwise'))

    def test_normalized_airfoil_coords(self):
        def test(*args,  **kwargs):
            # Generate a NACA airfoil
            x, y = airfoil_get_xy(*args, **kwargs)
            x_new, y_new = normalize_airfoil_coords(x, y)
            airfoil_is_normalized(x_new, y_new, raiseError=True)
        
        test('diamond')
        test('naca0012')
        test(os.path.join(scriptDir, '../data/airfoils/ffa_w3_211_coords.pwise'))

    def test_trailing_edge_angle(self):
        def test(*args, **kwargs):
            x, y = airfoil_get_xy(*args, **kwargs)
            x_new, y_new = normalize_airfoil_coords(x, y)
            angle_deg, result = airfoil_trailing_edge_angle(x_new, y_new, plot=False)
            #self.assertIsInstance(angle_deg, float)
            #self.assertGreaterEqual(angle_deg, 0.0)
            #self.assertLessEqual(angle_deg, 180.0)
            return angle_deg

        angle = test('diamond')
        angle = test('naca0012', sharp=False)
        angle = test('naca0012', sharp=True)
        angle = test(os.path.join(scriptDir, '../data/airfoils/ffa_w3_211_coords.pwise'))
        #plt.show()

    def test_leading_edge_radius(self):
        # NOTE: LEADING EDGE RADIUS IS NOT FINISHED
        def test(*args, **kwargs):
            x, y = airfoil_get_xy(*args, **kwargs)
            x_new, y_new = normalize_airfoil_coords(x, y)
            r, result = airfoil_leading_edge_radius(x_new, y_new, plot=False)
            #self.assertGreater(r, 0.0)
            return r

        r = test('diamond')
        r = test('naca0012', sharp=False)
        r = test('naca0012', sharp=True)
        r = test(os.path.join(scriptDir, '../data/airfoils/S809.csv'))
        r = test(os.path.join(scriptDir, '../data/airfoils/ffa_w3_211_coords.pwise'))
        #plt.show()


if __name__ == '__main__':
    unittest.main()