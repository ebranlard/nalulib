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
        for method in ['xmax', 'angle']:
            IUpper, ILower, ITE, iLE = airfoil_split_surfaces(x, y, method=method)
            np.testing.assert_array_equal(IUpper, [0, 1, 2])
            np.testing.assert_array_equal(ILower, [2,3,4])
            np.testing.assert_array_equal(ITE, [4,0])

    def test_resample_refine(self):
        def test(*args, factor_surf=3, **kwargs):
            x, y = airfoil_get_xy(*args, **kwargs)
            x, y = standardize_airfoil_coords(x, y)
            IUpper, ILower, ITE, iLE = airfoil_split_surfaces(x, y)
            # Refine grid
            x_new, y_new = resample_airfoil_refine(x, y, IUpper, ILower, ITE, factor_surf=factor_surf, factor_te=2)
            # All original points should be present in the refined grid (within tolerance)
            #ax = plot_standardized(x, y, label='Original', sty='ro', simple=True)
            #ax = plot_standardized(x_new, y_new, label='Refined', sty='k.', simple=True, ax=ax)
            curve_check_superset(x, y, x_new, y_new, raiseError=True, verbose=True) #, reltol=0.01)
            #plt.show()

        test('diamond')
        test('naca0012')
        test(os.path.join(scriptDir, '../data/airfoils/ffa_w3_211_coords.pwise'))

    def test_normalized_airfoil_coords(self):
        def test(*args,  **kwargs):
            # Generate a NACA airfoil
            x, y = airfoil_get_xy(*args, **kwargs)
            x_new, y_new = standardize_airfoil_coords(x, y)
            airfoil_is_standardized(x_new, y_new, raiseError=True)
        
        test('diamond')
        test('naca0012')
        test(os.path.join(scriptDir, '../data/airfoils/ffa_w3_211_coords.pwise'))

    def test_trailing_edge_angle(self):
        def test(*args, expected=0, **kwargs):
            x, y = airfoil_get_xy(*args, **kwargs)
            x_new, y_new = standardize_airfoil_coords(x, y)
            angle_deg, result = airfoil_trailing_edge_angle(x_new, y_new, plot=False)
            #self.assertIsInstance(angle_deg, float)
            #self.assertGreaterEqual(angle_deg, 0.0)
            np.testing.assert_almost_equal(angle_deg, expected, decimal=2)
            return angle_deg

        angle = test('diamond', expected=90)
        angle = test('naca0012', sharp=False, expected=15.91)
        angle = test('naca0012', sharp=True, expected=16.47)
        angle = test(os.path.join(scriptDir, '../data/airfoils/ffa_w3_211_coords.pwise'), expected=3.49)
        #plt.show()

    def test_leading_edge_radius(self):
        # NOTE: LEADING EDGE RADIUS IS NOT FINISHED
        def test(*args, expected=0, **kwargs):
            x, y = airfoil_get_xy(*args, **kwargs)
            x_new, y_new = standardize_airfoil_coords(x, y)
            r, result = airfoil_leading_edge_radius(x_new, y_new, plot=False)
            #self.assertAlmostEqual(r, expected)
            return r

        test('diamond', expected=0)
        test('naca0012', sharp=False, expected=0)
        test('naca0012', sharp=True, expected=0)
        test(os.path.join(scriptDir, '../data/airfoils/S809.csv'), expected=0)
        test(os.path.join(scriptDir, '../data/airfoils/ffa_w3_211_coords.pwise'), expected=0)
        #plt.show()


    def test_ITE(self):
        def test(*args, method='xmax', expected=None, **kwargs):
            x, y = airfoil_get_xy(*args, **kwargs)
            x_new, y_new = standardize_airfoil_coords(x, y)
            _,_,ITE,_ = airfoil_split_surfaces(x_new, y_new, method=method)
            #plot_airfoil(x_new,y_new)
            np.testing.assert_array_equal(ITE, expected)

        for method in ['xmax', 'angle']:
            test('diamond',  expected=[4, 0], method=method); 
            test('naca0012', expected=[300, 301, 0], method=method, sharp=False)
            test('naca0012', expected=[300, 0], method=method, sharp=True)
            test(os.path.join(scriptDir, '../data/airfoils/S809.csv'), expected=[65, 0], method=method)
            test(os.path.join(scriptDir, '../data/airfoils/ffa_w3_211_coords.pwise'), expected=[199, 200, 201, 0], method=method)
            test(os.path.join(scriptDir, '../data/airfoils/ffa_w3_211_coords.csv'), expected=[199, 200, 0], method=method)

        method = 'angle'
        test(os.path.join(scriptDir, '../data/airfoils/blunt_not_straight.csv'), expected=np.concatenate((np.arange(499,524), [0])), method=method)


    def test_problematic1(self):
        # File fx79w470a.dat has clearly a wrong point on line 3
        x, y, d = read_airfoil(os.path.join(scriptDir, '../data/airfoils/tests/fx79w470a.dat'), format='csv')
        x_new, y_new = standardize_airfoil_coords(x, y)
        # The following line should raise an exception because the data is wrong
        with self.assertRaises(Exception):
            IUpper, ILower, ITE, iLE = airfoil_split_surfaces(x_new, y_new)
        #plot_airfoil(x_new,y_new, simple=True)


    def test_problematic2(self):
        # File du91-w2-225_l40.csv, display a sharp trailing edge but with large angles... 
        # Most likely the data is wrong, but we still assume this has to be a sharp TE
        x, y, d = read_airfoil(os.path.join(scriptDir, '../data/airfoils/tests/du91-w2-225_nalu_l40.csv'), format='csv')
        x_new, y_new = standardize_airfoil_coords(x, y)
        TE_type = airfoil_TE_type(x, y)
        _, _, ITE, _ = airfoil_split_surfaces(x_new, y_new)
        #plot_airfoil(x_new,y_new)
        self.assertEqual(TE_type, 'sharp')
        np.testing.assert_equal(ITE, [40, 0])


    def test_te_type(self):
        def test(*args, method='xmax', expected='sharp', **kwargs):
            x, y = airfoil_get_xy(*args, **kwargs)
            x_new, y_new = standardize_airfoil_coords(x, y)
            TEtype = airfoil_TE_type(x_new, y_new, method=method)
            self.assertEqual(TEtype, expected)
            return TEtype

        for method in ['xmax', 'angle']:
            test('diamond',  method=method, expected='sharp'); 
            test('naca0012', method=method, expected='blunt', sharp=False)
            test('naca0012', method=method, expected='sharp', sharp=True)
            test(os.path.join(scriptDir, '../data/airfoils/S809.csv'), method=method, expected='sharp')
            test(os.path.join(scriptDir, '../data/airfoils/ffa_w3_211_coords.pwise'), method=method, expected='blunt')
        method = 'angle'
        test(os.path.join(scriptDir, '../data/airfoils/blunt_not_straight.csv'), expected='blunt', method=method)


if __name__ == '__main__':
    #TestAirfoilLib().test_ITE()
    #TestAirfoilLib().test_problematic1()
    #TestAirfoilLib().test_problematic2()
    unittest.main()
    #plt.show()
