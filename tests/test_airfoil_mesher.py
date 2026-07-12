import unittest
import numpy as np
import os
import matplotlib.pyplot as plt

from nalulib.airfoil_mesher import mesh_airfoil
from nalulib.airfoillib import airfoil_get_xy

scriptDir=os.path.dirname(__file__)

class TestAirfoilMesher(unittest.TestCase):

    def test_mesh_airfoil_blunt(self):
        """Mesh a dummy airfoil."""
        # --- Blunt TE: two distinct TE points at x=1 -> blunt
        upper = np.array([[1.0, 0.01], [0.5, 0.05], [0.0, 0.0]])
        lower = np.array([[0.0, 0.0], [0.5, -0.05], [1.0, -0.01], [1.0, 0.01]])
        coords = np.vstack([upper, lower])

        n_te = 4
        n = 5
        plot=False
        arf = mesh_airfoil(coords, respline=False, n=n, method='linear', n_te=n_te, check=False, verbose=False, plot=plot)
        # ensure internal split computed
        arf._split_surfaces()
        self.assertEqual(len(arf._IUpper), n)
        self.assertEqual(len(arf._ILower), n)
        self.assertEqual(len(arf._ITE) - 1, n_te)
        self.assertEqual(arf._TE_TYPE, 'blunt')
        np.testing.assert_array_almost_equal(arf._x[arf._IUpper], np.linspace(0,1,n)[::-1], decimal=3)
        np.testing.assert_array_almost_equal(arf._x[arf._ILower], np.linspace(0,1,n)      , decimal=3)
        np.testing.assert_array_almost_equal(arf._y[arf._IUpper], np.array([0.01, 0.03, 0.05, 0.025, 0.])     , decimal=4)
        np.testing.assert_array_almost_equal(arf._y[arf._ILower], np.array([0.0, -0.025, -0.05, -0.03, -0.01]), decimal=4)
        np.testing.assert_array_almost_equal(arf._y[arf._ITE], np.array([-0.01, -0.0033, 0.0033, 0.01, 0.01]), decimal=4)
        np.testing.assert_array_almost_equal(arf._x[arf._ITE], np.array([1, 1, 1, 1, 1]), decimal=4)
        # arf.plot(title='blunt (debug)')

    def test_mesh_airfoil_blunt_temindist(self):
        """  """
        # --- Blunt TE: two distinct TE points at x=1 -> blunt
        fullpath = os.path.join(scriptDir, '../data/airfoils/tests/fb90_coords.csv')
        n = 10
        plot=False
        arf = mesh_airfoil(fullpath, respline=False, n=n, method='hyperbolic', method_te='min_dist',  check=False, verbose=False, plot=plot)
        # ensure internal split computed
        arf._split_surfaces()
        self.assertEqual(len(arf._IUpper), n)
        self.assertEqual(len(arf._ILower), n)
        self.assertEqual(len(arf._ITE) - 1, 6)
        self.assertEqual(arf._TE_TYPE, 'blunt')


    def test_mesh_airfoil_sharp(self):
        """Mesh a dummy airfoil."""
        # --- Sharp TE: single TE point at x=1 -> sharp
        upper = np.array([[1.0, 0.01], [0.5, 0.05], [0.0, 0.0]])
        lower = np.array([[0.0, 0.0], [0.5, -0.05], [1.0, 0.01]])  # ends at same TE point
        coords = np.vstack([upper, lower])
        n = 5
        plot=False
        arf = mesh_airfoil(coords, respline=False, n=n, method='linear', check=False, verbose=False, plot=plot)
        arf._split_surfaces()
        self.assertEqual(len(arf._IUpper), n)
        self.assertEqual(len(arf._ILower), n)
        self.assertEqual(len(arf._ITE), 2)
        self.assertEqual(arf._TE_TYPE, 'sharp')
        np.testing.assert_array_almost_equal(arf._x[arf._IUpper], np.linspace(0,1,n)[::-1], decimal=3)
        np.testing.assert_array_almost_equal(arf._x[arf._ILower], np.linspace(0,1,n)      , decimal=3)
        np.testing.assert_array_almost_equal(arf._y[arf._IUpper], np.array([0.01, 0.03, 0.05, 0.025, 0.])     , decimal=4)
        np.testing.assert_array_almost_equal(arf._y[arf._ILower], np.array([0.0, -0.025, -0.05, -0.02, 0.01]), decimal=4)


if __name__ == '__main__':
    #TestAirfoilMesher().test_mesh_airfoil_blunt_temindist()
    #unittest.main()
    plt.show()
