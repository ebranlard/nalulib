import unittest
import numpy as np
from nalulib.exodus_cuboid import exo_cuboid

class TestExodusCuboid(unittest.TestCase):
    def test_cuboid_legacy(self):
        ls = (2.0, 3.0, 4.0)
        ns = (4, 3, 3)
        coords, conn, sidesets = exo_cuboid(filename=None, ls=ls, ns=ns, center_origin=True, legacy=True)
        # Check nodes shape and a few node values
        self.assertEqual(coords.shape, (4*3*3, 3))
        np.testing.assert_allclose(coords[0], [ 1.0,  1.5, 0.0])
        np.testing.assert_allclose(coords[-1], [-1.0, -1.5, 4.0])
        # Check connectivity shape and a few values
        self.assertEqual(conn.shape, (3*2*2, 8))
        self.assertTrue(np.all(conn >= 1))
        np.testing.assert_allclose(conn[0],  [1, 2, 6, 5, 13, 14, 18, 17])
        np.testing.assert_allclose(conn[-1], [19, 20, 24, 23, 31, 32, 36, 35])
        # Check sidesets: shape, range, and a few IDs
        ssinfo = sidesets[6] # back_bg
        np.testing.assert_equal(ssinfo["name"], "back_bg")
        np.testing.assert_allclose(ssinfo["elements"], [4, 1, 5, 2, 6, 3])
        np.testing.assert_allclose(ssinfo["sides"][0], 5)
        ssinfo = sidesets[2] # outlet_bg
        np.testing.assert_equal(ssinfo["name"], "outlet_bg")
        np.testing.assert_allclose(ssinfo["elements"], [4, 10, 1, 7])
        np.testing.assert_allclose(ssinfo["sides"][0], 4)

    def test_cuboid_large_legacy(self):
        ls = (120.0, 120.0, 4.0)
        ns = (166, 166, 121)
        coords, conn, sidesets = exo_cuboid(filename=None, ls=ls, ns=ns, center_origin=True, legacy=True)
        # Check nodes shape and a few node values
        self.assertEqual(coords.shape, (166*166*121, 3))
        np.testing.assert_allclose(coords[0], [ 60.0,  60.0, 0.0])
        np.testing.assert_allclose(coords[-1], [-60.0, -60.0, 4.0])
        # Check connectivity shape and a few values
        self.assertEqual(conn.shape, (165*165*120, 8))
        np.testing.assert_allclose(conn[0],  [1, 2, 168, 167, 27557, 27558, 27724, 27723])
        np.testing.assert_allclose(conn[-1], [3306553, 3306554, 3306720, 3306719, 3334109, 3334110, 3334276, 3334275])
        self.assertTrue(np.all(conn >= 1))
        # Check sidesets: shape, range, and a few IDs
        ssinfo = sidesets[6] # back_bg
        np.testing.assert_equal(ssinfo["name"], "back_bg")
        np.testing.assert_allclose(ssinfo["elements"][:8], [27061, 26896, 26731, 26566, 26401, 26236, 26071, 25906])
        np.testing.assert_allclose(ssinfo["sides"][0], 5)
        ssinfo = sidesets[2] # outlet_bg
        np.testing.assert_equal(ssinfo["name"], "outlet_bg")
        np.testing.assert_allclose(ssinfo["elements"][:8], [27061,  54286,  81511, 108736, 135961, 163186, 190411, 217636])
        np.testing.assert_allclose(ssinfo["sides"][0], 4)
        ssinfo = sidesets[3] # top
        np.testing.assert_equal(ssinfo["name"], "top_bg")
        np.testing.assert_allclose(ssinfo["elements"][:8], [1,  27226,  54451,  81676, 108901, 136126, 163351, 190576])
        ssinfo = sidesets[5] # front
        np.testing.assert_equal(ssinfo["name"], "front_bg")
        np.testing.assert_allclose(ssinfo["elements"][:8], [3266836, 3266671, 3266506, 3266341, 3266176, 3266011, 3265846, 3265681])
        np.testing.assert_allclose(ssinfo["sides"][0], 6)


if __name__ == "__main__":
    unittest.main()