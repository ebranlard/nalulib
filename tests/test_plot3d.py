import unittest
import numpy as np
from nalulib.plot3D_plot3D2exo import build_simple_quad_connectivity

class TestPlot3DQuadConnectivity(unittest.TestCase):

    def test_quad_connectivity_with_loop(self):
        dims = (5, 2, 3)  # ni, nk, nj
        conn = build_simple_quad_connectivity(dims, loop=True)
        conn_ref = np.array([
            [ 0,  4,  5,  1],
            [ 1,  5,  6,  2],
            [ 2,  6,  7,  3],
            [ 3,  7,  4,  0],
            [ 4,  8,  9,  5],
            [ 5,  9, 10,  6],
            [ 6, 10, 11,  7],
            [ 7, 11,  8,  4]
        ])
        self.assertEqual(conn.shape, (8, 4))
        np.testing.assert_array_equal(conn, conn_ref)

    def test_quad_connectivity_without_loop(self):
        dims = (5, 2, 3)  # ni, nk, nj
        conn = build_simple_quad_connectivity(dims, loop=False)
        conn_ref = np.array([
            [ 0,  5,  6,  1],
            [ 1,  6,  7,  2],
            [ 2,  7,  8,  3],
            [ 3,  8,  9,  4],
            [ 5, 10, 11,  6],
            [ 6, 11, 12,  7],
            [ 7, 12, 13,  8],
            [ 8, 13, 14,  9]
        ])
        self.assertEqual(conn.shape, (8, 4))
        np.testing.assert_array_equal(conn, conn_ref)

if __name__ == "__main__":
    unittest.main()