import numpy as np
import os

def create_quadrilateral_cells(layers, nNormal, nAirfoilPoints):
    cells = []
    for j in range(nNormal-1):
        for i in range(nAirfoilPoints-1):
            n0 = j * nAirfoilPoints + i
            n1 = n0 + 1
            n2 = n0 + nAirfoilPoints + 1
            n3 = n0 + nAirfoilPoints
            cells.append([n0, n1, n2, n3])

    # Wrap-around for last point to first (periodic at trailing edge)
    for j in range(nNormal-1):
        n0 = j * nAirfoilPoints + (nAirfoilPoints-1)
        n1 = j * nAirfoilPoints
        n2 = (j+1) * nAirfoilPoints
        n3 = (j+1) * nAirfoilPoints + (nAirfoilPoints-1)
        cells.append([n0, n1, n2, n3])

    return np.array(cells)

