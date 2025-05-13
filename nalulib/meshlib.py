import numpy as np
import os
import meshio

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

def save_mesh(points, cells, filename="airfoil_omesh.msh"):
    mesh = meshio.Mesh(
        points=np.hstack([points, np.zeros((points.shape[0], 1))]),  # add z=0
        cells=[("quad", cells)]
    )
    meshio.write(filename, mesh, file_format="gmsh", binary=False)
    print(f"Mesh successfully written to '{filename}' in ASCII format.")

def open_mesh_in_gmsh(filename="airfoil_omesh.msh"):
    os.system(f"gmsh {filename}")
    print(f"Opening mesh in GMSH: {filename}")