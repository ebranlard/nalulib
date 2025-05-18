import numpy as np
import os

from nalulib.gmesh_gmesh2exodus import gmsh2exo


def save_mesh(points, cells, filename="airfoil_omesh.msh"):
    import meshio
    mesh = meshio.Mesh(
        points=np.hstack([points, np.zeros((points.shape[0], 1))]),  # add z=0
        cells=[("quad", cells)]
    )
    meshio.write(filename, mesh, file_format="gmsh", binary=False)
    print(f"Mesh successfully written to '{filename}' in ASCII format.")

def open_mesh_in_gmsh(filename="airfoil_omesh.msh"):
    os.system(f"gmsh {filename}")
    print(f"Opening mesh in GMSH: {filename}")

