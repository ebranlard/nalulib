# coding: utf-8
import os
import numpy as np

from nalulib.exodusii.file import ExodusIIFile
import matplotlib.pyplot as plt

def plt3d2exo(filename, fileout=None, suffix=''):

    dims = np.loadtxt(filename,skiprows=1,max_rows=1,dtype=int)
    a = np.loadtxt(filename,skiprows=2)
    xyz = a.reshape((3, dims[0]*dims[1]*dims[2] )).transpose()
    nx = dims[0]-1
    ny = dims[2]
    print(dims)
    xyz_2d = np.zeros((nx * ny, 2))
    for j in range(ny):
        for i in range(nx):
            idx = i + j * dims[0] * dims[1]
            xyz_2d[i + j * nx] = np.r_[xyz[idx,0], xyz[idx,1]]

    #print('>>> coords2d\n', xyz_2d)

    conn = np.zeros((nx*(ny-1),4),dtype=int)
    for j in range(ny-1):
        for i in range(nx-1):
            conn[i + j * nx] = np.array([i + j * nx, i + (j+1) * nx, i+1 + (j+1) * nx, i+1 + j * nx ])
        conn[(nx-1) + j * nx] = np.array([(nx-1) + j * nx, nx-1 + (j+1) * nx, (j+1) * nx, j * nx])
    conn += 1

    # Get inflow and outflow elements on the farfield boundary
    elem_farfield = np.linspace(conn.shape[0]-nx+1, conn.shape[0], nx, dtype=int)
    coordsx_farfield = np.r_[0.5 * (xyz_2d[-nx:-1, 0] + xyz_2d[-nx+1:,0]), 0.5 *(xyz_2d[-1, 0] + xyz_2d[-nx, 0])]
    #for i, x in zip(elem_farfield, coordsx_farfield):
    #    print(f"Element {i}: x = {x}")
    elem_inflow = elem_farfield[np.where(coordsx_farfield < 0)]
    elem_outflow = elem_farfield[np.where(coordsx_farfield >= 0)]

    ndim = 2
    type = 'QUAD'
    node_count = nx * ny
    cell_count = nx * (ny-1)
    block_names = ['fluid']
    sideset_names = ['airfoil','inflow', 'outflow']
    sideset_cells = [np.linspace(0,nx-1,nx,dtype=int)+1, 
                     elem_inflow,
                     elem_outflow]  # +1 for 1-based indexing in Exodus
    sideset_sides = [4 * np.ones_like(sideset_cells[0], dtype=int), 
                     2 * np.ones_like(sideset_cells[1], dtype=int),
                     2 * np.ones_like(sideset_cells[2], dtype=int)]  # All sides are on the airfoil

    if fileout is None:
        base, ext = os.path.splitext(filename)
        fileout = base+suffix+'.exo'
    with ExodusIIFile(fileout, mode="w") as exof:
        exof.put_init('airfoil', ndim, node_count, cell_count,
                      len(block_names), 0, 3) #len(sideset_names))
        exof.put_coord(xyz_2d[:,0], xyz_2d[:,1], np.zeros_like(xyz_2d[:,0]))
        exof.put_coord_names(["X", "Y"])
        exof.put_element_block(1, 'QUAD', cell_count, 4)
        exof.put_element_block_name(1, 'Flow-QUAD')
        exof.put_element_conn(1, conn)
        # Side sets
        for i in range(len(sideset_names)):
            exof.put_side_set_param(i+1, len(sideset_cells[i]))
            exof.put_side_set_name(i+1, sideset_names[i])
            exof.put_side_set_sides(i+1, sideset_cells[i], sideset_sides[i])
    print('Written', fileout)

if __name__=="__main__":
    #write_airfoil2d('eta_30_inc_0.xyz')
    #write_airfoil2d('../examples/diamond_n2.fmt')
    write_airfoil2d('../prod/S809_volume.fmt', '_gan')
