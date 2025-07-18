import matplotlib.pyplot as plt
import numpy as np
import os
from nalulib.exodusii.file import ExodusIIFile
from nalulib.essentials import *
#from nalulib.gmesh import *
#from nalulib.meshlib import *
#from nalulib.gmesh_gmesh2exodus import gmesh_to_exo
from nalulib.exodus_core import set_omesh_inlet_outlet_ss, write_exodus
from nalulib.exodus_core import force_quad_positive_about_z
from nalulib.weio.plot3d_file import read_plot3d


def _plt3d_to_quads(input_file):
    # --- Read input
    dims = np.loadtxt(input_file, skiprows=1,max_rows=1,dtype=int)
    a = np.loadtxt(input_file, skiprows=2)
    xyz = a.reshape((3, dims[0]*dims[1]*dims[2] )).transpose()
    # --- Convert to 2D
    nx = dims[0]-1
    ny = dims[2]
    print('dims:', dims)
    xyz_2d = np.zeros((nx * ny, 2))
    for j in range(ny):
        for i in range(nx):
            idx = i + j * dims[0] * dims[1]
            xyz_2d[i + j * nx] = np.r_[xyz[idx,0], xyz[idx,1]]

    # --- Connectivity
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

    return xyz_2d, conn, sideset_names, sideset_cells, sideset_sides 


def build_simple_hex_connectivity(dims):
    """
    Build a simple hexahedral connectivity for a single block.
    Assumes periodic/looping in i, j, k directions.
    Returns:
        conn: (n_hex, 8) array of node indices
    """
    ni, nj, nk = dims
    conn = []
    for k in range(nk - 1):
        for j in range(nj - 1):
            for i in range(ni - 1):
                n0 = i     + j*ni     + k*ni*nj
                n1 = (i+1) + j*ni     + k*ni*nj
                n2 = (i+1) + (j+1)*ni + k*ni*nj
                n3 = i     + (j+1)*ni + k*ni*nj
                n4 = i     + j*ni     + (k+1)*ni*nj
                n5 = (i+1) + j*ni     + (k+1)*ni*nj
                n6 = (i+1) + (j+1)*ni + (k+1)*ni*nj
                n7 = i     + (j+1)*ni + (k+1)*ni*nj
                conn.append([n0, n1, n2, n3, n4, n5, n6, n7])
    return np.array(conn, dtype=int)

def build_simple_quad_connectivity(dims, loop=True):
    """
    Build a simple quadrilateral connectivity for a single block.
    For Plot3D with dims (ni, nj, nk):
      - ni: number of points along the airfoil (x-y plane)
      - nj: number of points in the z direction (spanwise)
      - nk: number of layers (should be 2 for a single surface)

    INPUTS:
      - loop: if True, points are assumed to loop (e.g. around an airfoil), and the
               point connectivity is such that the last point of the loop is not repeated.
    Returns:
        conn: (n_quad, 4) array of node indices
    """
    ni, nk, nj = dims

    # This function will build quads between the two j-planes (for each i)
    # For dims=[22,2,6], ni=22, nj=2, nk=6
    # We want quads between j=0 and j=1 for all i, for each k
    conn = np.zeros(((ni-1)*(nj-1),4),dtype=int)

    nii = ni
    nii2 = ni-1
    if loop:
        nii = ni-1
        nii2 = nii

    for j in range(nj-1): # "Layers"
        for i in range(nii-1): # Along chord
            n0 = i     + (j  ) * nii
            n1 = i     + (j+1) * nii
            n2 = (i+1) + (j+1) * nii
            n3 = (i+1) + (j  ) * nii 
            conn[i + j * nii2] = [n0, n1, n2, n3]

    if loop:
        # We need to fix the indices for the points that loops and were not added
        for j in range(nj-1):
            conn[(nii-1) + j * nii] = [(nii-1) + j * nii, nii-1 + (j+1) * nii, (j+1) * nii, j * nii]

    #if loop:  # Legacy
    #    nx, nk, ny = dims
    #    nx = nx -1
    #    conn = np.zeros((nx*(ny-1),4),dtype=int)
    #    for j in range(ny-1):
    #        for i in range(nx-1):
    #            conn[i + j * nx] = np.array([i + j * nx, i + (j+1) * nx, i+1 + (j+1) * nx, i+1 + j * nx ])
    #        conn[(nx-1) + j * nx] = np.array([(nx-1) + j * nx, nx-1 + (j+1) * nx, (j+1) * nx, j * nx])

    return conn

def extract_z_plane(coords, dims, loop=True):
    """
    Given full 3D coords for a Plot3D block with nk=2, extract only the surface (z=0) coordinates
    and return the reduced coordinates and a mapping from (i, j) to the new node indices.
    Args:
        coords: (n_points, 3) array
        dims: (ni, nj, nk) tuple
    Returns:
        coords2d: (ni*nj, 2) array (or 3 if you want to keep z=0)
    """
    ni, nj, nk = dims
    if loop:
        nii = ni-1 # coordinates loop, no need to store extra point
    else:
        nii = ni
    j=0 # Only keep j=0 plane (z=0)
    coords2d = np.zeros((nii*nk, 2))
    for k in range(nk):
        for i in range(nii):
            idx = i + j * ni + k * ni * nj
            coords2d[i + k *nii] = coords[idx, :2]

    if loop:
            # Check that the first and last point for each k are the same in the original coords
            for k in range(nk):
                idx_first = 0 + j * ni + k * ni * nj
                idx_last = (ni-1) + j * ni + k * ni * nj
                if not np.allclose(coords[idx_first], coords[idx_last], rtol=1e-10, atol=1e-12):
                    raise ValueError(
                        f"extract_z_plane: For loop=True, expected coords to be periodic in i for k={k}, "
                        f"but got {coords[idx_first]} and {coords[idx_last]}."
                    )


    return coords2d

def find_side_sets(coords, conn, dims, elem_type='QUAD', angle_center=None, inlet_start=None, inlet_span=None, outlet_start=None, inlet_name='inlet', outlet_name='outlet'):
    """
    For a structured quad mesh, find the elements and face indices along the airfoil (first layer) and far-field (last layer).
    Returns a side_sets dict and elem_to_face_nodes mapping.
    """
    ni, nk, nj = dims
    n_elem_i = ni - 1
    n_elem_j = nj - 1

    if elem_type=='HEX':
        SIDE_AIRFOIL = 5
        SIDE_FARFIELD = 6
    else:
        SIDE_AIRFOIL = 4
        SIDE_FARFIELD = 2

    # Airfoil: elements along j=0 (first row of elements)
    airfoil_elements = []
    for i in range(n_elem_i):
        elem_id = i  # first row: elements 0 to n_elem_i-1
        airfoil_elements.append(elem_id + 1)  # Exodus is 1-based

    # Far-field: elements along j=nj-2 (last row of elements)
    farfield_elements = []
    farfield_sides = []
    elem_to_face_nodes = {}
    for i in range(n_elem_i):
        elem_id = (n_elem_j - 1) * n_elem_i + i  # last row
        farfield_sides.append(SIDE_FARFIELD)  
        elem_to_face_nodes[(elem_id+1, SIDE_FARFIELD)] = np.array([id+1  for id in conn[elem_id]]) 
        farfield_elements.append(elem_id + 1)

    side_sets_list = []
    side_sets_list += [{"elements": airfoil_elements, "sides": [SIDE_AIRFOIL]*len(airfoil_elements), "name": "airfoil"}]
    #side_sets[2] = {"elements": farfield_elements, "sides": farfield_sides, "name": "farfield"}

    # --- Find inlet and outlet side sets
    new_side_sets = set_omesh_inlet_outlet_ss(coords, farfield_elements, farfield_sides, elem_to_face_nodes, angle_center, inlet_start, inlet_span, outlet_start, debug=False, 
                                              inlet_name=inlet_name, outlet_name=outlet_name)
    side_sets_list += new_side_sets
    # Create a dictionary for side sets
    side_sets={}
    for i, side_set in enumerate(side_sets_list):
        side_sets[i+1] = side_set
    return side_sets


def plt3d2exo_legacy(input_file, output_file=None,
                 block_base='fluid', inlet_name='inflow', outlet_name='outflow'):

    if output_file is None:
        output_file = os.path.splitext(input_file)[0] + ".exo"

    dims = np.loadtxt(input_file,skiprows=1,max_rows=1,dtype=int)
    a = np.loadtxt(input_file,skiprows=2)
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
    node_count = nx * ny
    cell_count = nx * (ny-1)
    block_names = [block_base]
    sideset_names = ['airfoil',inlet_name, outlet_name]
    sideset_cells = [np.linspace(0,nx-1,nx,dtype=int)+1, 
                     elem_inflow,
                     elem_outflow]  # +1 for 1-based indexing in Exodus
    sideset_sides = [4 * np.ones_like(sideset_cells[0], dtype=int), 
                     2 * np.ones_like(sideset_cells[1], dtype=int),
                     2 * np.ones_like(sideset_cells[2], dtype=int)]  # All sides are on the airfoil

    with ExodusIIFile(output_file, mode="w") as exof:
        exof.put_init('airfoil', ndim, node_count, cell_count,
                      len(block_names), 0, 3) #len(sideset_names))
        exof.put_coord(xyz_2d[:,0], xyz_2d[:,1], np.zeros_like(xyz_2d[:,0]))
        exof.put_coord_names(["X", "Y"])
        exof.put_element_block(1, 'QUAD', cell_count, 4)
        exof.put_element_block_name(1, block_base+'-QUAD')
        exof.put_element_conn(1, conn)
        # Side sets
        for i in range(len(sideset_names)):
            exof.put_side_set_param(i+1, len(sideset_cells[i]))
            exof.put_side_set_name(i+1, sideset_names[i])
            exof.put_side_set_sides(i+1, sideset_cells[i], sideset_sides[i])
    print('Exodus file written: ', output_file, '(legacy)')


def plt3d2exo(input_file, output_file=None, flatten=True, angle_center=None, inlet_start=None, inlet_span=None, outlet_start=None, loop=True,
                 verbose=False, debug=False, 
                 check_zpos=True,
                 block_base='fluid', inlet_name='inlet', outlet_name='outlet',
                 legacy=False
                 ):
    if legacy:
        plt3d2exo_legacy(input_file, output_file, block_base=block_base, inlet_name=inlet_name, outlet_name=outlet_name)
        return

    # --- Read Plot3D file
    coords, dims = read_plot3d(input_file, verbose=verbose, singleblock=True)

    # --- Misc checks
    if dims[-1]==1:
        raise Exception('Last input file dimension needs to be >1.\nInput file dimensions are: {}\nHas mesh been generated? '.format(dims))

    if output_file is None:
        output_file = os.path.splitext(input_file)[0] + ".exo"
        if flatten:
            output_file = rename_n(output_file, nSpan=1)

    if flatten:
        # ---
        if verbose:
            print("Flattening 3D mesh to 2D surface mesh..., loop=", loop)
        # Create a 2D mesh from the 3D coordinates
        coords = extract_z_plane(coords, dims, loop=loop)
        conn =  build_simple_quad_connectivity(dims, loop=loop)
        if check_zpos:
            conn= force_quad_positive_about_z(conn, coords, verbose=True)

        block_name = block_base+'-QUAD'; num_dim=2; elem_type='QUAD'

    else:
        conn = build_simple_hex_connectivity(dims)
        block_name = block_base+'-HEX'; num_dim=3; elem_type='HEX'


    if debug:
        print('Coords 1 ', coords[0,:])
        print('Coords 2 ', coords[1,:])
        print('Coords -2', coords[-2,:])
        print('Coords -1', coords[-1,:])
        print('Conn 1 ', conn[0,:] )
        print('Conn 2 ', conn[1,:] )
        print('Conn -2', conn[-2,:])
        print('Conn -1', conn[-1,:])


    if verbose:
        print("Number of elements:", conn.shape[0], '- nodes per element:', conn.shape[1])

    side_sets = find_side_sets(coords, conn, dims, elem_type=elem_type, 
                               angle_center=angle_center, inlet_start=inlet_start, inlet_span=inlet_span, outlet_start=outlet_start,
                               inlet_name=inlet_name, outlet_name = outlet_name
                               )
    conn += 1
    write_exodus(output_file, coords, conn, verbose=True, side_sets=side_sets, block_name=block_name, num_dim=num_dim, title='airfoil')

    if debug and flatten:
        n=dims[0][0]
        plt.figure()
        plt.plot(coords[0, 0], coords[0, 1], 'ro')
        # Plot coordinates layer by layer, "n" points at a time
        for i in range(0, 10, 1):
            x, y = coords[i*n :(i+1)*n, 0], coords[i*n:(i+1)*n, 1]
            if len(x)==0:
                    print('Maximum layer reached', i)
                    break
        plt.plot(x,y, '-')
        plt.show()

def plt3d2exo_CLI():
    """
    Command-line interface for converting a Plot3D file to an Exodus file.
    """
    import argparse
    parser = argparse.ArgumentParser(description="Convert a Plot3D file to an Exodus file.")
    parser.add_argument("input_file", type=str, help="Path to the input Plot3D file.")
    parser.add_argument("-o", "--output_file", type=str, default=None, help="Path to the output Exodus file.")
    parser.add_argument("--angle-center", type=float, nargs=2, default=(0.0, 0.0), help="Center for angle estimation (x, y). Default is (0.0, 0.0).")
    parser.add_argument("--inlet-start", type=float, default=None, help="Start angle of inlet segment (in degrees). Optional.")
    parser.add_argument("--inlet-span", type=float, default=None, help="Angular span of inlet segment (in degrees). Optional.")
    parser.add_argument("--outlet-start", type=float, default=None, help="Start angle of outlet segment (in degrees). Optional.")
    parser.add_argument("-v", "--verbose", action="store_true", help="Enable verbose output.")
    parser.add_argument("-d", "--debug", action="store_true", help="Enable debug mode with plots.")
    parser.add_argument("--inlet-name", type=str, default="inlet", help="Name for the inlet sideset (default: 'inlet'. alternative: 'inflow').")
    parser.add_argument("--outlet-name", type=str, default="outlet", help="Name for the outlet sideset (default: 'outlet'. alternative: 'outflow').")
    parser.add_argument("--block-base", type=str, default="fluid", help="Base name for the block (default: 'fluid', alternative: 'Flow').")
    parser.add_argument("--flatten", action="store_true", help="Write quads instead of hex.")
    parser.add_argument("--loop", action="store_true", help="Assume poitns are looped (only used with --flatten).")
    parser.add_argument("--no-zpos-check", action="store_true", help="Do not check and enforce the orientation of quads about Z axis (default: check).")
    parser.add_argument("--legacy", action="store_true", help="Use legacy writer (implies --loop and --flatten)")
    args = parser.parse_args()

    # Convert angles to radians if provided
    inlet_start = np.radians(args.inlet_start) if args.inlet_start is not None else None
    inlet_span = np.radians(args.inlet_span) if args.inlet_span is not None else None
    outlet_start = np.radians(args.outlet_start) if args.outlet_start is not None else None

    plt3d2exo(
        input_file=args.input_file,
        output_file=args.output_file,
        verbose=args.verbose,
        flatten=args.flatten,
        angle_center=tuple(args.angle_center),
        inlet_start=inlet_start,
        inlet_span=inlet_span,
        outlet_start=outlet_start,
        debug=args.debug,
        inlet_name=args.inlet_name,
        outlet_name=args.outlet_name,
        block_base=args.block_base,
        loop=args.loop,
        check_zpos= not args.no_zpos_check,
        legacy=args.legacy,
    )

    # Ganesh

if __name__ == "__main__":
    input_file = 'naca0012_rans_vol.fmt'
    plt3d_to_exo(input_file, verbose=True)
