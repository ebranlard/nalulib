import numpy as np
import os
from nalulib.exodusii.file import ExodusIIFile
from nalulib.essentials import *
#from nalulib.gmesh import *
#from nalulib.meshlib import *
#from nalulib.gmesh_gmesh2exodus import gmesh_to_exo
from nalulib.exodus_core import set_omesh_inlet_outlet_ss, write_exodus
import matplotlib.pyplot as plt



def read_plot3d(filename, verbose=False):
    """
    Reads a simple multi-block Plot3D file (formatted, ASCII).
    Returns:
        coords: (n_points, 3) array
        dims: (n_blocks, 3) list of (ni, nj, nk)
    """
    print(f"Reading Plot3D file: {filename}")
    with open(filename, "r") as f:
        nblocks = int(f.readline())
        dims = []
        for _ in range(nblocks):
            dims.append(tuple(int(x) for x in f.readline().split()))
        coords_list = []
        for block in range(nblocks):
            ni, nj, nk = dims[block]
            npts = ni * nj * nk
            block_coords = np.zeros((npts, 3))
            for idim in range(3):
                for k in range(nk):
                    for j in range(nj):
                        for i in range(ni):
                            idx = i + j * ni + k * ni * nj
                            val = float(f.readline())
                            block_coords[idx, idim] = val
            coords_list.append(block_coords)
        coords = np.vstack(coords_list)

    if verbose:
        print("Coordinates shape:", coords.shape)
        print("Block dimensions:", dims)
        # Print min/max for x, y, z
        x_min, x_max = coords[:, 0].min(), coords[:, 0].max()
        y_min, y_max = coords[:, 1].min(), coords[:, 1].max()
        z_min, z_max = coords[:, 2].min(), coords[:, 2].max()
        print(f"x range: [{x_min:.6f}, {x_max:.6f}]")
        print(f"y range: [{y_min:.6f}, {y_max:.6f}]")
        print(f"z range: [{z_min:.6f}, {z_max:.6f}]")

    return coords, dims

def build_simple_hex_connectivity(dims):
    """
    Build a simple hexahedral connectivity for a single block.
    Assumes periodic/looping in i, j, k directions.
    Returns:
        conn: (n_hex, 8) array of node indices
    """
    ni, nj, nk = dims[0]  # Only first block for simplicity
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

def build_simple_quad_connectivity(dims):
    """
    Build a simple quadrilateral connectivity for a single block.
    For Plot3D with dims (ni, nj, nk):
      - ni: number of points along the airfoil (x-y plane)
      - nj: number of points in the z direction (spanwise)
      - nk: number of layers (should be 2 for a single surface)
    Returns:
        conn: (n_quad, 4) array of node indices
    """
    ni, nk, nj = dims[0]  # Only first block for simplicity
    # This function will build quads between the two j-planes (for each i)
    # For dims=[22,2,6], ni=22, nj=2, nk=6
    # We want quads between j=0 and j=1 for all i, for each k
    conn = []
    for j in range(nj-1): # "Layers"
        for i in range(ni - 1): # Along chord
            n0 = i     + (j  ) * ni
            n1 = (i+1) + (j  ) * ni 
            n2 = (i+1) + (j+1) * ni
            n3 = i     + (j+1) * ni
            conn.append([n0, n3, n2, n1])
    return np.array(conn, dtype=int)

def simplify_coords_for_surface(coords, dims):
    """
    Given full 3D coords for a Plot3D block with nk=2, extract only the surface (z=0) coordinates
    and return the reduced coordinates and a mapping from (i, j) to the new node indices.
    Args:
        coords: (n_points, 3) array
        dims: (ni, nj, nk) tuple
    Returns:
        coords2d: (ni*nj, 2) array (or 3 if you want to keep z=0)
    """
    ni, nj, nk = dims[0]
    # Only keep j=0 plane (z=0)
    coords2d = []
    j=0
    for k in range(nk):
        for i in range(ni):
            idx = i + j * ni + k * ni * nj
            coords2d.append(coords[idx, :2])
    return np.array(coords2d)

def find_side_sets(coords, conn, dims, elem_type='QUAD', angle_center=None, inlet_start=None, inlet_span=None, outlet_start=None):
    """
    For a structured quad mesh, find the elements and face indices along the airfoil (first layer) and far-field (last layer).
    Returns a side_sets dict and elem_to_face_nodes mapping.
    """
    ni, nk, nj = dims[0]
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
    new_side_sets = set_omesh_inlet_outlet_ss(coords, farfield_elements, farfield_sides, elem_to_face_nodes, angle_center, inlet_start, inlet_span, outlet_start, debug=False)
    side_sets_list += new_side_sets
    # Create a dictionary for side sets
    side_sets={}
    for i, side_set in enumerate(side_sets_list):
        side_sets[i+1] = side_set
    return side_sets


def plt3d_to_exo(input_file, output_file=None, flatten=True, angle_center=None, inlet_start=None, inlet_span=None, outlet_start=None, verbose=False, debug=False):

    # Read Plot3D file
    coords, dims = read_plot3d(input_file, verbose=verbose)

    if output_file is None:
        output_file = os.path.splitext(input_file)[0] + ".exo"
        if flatten:
            output_file = rename_n(output_file, nSpan=1)

    if flatten:
        # ---
        if verbose:
            print("Flattening 3D mesh to 2D surface mesh...")
        # Create a 2D mesh from the 3D coordinates
        coords = simplify_coords_for_surface(coords, dims)
        conn =  build_simple_quad_connectivity(dims)
        block_name = 'fluid-quad'; num_dim=2; elem_type='QUAD'

    else:
        conn = build_simple_hex_connectivity(dims)
        block_name = 'fluid-hex'; num_dim=3; elem_type='HEX'

    if verbose:
        print("Number of elements:", conn.shape[0], '- nodes per element:', conn.shape[1])

    side_sets = find_side_sets(coords, conn, dims, elem_type=elem_type, angle_center=angle_center, inlet_start=inlet_start, inlet_span=inlet_span, outlet_start=outlet_start)
    conn += 1
    write_exodus(output_file, coords, conn, verbose=True, side_sets=side_sets, block_name=block_name, num_dim=num_dim)

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

def plt3d2exo():
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
    parser.add_argument("--flatten", action="store_true", help="Write quads only.")
    parser.add_argument("-v", "--verbose", action="store_true", help="Enable verbose output.")
    parser.add_argument("-d", "--debug", action="store_true", help="Enable debug mode with plots.")
    args = parser.parse_args()

    # Convert angles to radians if provided
    inlet_start = np.radians(args.inlet_start) if args.inlet_start is not None else None
    inlet_span = np.radians(args.inlet_span) if args.inlet_span is not None else None
    outlet_start = np.radians(args.outlet_start) if args.outlet_start is not None else None

    plt3d_to_exo(
        input_file=args.input_file,
        output_file=args.output_file,
        verbose=args.verbose,
        flatten=args.flatten,
        angle_center=tuple(args.angle_center),
        inlet_start=inlet_start,
        inlet_span=inlet_span,
        outlet_start=outlet_start,
        debug=args.debug
    )

if __name__ == "__main__":
    input_file = 'naca0012_rans_vol.fmt'
    plt3d_to_exo(input_file, verbose=True)
