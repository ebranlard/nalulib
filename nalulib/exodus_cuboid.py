import numpy as np
from nalulib.exodus_core import write_exodus_hex, write_exodus_quads

def _parse_dims(ls, ns):
    # Accepts scalars, tuples, or lists for ls and ns
    if isinstance(ls, (int, float)): ls = (ls,)
    if isinstance(ns, int): ns = (ns,)
    lx = ls[0] if len(ls) > 0 else 1
    ly = ls[1] if len(ls) > 1 else lx
    lz = ls[2] if len(ls) > 2 else 0
    nx = ns[0] if len(ns) > 0 else 2
    ny = ns[1] if len(ns) > 1 else nx
    nz = ns[2] if len(ns) > 2 else 1
    return lx, nx, ly, ny, lz, nz

def _cuboid_nodes(lx, nx, ly, ny, lz, nz, legacy=False, center_origin=True):
    # Generates node coordinates for a regular cuboid mesh, centered on origin
    # Node order: x changes fastest, then y, then z (to match your example)

    if legacy:
        # NOTE: match the pointwise outputs..
        x = np.linspace(lx/2, -lx/2, nx)
        y = np.linspace(ly/2, -ly/2, ny)
        z = np.linspace(0, lz, nz) if nz > 1 and lz > 0 else np.zeros(1)
    else:
        x = np.linspace(-lx/2, lx/2, nx)
        y = np.linspace(-ly/2, ly/2, ny)
        z = np.linspace(0, lz, nz) if nz > 1 and lz > 0 else np.zeros(1)

    if not center_origin:
        x += lx / 2
        y += ly / 2

    coords = np.zeros((nx*ny*nz, 3))
    idx = 0
    for k in range(nz):
        for j in range(ny):
            for i in range(nx):
                coords[idx, :] = x[i], y[j], z[k]
                idx += 1
    return coords, x, y, z

def _cuboid_hex_connectivity(nx, ny, nz, legacy=False):
    # Returns HEX8 connectivity (1-based) for a regular mesh
    nxy = nx * ny
    conn0 = np.zeros(((nx-1)*(ny-1)*(nz-1), 8), dtype=int)
    idx = 0
    for k in range(nz-1):
        for j in range(ny-1):
            for i in range(nx-1):
                n0 = i + j*nx + k*nxy
                n1 = n0 + 1
                n2 = n0 + nx + 1
                n3 = n0 + nx
                n4 = n0 + nxy
                n5 = n1 + nxy
                n6 = n2 + nxy
                n7 = n3 + nxy
                conn0[idx,:] = [n0, n1, n2, n3, n4, n5, n6, n7]
                idx += 1
    return conn0

def _cuboid_quad_connectivity(nx, ny):
    # Returns QUAD4 connectivity (1-based) for a regular mesh
    conn = np.zeros(((nx-1)*(ny-1), 4), dtype=int)
    idx = 0
    for j in range(ny-1):
        for i in range(nx-1):
            n0 = i + j*nx
            n1 = n0 + 1
            n2 = n0 + nx + 1
            n3 = n0 + nx
            conn[idx,:] = [n0+1, n1+1, n2+1, n3+1]
            idx += 1
    return conn

def _cuboid_hex_sidesets(nx, ny, nz, legacy=False):
    # NOTE:  I did all the reversed to match the pointwise outputs..
    # But this is all because the bounds go from +lx/2 to -lx/2, etc.
    # It might be best to use something more intuitive...


    nElemX = nx-1
    nElemY = ny-1
    nElemZ = nz-1
    nxy    = nElemX * nElemY

    # Back (z=0): k=0, side 5 - [ OK ]
    back_elems  = np.empty(nElemX*nElemY, dtype=int)
    idx = 0
    k=0 
    for i in range(nElemX):
        for j in reversed(range(nElemY)):
            back_elems[idx] = 1 + i + j*nElemX + k*nxy
            idx += 1
    back_sides  = np.ones_like(back_elems) * 5

    # Front (z=lz): k=nElemZ-1, side 6 - [ OK ]
    front_elems = np.empty(nElemX*nElemY, dtype=int)
    idx = 0
    k = (nElemZ-1)
    for i in range(nElemX):
        for j in reversed(range(nElemY)):
            front_elems[idx] = 1 + i + j*nElemX + k*nxy
            idx += 1
    front_sides = np.ones_like(front_elems) * 6

    # Bottom (y=-ly/2): j=ny, side 3
    bottom_elems = np.empty(nElemX*nElemZ, dtype=int)
    idx = 0
    j = ny-2 # HM, that seems weird
    for i in reversed(range(nElemX)):
        for k in range(nElemZ):
            bottom_elems[idx] = 1 + i + j*nElemX + k*nxy
            idx += 1
    bottom_sides = np.ones_like(bottom_elems) * 3

    # Top (y=+ly/2): j=nElemY-1, side 1
    top_elems = np.empty(nElemX*nElemZ, dtype=int)
    idx = 0
    j= 0
    for i in range(nElemX):
        for k in range(nElemZ):
            top_elems[idx] = 1 + i + j*nElemX + k*nxy
            idx += 1
    top_sides = np.ones_like(top_elems) * 1

    # Inlet (x=-lx/2)
    inlet_elems = np.empty(nElemY*nElemZ, dtype=int)
    idx = 0
    i = nx-2
    for j in range(nElemY):
        for k in range(nElemZ):
            inlet_elems[idx] = 1 + i + j*nElemX + k*nxy
            idx += 1
    inlet_sides = np.ones_like(inlet_elems) * 2

    # Outlet (x=+lx/2): i=0, side 4
    outlet_elems = np.empty(nElemY*nElemZ, dtype=int)
    idx = 0
    i = 0 
    for j in reversed(range(nElemY)):
        for k in range(nElemZ):
            outlet_elems[idx] = 1 + i+ j*nElemX + k*nxy
            idx += 1
    outlet_sides = np.ones_like(outlet_elems) * 4

    return {
        6: {"elements": back_elems,   "sides": back_sides,   "name": "back_bg"},    # z=0
        4: {"elements": bottom_elems, "sides": bottom_sides, "name": "bottom_bg"},  # y=-ly/2
        5: {"elements": front_elems,  "sides": front_sides,  "name": "front_bg"},   # z=lz
        1: {"elements": inlet_elems,  "sides": inlet_sides,  "name": "inlet_bg"},   # x=-lx/2
        2: {"elements": outlet_elems, "sides": outlet_sides, "name": "outlet_bg"},  # x=+lx/2
        3: {"elements": top_elems,    "sides": top_sides,    "name": "top_bg"},     # y=+ly/2
    }

def exo_cuboid(filename, ls, ns, center_origin=True, block_name='background-HEX', verbose=False, legacy=True):
    """ Creates a regular cuboid mesh and writes it to an Exodus file."""
    lx, nx, ly, ny, lz, nz = _parse_dims(ls, ns)
    if verbose:
        print(f"Creating cuboid mesh")
        print(f"Lengths        : {lx}, {ly}, {lz}")
        print(f"Number of nodes: {nx}, {ny}, {nz}")
        print(f"Block name     : {block_name}" )
        print(f"Legacy mode    : {legacy}")

    if not legacy:
        raise NotImplementedError("Only legacy mode is implemented for cuboid meshes.")

    coords, x, y, z  = _cuboid_nodes(lx, nx, ly, ny, lz, nz, legacy=legacy, center_origin=center_origin)
    # print min max values
    if verbose:
        print(f"x range        : [{x[0]:.6f}, {x[-1]:.6f}]")
        print(f"y range        : [{y[0]:.6f}, {y[-1]:.6f}]")
        print(f"z range        : [{z[0]:.6f}, {z[-1]:.6f}]")

    if nz > 1 and lz > 0:
        conn0    = _cuboid_hex_connectivity(nx, ny, nz, legacy=legacy)
        conn     = conn0 + 1
        sidesets = _cuboid_hex_sidesets(nx, ny, nz, legacy=legacy)
        if verbose:
            print(f"Nodes          : {coords.shape[0]}" )
            print(f"Elements       : {conn.shape[0]}")
        if filename is not None:
            write_exodus_hex(filename, coords, conn, block_name=block_name, verbose=verbose, side_sets=sidesets)
    else:
        conn = _cuboid_quad_connectivity(nx, ny)
        # TODO: add 2D sidesets if needed
        if verbose:
            print(f"Nodes          : {coords.shape[0]}" )
            print(f"Elements       : {conn.shape[0]}")
        if filename is not None:
            write_exodus_quads(filename, coords, conn, block_name=block_name.replace('HEX','QUAD'), verbose=True)


    return coords, conn, sidesets

def exo_cuboid_CLI():
    import argparse
    parser = argparse.ArgumentParser(description="Create a regular cuboid mesh and write to Exodus file.")
    parser.add_argument("-o", "--output", type=str, help="Output Exodus filename")
    parser.add_argument("--ls", type=float, nargs='+', required=True, help="Dimensions (lx [ly [lz]])")
    parser.add_argument("--ns", type=int, nargs='+', required=True, help="Number of nodes (nx [ny [nz]])")
    parser.add_argument("--no-center", action='store_true', help="Do not center mesh on origin")
    parser.add_argument("--block-name", type=str, default='background-HEX', help="Block name")
    parser.add_argument("--verbose", action='store_true', help="Verbose output")
    parser.add_argument("--legacy", action='store_true', help="Store in order similar to pointwise background")
    args = parser.parse_args()
    exo_cuboid(args.output, args.ls, args.ns, center_origin=not args.no_center, block_name=args.block_name, verbose=args.verbose, legacy=args.legacy)

if __name__ == "__main__":
    exo_cuboid_CLI()
    #exo_cuboid(
        #filename      = "background2.exo",
        #ls            = (120, 120, 4),
        #ns            = (166, 166, 121),
        #center_origin = True,
        #block_name    = "background-HEX",
        #verbose       = True,
        #legacy=True
    #)