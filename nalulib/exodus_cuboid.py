import numpy as np
from nalulib.exodus_core import write_exodus_hex, write_exodus_quads
from nalulib.plot3D_plot3D2exo import build_simple_hex_connectivity # TODO merge

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

def _cuboid_quad_connectivity(nx, ny):
    # TODO use build_simple_quad_connectivity

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
    nElemX = nx-1; nElemY = ny-1; nElemZ = nz-1; nxy = nElemX * nElemY

    # Back (z=0): k=0
    if legacy: # Using reversed...
        back_elems = np.array([1 + i + j*nElemX for i in range(nElemX) for j in reversed(range(nElemY))], dtype=int)
    else:
        back_elems = np.array([1 + i + j*nElemX for i in range(nElemX) for j in range(nElemY)], dtype=int)
    back_sides = np.ones_like(back_elems) * (5 if legacy else 6)

    # Front (z=lz): k=nElemZ-1
    k = nElemZ-1
    if legacy: # Using reversed...
        front_elems = np.array([1 + i + j*nElemX + k*nxy for i in range(nElemX) for j in reversed(range(nElemY))], dtype=int)
    else:
        front_elems = np.array([1 + i + j*nElemX + k*nxy for i in range(nElemX) for j in range(nElemY)], dtype=int)
    front_sides = np.ones_like(front_elems) * (6 if legacy else 5)

    # Bottom (y=-ly/2): j=ny-2 (legacy) or j=0 (else)
    j = ny-2 if legacy else 0
    if legacy: # Using reversed...
        bottom_elems = np.array([1 + i + j*nElemX + k*nxy for i in reversed(range(nElemX)) for k in range(nElemZ)], dtype=int)
    else:
        bottom_elems = np.array([1 + i + j*nElemX + k*nxy for i in range(nElemX) for k in range(nElemZ)], dtype=int)
    bottom_sides = np.ones_like(bottom_elems) * (3 if legacy else 1)

    # Top (y=+ly/2): j=0 (legacy) or j=nElemY-1 (else)
    j = 0 if legacy else nElemY-1
    top_elems = np.array([1 + i + j*nElemX + k*nxy for i in range(nElemX) for k in range(nElemZ)], dtype=int)
    top_sides = np.ones_like(top_elems) * (1 if legacy else 3)

    # Inlet (x=-lx/2): i=nx-2 (legacy) or i=0 (else)
    i = nx-2 if legacy else 0
    inlet_elems = np.array([1 + i + j*nElemX + k*nxy for j in range(nElemY) for k in range(nElemZ)], dtype=int)
    inlet_sides = np.ones_like(inlet_elems) * (2 if legacy else 4)

    # Outlet (x=+lx/2): i=0 (legacy) or i=nx-2 (else)
    i = 0 if legacy else nx-2
    if legacy: # Using reversed...
        outlet_elems = np.array([1 + i + j*nElemX + k*nxy for j in reversed(range(nElemY)) for k in range(nElemZ)], dtype=int)
    else:
        outlet_elems = np.array([1 + i + j*nElemX + k*nxy for j in range(nElemY) for k in range(nElemZ)], dtype=int)
    outlet_sides = np.ones_like(outlet_elems) * (4 if legacy else 2)

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
    coords, x, y, z  = _cuboid_nodes(lx, nx, ly, ny, lz, nz, legacy=legacy, center_origin=center_origin)
    if verbose:
        print(f"Creating cuboid mesh")
        print(f"Lengths        : {lx}, {ly}, {lz}")
        print(f"Number of nodes: {nx}, {ny}, {nz}")
        print(f"Block name     : {block_name}" )
        print(f"Legacy mode    : {legacy}")
        print(f"x range        : [{x[0]:.6f}, {x[-1]:.6f}]")
        print(f"y range        : [{y[0]:.6f}, {y[-1]:.6f}]")
        print(f"z range        : [{z[0]:.6f}, {z[-1]:.6f}]")

    if nz > 1 and lz > 0:
        conn0    = build_simple_hex_connectivity((nx, ny, nz))
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
    parser.add_argument("-l", "--ls", type=float, nargs='+', required=True, help="Dimensions (lx [ly [lz]])")
    parser.add_argument("-n", "--ns", type=int, nargs='+', required=True, help="Number of nodes (nx [ny [nz]])")
    parser.add_argument("--no-center", action='store_true', help="Do not center mesh on origin")
    parser.add_argument("--block-name", type=str, default='background-HEX', help="Block name")
    parser.add_argument("-v", "--verbose", action='store_true', help="Verbose output")
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