import numpy as np
from nalulib.exodusii.file import ExodusIIFile
from nalulib.essentials import Timer

# --------------------------------------------------------------------------
# --- Connectivity
# --------------------------------------------------------------------------
# NOTE: this gives a negative orientation about z
#HEX_FACES = [
#    [1, 5, 4, 0],  # 1 
#    [1, 2, 6, 5],  # 2
#    [3, 2, 6, 7],  # 3
#    [0, 3, 7, 4],  # 4
#    [0, 1, 2, 3],  # 5
#    [4, 5, 6, 7],  # 6
#]

# NOTE: this gives a positive orientation about z
HEX_FACES = [
    [0, 4, 5, 1],  # 1 
    [5, 6, 2, 1],  # 2
    [7, 6, 2, 3],  # 3
    [4, 7, 3, 0],  # 4
    [3, 2, 1, 0],  # 5
    [7, 6, 5, 4],  # 6
]


QUAD_SIDES = [
    [0, 1],
    [1, 2],
    [2, 3],
    [3, 0]]

ELEMTYPE_2_SIDE = {'HEX': HEX_FACES, 'HEX8': HEX_FACES, 'QUAD': QUAD_SIDES}
ELEMTYPE_2_NUM_DIM = {'HEX': 3, 'HEX8': 3, 'QUAD': 2}
NUM_DIM_2_ELEMTYPE = {3: 'HEX8', 2: 'QUAD'}

# --------------------------------------------------------------------------
# --- QUADS
# --------------------------------------------------------------------------
def write_exodus_quads(filename, coords, conn, block_name='fluid-quad', **kwargs):
    """
    Write a 2D quad mesh to an Exodus file.
    Args:
        coords: (n_points, 2) or (n_points, 3) array of node coordinates
        conn: (n_quads, 4) array of node indices (1-based)
        filename: output Exodus filename
        title: Exodus file title
        **kwargs: see write_exodus
    """
    coords = np.array(coords)
    # Ensure coords is (n_points, 3)
    if coords.shape[1] == 2:
        coords = np.hstack([coords, np.zeros((coords.shape[0], 1))])
    else:
        coords = coords
    write_exodus(filename, coords, conn, block_name=block_name, num_dim=2, **kwargs )


def write_exodus_hex(filename, coords, conn, block_name='fluid-hex', **kwargs):
    """
      - **kwargs:  see write_exodus
    """
    write_exodus(filename, coords, conn, block_name=block_name, num_dim=3, **kwargs )

def write_exodus(filename, coords, conn, title="", verbose=True, side_sets=None, num_dim=3, block_name='fluid', profiler=False, double=False):
    """ 
    INPUTS:
      - filename: output Exodus filename
      - coords: (n_points, 3) array of node coordinates
      - conn: (n_quads, 4) array of node indices (1-based)
    """
    # --- Sanitize inputs
    if side_sets is None:
        side_sets={}
    coords = np.array(coords)
    conn = np.array(conn)
    if coords.shape[1] == 2:
        coords = np.hstack([coords, np.zeros((coords.shape[0], 1))])
    else:
        coords = coords

    assert coords.shape[1] == 3, "Coords should be of size 3"
    assert np.min(conn) == 1, "Connectivity must be 1-based"
    assert np.max(conn)<= len(coords), "Connectivity must be less than number of points"

    with Timer('Writing file', silent=not profiler, writeBefore=True):
        with ExodusIIFile(filename, mode="w") as exo:
            exo.put_init(
                title=title,
                num_dim=num_dim,
                num_nodes=len(coords),
                num_elem=len(conn),
                num_elem_blk=1,
                num_node_sets=0,
                num_side_sets=len(side_sets),
                double=double #  Added by Emmanuel
            )

            # Write node coordinates
            exo.put_coord(coords[:, 0], coords[:, 1], coords[:, 2])
            if num_dim == 2:
                exo.put_coord_names(["X", "Y"])
            else:   
                exo.put_coord_names(["X", "Y", "Z"])

            # Write element block
            exo.put_element_block(1, NUM_DIM_2_ELEMTYPE[num_dim], len(conn), 2**num_dim)
            exo.put_element_conn(1, conn)
            exo.put_element_block_name(1, block_name)

            # Write side sets
            for side_set_id, side_set_data in side_sets.items():
                exo.put_side_set_param(side_set_id, len(side_set_data["elements"]))
                exo.put_side_set_sides(side_set_id, side_set_data["elements"], side_set_data["sides"])
                exo.put_side_set_name(side_set_id, side_set_data["name"])
            exo.close()
    del exo



    if verbose:
        print(f"Exodus file written: {filename}")

def quad_is_positive_about_z(node_ids, coords, ioff=1):
    """
    Checks if the 4 node IDs (for a quad face in the x-y plane) are ordered such that
    the normal vector points in the positive z direction (right-hand rule).
    Returns True if the order is positive about z, False otherwise.

    Args:
        node_ids: list/array of 4 node indices (ordered as in the element connectivity)
        coords: (n_nodes, 3) array of node coordinates
        ioff : typically, node_ids in Exodus start at 1, so for python indexing with offset by ioff=1
               If node_ids use python indexing, then ioff=0, and the node_ids correspond 
               directly to the index in coords

    Returns:
        True if the quad is ordered positively about z, False otherwise.
    """
    # Get the x, y coordinates of the quad
    pts = np.asarray([coords[n-ioff][:2] for n in node_ids])
    # Compute the signed area (shoelace formula)
    area = 0.5 * (
        pts[0,0]*pts[1,1] + pts[1,0]*pts[2,1] + pts[2,0]*pts[3,1] + pts[3,0]*pts[0,1]
        - pts[1,0]*pts[0,1] - pts[2,0]*pts[1,1] - pts[3,0]*pts[2,1] - pts[0,0]*pts[3,1]
    )
    return area > 0

def force_quad_positive_about_z(conn, coords, verbose=False, ioff=1):
    """
    For each quad in conn, check if it is positive about z using quad_is_positive_about_z.
    If not, remap the connectivity from [0,1,2,3] to [3,2,1,0] 
    Modifies conn in-place and returns it.
    """
    nquads = conn.shape[0]
    conn_min = np.min(conn)
    if conn_min==0:
        if ioff!=0:
            # We force the user to think about the connectivity indexing
            raise Exception("Connectivity starts at 0, ioff should be 0")
    elif conn_min==1:
        if ioff!=1:
            raise Exception("Connectivity starts at 1 but ioff is 0, it's ok, but not currently anticipated. Contact dev")
        pass # Hopefully the user knows
        #ioff=1
    else:
        raise Exception('The code should work in that case, but such application are currently not anticipated. Contact dev.')

    if np.max(conn)!=len(coords)+ioff-1:
        raise Exception('[WARN] force_quad_positive_about_z is called with a connectivity that doesnt cover the nodes exactly. Connecitivyt min-max IDs [{} {}], number of coords: {}'.format(conn_min, np.max(conn), len(coords)-1+ioff))
    #print('conn max', np.max(conn))
    #print('conn max', len(coords)-1)
    nRev=0
    for iquad in range(nquads):
        if not quad_is_positive_about_z(conn[iquad], coords, ioff=ioff):
            # Remap: [0,1,2,3] -> [3,2,1,0]
            conn[iquad] = [conn[iquad][3], conn[iquad][2], conn[iquad][1], conn[iquad][0]]
            nRev+= 1
    if nRev>0 or verbose:
        print(f"[INFO] Reversed {nRev}/{nquads} quads to ensure positive orientation about z.")
    return conn



# --------------------------------------------------------------------------
# --- Side sets
# --------------------------------------------------------------------------
def check_exodus_side_sets_exist(exo, side_set_names):
    """
    Checks whether the given side set name or list of names exists in the Exodus file.
    If not, raises a ValueError and prints the list of allowed side set names.
    """
    # Get all side set names from the Exodus file
    allowed_names = [str(exo.get_side_set_name(ss_id)) for ss_id in exo.get_side_set_ids()]
    if isinstance(side_set_names, str):
        side_set_names = [side_set_names]
    missing = [name for name in side_set_names if name not in allowed_names]
    if missing:
        raise ValueError(
            f"Side set(s) {missing} not found in Exodus file.\n"
            f"Allowed side sets are: {allowed_names}\n"
            f"Note: inlet and outlet name can be specified as arguments (--inlet-name, --outlet-name)."
        )


def set_omesh_inlet_outlet_ss(node_coords, combined_elements, combined_sides, elem_to_face_nodes, angle_center=None, 
                              inlet_start=None, inlet_span=None, outlet_start=None, 
                              inlet_name='inlet', outlet_name='outlet',
                              debug=False):
    """
    Adjust inlet and outlet side sets based on angle

    Parameters:
        side_sets (dict): Original side sets.
        node_coords (np.ndarray): Rotated node coordinates (N x 3).
        angle_center (tuple): Center for defining angles
        inlet_start (float): Start angle of the inlet segment (in radians).
        inlet_span (float): Angular span of the inlet segment (in radians).
        outlet_start (float): Start angle of the outlet segment (in radians).
        elem_to_face_nodes (dict): Mapping of (element ID, side ID) to node IDs.

    Returns:
        dict: Adjusted side sets.
    """
    from nalulib.angles import is_angle_in_segment
    if angle_center is None:    
        angle_center = (0.0, 0.0)

    if inlet_start is None:
        inlet_start = 90 * np.pi / 180
    if inlet_span is None:
        inlet_span = 180 * np.pi / 180
    if outlet_start is None:
        outlet_start = 270 * np.pi / 180


    combined_elements = np.array(combined_elements)
    combined_sides = np.array(combined_sides)
    if debug:
        print('Combined elements:', combined_elements)
        print('Combined sides:', combined_sides)

    # Compute angles for the middle points of the combined elements' faces
    num_faces = len(combined_elements)
    midpoints = np.zeros((num_faces, 2))  # Only X and Y are needed for 2D angles
    combined_mid_angles = np.zeros(num_faces)
    for i, (elem, side) in enumerate(zip(combined_elements, combined_sides)):
        face_node_ids = elem_to_face_nodes[(elem, side)]
        face_coords = node_coords[face_node_ids - 1]  # Convert to 0-based indexing
        midpoints[i] = np.mean(face_coords[:, :2], axis=0)  # Compute the midpoint (X, Y)
        combined_mid_angles[i] = np.arctan2(midpoints[i, 1] - angle_center[1], midpoints[i, 0] - angle_center[0])

    # Normalize angles to [0, 2π]
    combined_mid_angles = np.mod(combined_mid_angles, 2 * np.pi) 

    if debug:
        print('Combined mid angles (degrees):', np.around(np.degrees(combined_mid_angles),1).astype(int))
        print('Segment: Inlet:', np.degrees(inlet_start), 'Span:', np.degrees(inlet_span), 'End:', np.degrees(inlet_start + inlet_span))

    # Determine new inlet and outlet elements based on angular segments
    outlet_span = 2 * np.pi - inlet_span
    inlet_mask  = is_angle_in_segment(combined_mid_angles, inlet_start, inlet_span, strict_upper=False)
    outlet_mask = is_angle_in_segment(combined_mid_angles, outlet_start, outlet_span, strict_upper=False)
    # Check if the two are disjoint and forms a complete set    
    try:
        assert np.all(np.logical_xor(inlet_mask, outlet_mask)), "Inlet and outlet masks are not disjoint."
        assert np.all(inlet_mask | outlet_mask), "Inlet and outlet masks do not cover all elements!"    
    except AssertionError as e:
        print('[WARN] Inlet and outlet masks are not disjoint or do not cover all elements! Using a strict upper bound')
        inlet_mask  = is_angle_in_segment(combined_mid_angles, inlet_start, inlet_span, strict_upper=True)
        outlet_mask = is_angle_in_segment(combined_mid_angles, outlet_start, outlet_span, strict_upper=True)
        try:
            assert np.all(np.logical_xor(inlet_mask, outlet_mask)), "Inlet and outlet masks are not disjoint."
            assert np.all(inlet_mask | outlet_mask), "Inlet and outlet masks do not cover all elements!"    
        except AssertionError as e:
            print('[WARN] Inlet and outlet masks are not disjoint or do not cover all elements despite upper bound! Using a negative mask for outlet.')
            outlet_mask = np.logical_not(inlet_mask)

    if debug:
        print('Inlet mask:', inlet_mask)
        print('Outlet mask:', outlet_mask)

    new_inlet_elements = combined_elements[inlet_mask]
    new_inlet_sides = combined_sides[inlet_mask]
    new_outlet_elements = combined_elements[outlet_mask]
    new_outlet_sides = combined_sides[outlet_mask]

    # Ensure no intersection and full coverage
    inlet_set = set(zip(new_inlet_elements, new_inlet_sides))
    outlet_set = set(zip(new_outlet_elements, new_outlet_sides))
    assert inlet_set.isdisjoint(outlet_set), "Inlet and outlet side sets have overlapping elements!"
    assert inlet_set.union(outlet_set) == set(zip(combined_elements, combined_sides)), \
        "Inlet and outlet side sets do not cover all combined elements!"

    # Update side sets
    side_sets = []
    side_sets.append( {
        "elements": new_inlet_elements.tolist(),
        "sides": new_inlet_sides.tolist(),
        "name": inlet_name
    } )
    side_sets.append({
        "elements": new_outlet_elements.tolist(),
        "sides": new_outlet_sides.tolist(),
        "name": outlet_name
    })
    return side_sets
