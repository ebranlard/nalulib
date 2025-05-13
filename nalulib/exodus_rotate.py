import numpy as np
import matplotlib.pyplot as plt
import argparse
import os
import re
# Local
from nalulib.essentials import *
from nalulib.exodusii.file import ExodusIIFile


HEX_FACES = [
    [1, 5, 4, 0],  # 1 
    [1, 2, 6, 5],  # 2
    [3, 2, 6, 7],  # 3
    [0, 3, 7, 4],  # 4
    [0, 1, 2, 3],  # 5
    [4, 5, 6, 7],  # 6
]

QUAD_SIDES = [
    [0, 1],
    [1, 2],
    [2, 3],
    [3, 0]]

    #s = s.lower()

def extract_aoa(s):
    match = re.search(r'_aoa(-?\d+)', s, re.IGNORECASE)
    return int(match.group(1)) if match else None

def replace_aoa(s, new_value):
    if re.search(r'_aoa-?\d+', s, re.IGNORECASE):
        return re.sub(r'_aoa-?\d+', f'_aoa{new_value}', s, flags=re.IGNORECASE)
    else:
        return s + f'_aoa{new_value}'


def rotate_coordinates(coords, angle, center):
    """
    Rotate coordinates about the z-axis by a given angle and center.

    Parameters:
        coords (np.ndarray): Node coordinates (N x 2 or N x 3).
        angle (float): Rotation angle in radians.
        center (tuple): Center of rotation (x, y).

    Returns:
        np.ndarray: Rotated coordinates.
    """
    x_center, y_center = center
    # Subtract the center to shift the rotation origin
    x1 = coords[:, 0] - x_center
    y1 = coords[:, 1] - x_center

    # Apply the rotation matrix
    cos_angle = np.cos(angle)
    sin_angle = np.sin(angle)
    #x2 = cos_angle * x1 - sin_angle * y1 + x_center
    #y2 = sin_angle * x1 + cos_angle * y1 + y_center
    x2 =  cos_angle * x1 + sin_angle * y1 + x_center
    y2 = -sin_angle * x1 + cos_angle * y1 + y_center

    # Handle 3D case by preserving the z-coordinate if present
    if coords.shape[1] == 3:
        return np.column_stack((x2, y2, coords[:, 2]))
    else:
        return np.column_stack((x2, y2))

def compute_angles(coords, center):
    """
    Compute angles of nodes relative to the center.

    Parameters:
        coords (np.ndarray): Node coordinates (N x 3).
        center (tuple): Center of rotation (x, y).

    Returns:
        np.ndarray: Angles in radians.
    """
    x_center, y_center = center
    relative_coords = coords[:, :2] - np.array([x_center, y_center])
    angles = np.arctan2(relative_coords[:, 1], relative_coords[:, 0])
    return np.mod(angles, 2 * np.pi)  # Ensure angles are in [0, 2π]

def is_angle_in_segment(angles, start_angle, span, strict_upper=True):
    """
    Check if angles are within a given angular segment.

    Parameters:
        angles (np.ndarray): Angles to check (in radians, normalized to [0, 2π]).
        start_angle (float): Start angle of the segment (in radians, normalized to [0, 2π]).
        span (float): Angular span of the segment (in radians).
        strict_upper (bool): Whether to use a strict upper bound for the segment.

    Returns:
        np.ndarray: Boolean array indicating whether each angle is within the segment.
    """
    angles = np.mod(angles, 2 * np.pi)
    start_angle = np.mod(start_angle, 2 * np.pi)
    end_angle = np.mod(start_angle + span, 2 * np.pi)

    if strict_upper:
        if end_angle > start_angle:
            return (angles >= start_angle) & (angles < end_angle)
        else:  # Segment wraps around 0
            return (angles >= start_angle) | (angles < end_angle)
    else:
        if end_angle > start_angle:
            return (angles >= start_angle) & (angles <= end_angle)
        else:  # Segment wraps around 0
            return (angles >= start_angle) | (angles <= end_angle)

def angle_segment(angles):
    """
    Find the angular segment that contains all the given angles in the anti-clockwise direction.

    Parameters:
        angles (np.ndarray): Angles in radians (normalized to [0, 2π]).

    Returns:
        tuple: Start angle and span of the segment (in radians).
    """
    # Normalize angles to [0, 2π]
    angles = np.mod(angles, 2 * np.pi)
    angles = np.sort(angles)

    # Compute differences between consecutive angles, including wrap-around
    diffs = np.diff(np.concatenate((angles, [angles[0] + 2 * np.pi])))

    # Find the largest gap (clockwise direction)
    max_gap_idx = np.argmax(diffs)
    start_angle = angles[(max_gap_idx + 1) % len(angles)]  # Start angle is after the largest gap
    span = 2 * np.pi - diffs[max_gap_idx]  # Span is the complement of the largest gap

    return start_angle, span

def adjust_side_sets(side_sets, node_coords, angle_center, inlet_start, inlet_span, outlet_start, outlet_span, elem_to_face_nodes, debug=False):
    """
    Adjust inlet and outlet side sets after rotation.

    Parameters:
        side_sets (dict): Original side sets.
        node_coords (np.ndarray): Rotated node coordinates (N x 3).
        angle_center (tuple): Center for defining angles
        inlet_start (float): Start angle of the inlet segment (in radians).
        inlet_span (float): Angular span of the inlet segment (in radians).
        outlet_start (float): Start angle of the outlet segment (in radians).
        outlet_span (float): Angular span of the outlet segment (in radians).
        elem_to_face_nodes (dict): Mapping of (element ID, side ID) to node IDs.

    Returns:
        dict: Adjusted side sets.
    """
    # Combine inlet and outlet elements and sides
    combined_elements = []
    combined_sides = []
    for side_set_id, side_set_data in side_sets.items():
        if side_set_data["name"] in ["inlet", "outlet"]:
            combined_elements.extend(side_set_data["elements"])
            combined_sides.extend(side_set_data["sides"])

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
    adjusted_side_sets = {}
    for side_set_id, side_set_data in side_sets.items():
        if side_set_data["name"] == "inlet":
            adjusted_side_sets[side_set_id] = {
                "elements": new_inlet_elements.tolist(),
                "sides": new_inlet_sides.tolist(),
                "name": "inlet"
            }
        elif side_set_data["name"] == "outlet":
            adjusted_side_sets[side_set_id] = {
                "elements": new_outlet_elements.tolist(),
                "sides": new_outlet_sides.tolist(),
                "name": "outlet"
            }
        else:
            adjusted_side_sets[side_set_id] = side_set_data

    return adjusted_side_sets

def find_inlet_outlet_segments(side_sets, node_coords, center, elem_to_face_nodes, debug=False, verbose=True):
    """
    Find the angular segments for inlet and outlet nodes.

    Parameters:
        side_sets (dict): Side sets from the Exodus file.
        node_coords (np.ndarray): Node coordinates (N x 3).
        center (tuple): Center of rotation (x, y).
        elem_type (str): Element type ("HEX8" or "QUAD").

    Returns:
        tuple: Start angle and span for inlet and outlet segments.
    """
    # Identify inlet and outlet nodes
    inlet_nodes = []
    outlet_nodes = []
    for side_set_id, side_set_data in side_sets.items():
        if side_set_data["name"] == "inlet":
            for elem, side in zip(side_set_data["elements"], side_set_data["sides"]):
                inlet_nodes.extend(elem_to_face_nodes[(elem, side)])
        elif side_set_data["name"] == "outlet":
            for elem, side in zip(side_set_data["elements"], side_set_data["sides"]):
                outlet_nodes.extend(elem_to_face_nodes[(elem, side)])

    inlet_nodes = np.unique(inlet_nodes)
    outlet_nodes = np.unique(outlet_nodes)

    # Extract coordinates for inlet and outlet nodes
    inlet_coords = node_coords[inlet_nodes - 1]
    outlet_coords = node_coords[outlet_nodes - 1]

    # Compute angles for inlet and outlet nodes
    inlet_angles = compute_angles(inlet_coords, center)
    outlet_angles = compute_angles(outlet_coords, center)

    # Compute angular segments
    inlet_start, inlet_span = angle_segment(inlet_angles)
    outlet_start, outlet_span = angle_segment(outlet_angles)

    # Debugging: Print angles
    if debug:
        print('Inlet Angles (degrees):', np.degrees(inlet_angles))
        print('Outlet Angles (degrees):', np.degrees(outlet_angles))
        # Plot inlet and outlet coordinates
        plt.figure(figsize=(8, 8))
        plt.scatter(inlet_coords[:, 0], inlet_coords[:, 1], color='blue', label='Inlet Nodes')
        plt.plot(outlet_coords[:, 0], outlet_coords[:, 1], '+', color='red', label='Outlet Nodes')
        plt.scatter(center[0], center[1], color='green', marker='x', s=100, label='Center of Rotation')
        plt.xlabel('X')
        plt.ylabel('Y')
        plt.title('Inlet and Outlet Node Coordinates')
        plt.legend()
        plt.axis('equal')
        plt.grid(True)

    # Verify that outlet_span is close to  2 * np.pi - inlet_span 
    assert np.isclose(outlet_span, 2 * np.pi - inlet_span, atol= np.pi/20), "Outlet span is not complementary to inlet span!"

    return inlet_start, inlet_span, outlet_start, outlet_span

def rotate_exodus(input_file, output_file, angle, center, angle_center, inlet_start, inlet_span, outlet_start, outlet_span, keep_io_side_set, verbose, profiler=False):
    """
    Rotate an Exodus file's node coordinates and adjust side sets.

    Parameters:
        input_file (str): Path to the input Exodus file.
        output_file (str): Path to the output Exodus file.
        angle (float): Rotation angle in degrees.
        center (tuple): Center of rotation (x, y).
        angle_center (tuple): Center for angle estimation (x, y).
        inlet_start (float): Start angle of inlet segment (in radians).
        inlet_span (float): Angular span of inlet segment (in radians).
        outlet_start (float): Start angle of outlet segment (in radians).
        outlet_span (float): Angular span of outlet segment (in radians).
        keep_io_side_set (bool): Whether to keep inlet and outlet side sets unchanged.
        verbose (bool): Enable verbose output.
        profiler (bool): Enable profiling with timers.
    """
    angle_rad = np.radians(angle)


    with Timer("Opening Exodus file", silent=not profiler):
        print(f"Opening Exodus file: {input_file}")
        with ExodusIIFile(input_file, mode="r") as exo:
            node_coords = exo.get_coords()
            side_sets = {
                ss_id: {
                    "elements": exo.get_side_set(ss_id).elems,
                    "sides": exo.get_side_set(ss_id).sides,
                    "name": exo.get_side_set_name(ss_id)
                }
                for ss_id in exo.get_side_set_ids()
            }
            blk_id = exo.get_element_block_ids()[0]
            elem_conn = exo.get_element_conn(blk_id)
            elem_type = exo.get_element_block(blk_id).elem_type  # Assuming all blocks have the same type
    FACE_MAP = {'HEX': HEX_FACES, 'HEX8': HEX_FACES, 'QUAD': QUAD_SIDES}[elem_type]
    num_dim = {'HEX': 3, 'HEX8': 3, 'QUAD': 2}[elem_type]

    # Precompute a mapping of element ID and side ID to node IDs for inlet and outlet elements only
    with Timer("Precomputing element-to-face mapping", silent=not profiler):
        elem_to_face_nodes = {}
        for side_set_id, side_set_data in side_sets.items():
            if side_set_data["name"] in ["inlet", "outlet"]:
                for elem, side in zip(side_set_data["elements"], side_set_data["sides"]):
                    face_node_ids = elem_conn[elem - 1][FACE_MAP[side - 1]]  # Convert to 0-based indexing
                    elem_to_face_nodes[(elem, side)] = face_node_ids

    # Find inlet and outlet angular segments (before rotation)
    if inlet_start is None or inlet_span is None or outlet_start is None or outlet_span is None:
        with Timer("Finding inlet and outlet angular segments", silent=not profiler):
            inlet_start, inlet_span, outlet_start, outlet_span = find_inlet_outlet_segments(
                side_sets, node_coords, angle_center, elem_to_face_nodes, verbose=verbose
            )

    # Rotate node coordinates
    with Timer("Rotating node coordinates", silent=not profiler):
        if verbose:
            print('Rotating about center:', center, 'by angle:', angle)
        rotated_coords = rotate_coordinates(node_coords, angle_rad, center)

    # Adjust side sets if needed
    if not keep_io_side_set:
        if verbose:
            print(f"Angle center       : {angle_center}")
            print(f"Angle span, inlet  : Start={np.degrees(inlet_start):5.1f}°, Span={np.degrees(inlet_span):5.1f}° (anticlockwise)")
            print(f"Angle span, outlet : Start={np.degrees(outlet_start):5.1f}°, Span={np.degrees(outlet_span):5.1f}° (anticlockwise)")

        with Timer("Adjusting side sets", silent=not profiler):
            side_sets = adjust_side_sets(
                side_sets, rotated_coords, angle_center, inlet_start, inlet_span, outlet_start, outlet_span, elem_to_face_nodes
            )

    # Write rotated mesh to a new Exodus file

    if output_file is None:
        base_dir = os.path.dirname(input_file)       # '/home/user/data'
        base_name = os.path.basename(input_file)     # 'file_n10.exo'
        base, ext = os.path.splitext(base_name)  # name: 'file_n10', ext: '.exo'
        aoa = extract_aoa(base)
        print('>>> Old AoA: ', aoa)
        if aoa is None:
            base += '_rot'+int(angle)
        else:
            aoa_new = int(aoa + angle)
            print('>>> New AoA: ', aoa_new)
            base = replace_aoa(base, aoa_new)

        output_file = os.path.join(base_dir, base+ext)
        print('>>> outputfile', output_file )

    with Timer("Writing rotated Exodus file", silent=not profiler):
        with ExodusIIFile(output_file, mode="w") as exo_out:
            exo_out.put_init(
                title=f"Rotated Mesh by {angle}°",
                num_dim=num_dim,
                num_nodes=len(rotated_coords),
                num_elem=exo.num_elems(),
                num_elem_blk=exo.num_elem_blk(),
                num_node_sets=exo.num_node_sets(),
                num_side_sets=len(side_sets),
                double=True
            )
            if num_dim == 2:
                exo_out.put_coord(rotated_coords[:, 0], rotated_coords[:, 1])
                exo_out.put_coord_names(["X", "Y"])
            else:
                exo_out.put_coord(rotated_coords[:, 0], rotated_coords[:, 1], rotated_coords[:, 2])
                exo_out.put_coord_names(["X", "Y", "Z"])

            # Write element blocks
            for blk_id in exo.get_element_block_ids():
                blk_info = exo.get_element_block(blk_id)
                exo_out.put_element_block(blk_id, blk_info.elem_type, blk_info.num_block_elems, blk_info.num_elem_nodes)
                exo_out.put_element_conn(blk_id, exo.get_element_conn(blk_id))
                name = exo.get_element_block_name(blk_id)
                exo_out.put_element_block_name(blk_id, name)

            # Write side sets
            for side_set_id, side_set_data in side_sets.items():
                exo_out.put_side_set_param(side_set_id, len(side_set_data["elements"]))
                exo_out.put_side_set_sides(side_set_id, side_set_data["elements"], side_set_data["sides"])
                exo_out.put_side_set_name(side_set_id, side_set_data["name"])
    print(f"Written Exodus file: {output_file}")

def exo_rotate():
    """
    Command-line interface for rotating an Exodus file.
    """
    parser = argparse.ArgumentParser(description="Rotate an Exodus file's node coordinates about the z-axis.")
    parser.add_argument("input_file", type=str, help="Path to the input Exodus file.")
    parser.add_argument("-o", "--output_file", type=str, default=None, help="Path to the output Exodus file.")
    parser.add_argument("-a", "--angle", type=float, required=True, help="Rotation angle in degrees.")
    parser.add_argument("-c", "--center", type=float, nargs=2, default=(0.0, 0.0), help="Center of rotation (x, y). Default is (0.0, 0.0).")
    parser.add_argument("--angle-center", type=float, nargs=2, default=None, help="Center for angle estimation (x, y). Default is the same as --center.")
    parser.add_argument("--keep-io-side-set", action="store_true", help="Keep inlet and outlet side sets unchanged.")
    parser.add_argument("--bc-angles-half", action="store_true", help="Set inlet and outlet angles to predefined values (90°, 270° with 180° spans).")
    parser.add_argument("--bc-angles", type=float, nargs=2, metavar=("INLET_START", "INLET_SPAN"),
                        help="Define inlet start and inlet span (in degrees), anticlockwise.")
    parser.add_argument("--verbose", default=False, action="store_true", help="Enable verbose output.")
    parser.add_argument("--profiler", action="store_true", help="Enable profiling with timers.")

    args = parser.parse_args()


    # Determine angle center
    angle_center = tuple(args.angle_center) if args.angle_center else tuple(args.center)

    # Determine inlet and outlet angles
    if args.bc_angles_half:
        inlet_start = np.radians(90)
        inlet_span = np.radians(180)
        outlet_start = np.radians(270)
        outlet_span = np.radians(180)
    elif args.bc_angles:
        inlet_start = np.radians(args.bc_angles[0])
        inlet_span = np.radians(args.bc_angles[1])
        outlet_start = np.radians(args.bc_angles[0] + args.bc_angles[1])
        outlet_span = 2 * np.pi - inlet_span  # Infer outlet span
    else:
        inlet_start = None
        inlet_span = None
        outlet_start = None
        outlet_span = None

    rotate_exodus(
        input_file=args.input_file,
        output_file=args.output_file,
        angle=args.angle,
        center=tuple(args.center),
        angle_center=angle_center,
        inlet_start=inlet_start,
        inlet_span=inlet_span,
        outlet_start=outlet_start,
        outlet_span=outlet_span,
        keep_io_side_set=args.keep_io_side_set,
        verbose=args.verbose,
        profiler=args.profiler
    )

if __name__ == "__main__":
    exo_rotate()
    plt.show()
