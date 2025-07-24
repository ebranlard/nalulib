import numpy as np
import matplotlib.pyplot as plt
import argparse
import os
import re
# Local
from nalulib.essentials import *
from nalulib.exodusii.file import ExodusIIFile
from nalulib.angles import is_angle_in_segment, compute_angles, angle_segment
from nalulib.exodus_core import HEX_FACES, QUAD_SIDES
from nalulib.exodus_core import HEX_FACES, QUAD_SIDES, ELEMTYPE_2_NUM_DIM, ELEMTYPE_2_SIDE
from nalulib.exodus_core import check_exodus_side_sets_exist, set_omesh_inlet_outlet_ss



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
    x2 =  cos_angle * x1 + sin_angle * y1 + x_center
    y2 = -sin_angle * x1 + cos_angle * y1 + y_center

    # Handle 3D case by preserving the z-coordinate if present
    if coords.shape[1] == 3:
        return np.column_stack((x2, y2, coords[:, 2]))
    else:
        return np.column_stack((x2, y2))

def inlet_outlet_angle_segments(side_sets, node_coords, center, elem_to_face_nodes, 
                                inlet_name='inlet', outlet_name='outlet',
                                debug=False, verbose=True):
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
        if side_set_data["name"] == inlet_name:
            for elem, side in zip(side_set_data["elements"], side_set_data["sides"]):
                inlet_nodes.extend(elem_to_face_nodes[(elem, side)])
        elif side_set_data["name"] == outlet_name:
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

    return inlet_start, inlet_span, outlet_start

def exo_rotate(input_file, output_file, angle, center=(0,0), angle_center=None, 
                  inlet_start=None, inlet_span=None, outlet_start=None, keep_io_side_set=False, 
                  inlet_name='inlet', outlet_name='outlet',
                  verbose=False, profiler=False, debug=False):
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
        keep_io_side_set (bool): Whether to keep inlet and outlet side sets unchanged.
        verbose (bool): Enable verbose output.
        profiler (bool): Enable profiling with timers.
    """

    # --- Default arguments
    if center is None:
        center =(0.0, 0.0)
    # Determine angle center
    angle_center = tuple(angle_center) if angle_center else tuple(center)



    # --- Open the baseline exodus file and precompute necessary data
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
            # 
            check_exodus_side_sets_exist(exo, [inlet_name, outlet_name])


    FACE_MAP = ELEMTYPE_2_SIDE[elem_type]
    num_dim = ELEMTYPE_2_NUM_DIM[elem_type]

    # Precompute a mapping of element ID and side ID to node IDs for inlet and outlet elements only
    with Timer("Precomputing element-to-face mapping", silent=not profiler):
        elem_to_face_nodes = {}
        for side_set_id, side_set_data in side_sets.items():
            if side_set_data["name"] in [inlet_name, outlet_name]:
                for elem, side in zip(side_set_data["elements"], side_set_data["sides"]):
                    face_node_ids = elem_conn[elem - 1][FACE_MAP[side - 1]]  # Convert to 0-based indexing
                    elem_to_face_nodes[(elem, side)] = face_node_ids

    # Find inlet and outlet angular segments (before rotation)
    if not keep_io_side_set:
        if inlet_start is None or inlet_span is None or outlet_start is None:
            with Timer("Finding inlet and outlet angular segments", silent=not profiler):
                inlet_start, inlet_span, outlet_start  = inlet_outlet_angle_segments(side_sets, node_coords, angle_center, elem_to_face_nodes, 
                    inlet_name=inlet_name, outlet_name=outlet_name, debug=debug, verbose=verbose)
    
    if hasattr(angle, '__len__'):
        angles = angle
        if len(output_file) != len(angles):
            raise ValueError("Output file list must match the number of angles.")
        output_files = output_file
    else:
        angles = [angle]
        output_files = [output_file]

    # --- Loop on different angles
    for i, (angle, output_file) in enumerate(zip(angles, output_files)):
        if verbose:
            print('Rotating about center:', center, 'by angle:', angle, '(output file: {output_file})')

        # Rotate node coordinates
        with Timer("Rotating node coordinates", silent=not profiler):
            rotated_coords = rotate_coordinates(node_coords, np.radians(angle), center)
        
        if debug:
            blk_id = exo.get_element_block_ids()[0]
            blk_info = exo.get_element_block(blk_id)
            conn = exo.get_element_conn(blk_id)
            if blk_info.elem_type.lower() in ['quad']:
                from nalulib.exodus_core import negative_quads
                import matplotlib.pyplot as plt
                Ineg1 = negative_quads(conn, node_coords, ioff=1, plot=False)
                print(f"Negative quads before rotation: {len(Ineg1)} - len(conn)", len(conn))
                Ineg2 = negative_quads(conn, rotated_coords, ioff=1, plot=True, old_coords=node_coords)
                print(f"Negative quads after rotation: {len(Ineg2)}")
                if len(Ineg2)>0:
                    plt.show()

        # Adjust side sets if needed
        if not keep_io_side_set:
            if verbose:
                print('[INFO] Adjusting side sets...')
                print(f"Angle center       : {angle_center}")
                print(f"Angle span, inlet  : Start={np.degrees(inlet_start):5.1f}°, Span={np.degrees(inlet_span):5.1f}° (anticlockwise)")
                print(f"Angle span, outlet : Start={np.degrees(outlet_start):5.1f}°, Span={np.degrees(2*np.pi-inlet_span):5.1f}° (anticlockwise)")

            with Timer("Adjusting side sets", silent=not profiler):
                # Combine inlet and outlet elements and sides
                combined_elements = []
                combined_sides = []
                for side_set_id, side_set_data in side_sets.items():
                    if side_set_data["name"] in [inlet_name, outlet_name]:
                        combined_elements.extend(side_set_data["elements"])
                        combined_sides.extend(side_set_data["sides"])
                combined_elements = np.array(combined_elements)
                combined_sides = np.array(combined_sides)
                # Find the new inlet and outlet side sets based on angular position
                new_side_sets = set_omesh_inlet_outlet_ss(rotated_coords, combined_elements, combined_sides, elem_to_face_nodes, angle_center, inlet_start, inlet_span, outlet_start, inlet_name=inlet_name, outlet_name=outlet_name)
                for side_set_id, side_set_data in side_sets.items():
                    if side_set_data["name"] in [inlet_name, outlet_name]:
                        for new_side_set in new_side_sets:
                            if side_set_data["name"] == new_side_set["name"]:
                                side_sets[side_set_id] = new_side_set
                                break

        # Write rotated mesh to a new Exodus file
        if output_file is None:
            # If user does not specify output file name, we look for _aoaXXX, if found, we replace it.
            base_dir = os.path.dirname(input_file)
            base_name = os.path.basename(input_file)
            base, ext = os.path.splitext(base_name)
            aoa = extract_aoa(base)
            if aoa is None:
                base += '_aoa'+str(int(angle))
            else:
                aoa_new = int(aoa + angle)
                base = replace_aoa(base, aoa_new)

            output_file = os.path.join(base_dir, base+ext)

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

def exo_rotate_CLI():
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
    parser.add_argument("-v", "--verbose", default=False, action="store_true", help="Enable verbose output.")
    parser.add_argument("--profiler", action="store_true", help="Enable profiling with timers.")
    parser.add_argument("--inlet-name", type=str, default="inlet", help="Name for the inlet sideset (default: 'inlet'. alternative: 'inflow').")
    parser.add_argument("--outlet-name", type=str, default="outlet", help="Name for the outlet sideset (default: 'outlet'. alternative: 'outflow').")

    args = parser.parse_args()



    # Determine inlet and outlet angles
    if args.bc_angles_half:
        inlet_start = np.radians(90)
        inlet_span = np.radians(180)
        outlet_start = np.radians(270)
    elif args.bc_angles:
        inlet_start = np.radians(args.bc_angles[0])
        inlet_span = np.radians(args.bc_angles[1])
        outlet_start = np.radians(args.bc_angles[0] + args.bc_angles[1])
    else:
        inlet_start = None
        inlet_span = None
        outlet_start = None

    exo_rotate(
        input_file=args.input_file,
        output_file=args.output_file,
        angle=args.angle,
        center=tuple(args.center),
        angle_center=args.angle_center,
        inlet_start=inlet_start,
        inlet_span=inlet_span,
        outlet_start=outlet_start,
        keep_io_side_set=args.keep_io_side_set,
        inlet_name=args.inlet_name,
        outlet_name=args.outlet_name,
        verbose=args.verbose,
        profiler=args.profiler
    )

if __name__ == "__main__":
    exo_rotate_CLI()
    plt.show()
