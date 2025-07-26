import numpy as np
import sys
import os
# Local
from nalulib.essentials import *
from nalulib.exodusii.file import ExodusIIFile
from nalulib.exodus_core import HEX_FACES, QUAD_SIDES, write_exodus_quads
from nalulib.exodus_core import quad_is_positive_about_z


# TODO
SIDE_SETS_EXCLUDE=['front', 'back','wing-pp', 'wing_pp', 'front_bg', 'back_bg']

def exo_hex_to_quads(exo, z_ref=None, verbose=True, debug=False, profiler=False, check_zpos=True):
    # Read basic information
    num_nodes = exo.num_nodes()
    num_elems = exo.num_elems()
    node_coords = exo.get_coords()
    element_blocks = exo.get_element_block_ids()

    # Analyze z values
    z_values = [coord[2] for coord in node_coords]
    unique_z_values = np.unique(z_values)
    # Determine z_ref if not provided
    if z_ref is None:
        z_ref = node_coords[0][2]  # Use the Z value of the first node
    # Find nodes in the z_ref plane
    nodes_id_in_plane = np.where(np.isclose(node_coords[:, 2], z_ref))[0] + 1 # +1 for 1-based indexing

    # --- Debug outputs, well aligned
    if verbose:
        print(f"Number of blocks         : {len(element_blocks)}")
        print(f"Number of nodes          : {len(node_coords)}")
        print(f"Number of unique z values: {len(unique_z_values)}")
        print(f"Range of z values        : {min(z_values)} to {max(z_values)}")
        print(f"First few z values       : {unique_z_values[:3]}")
        print(f"Value of  z_ref          : {z_ref}")
        print(f"Nodes in the z_ref plane : {len(nodes_id_in_plane)}")
    if len(nodes_id_in_plane) == 0:
        raise ValueError(f"No nodes found in the z_ref plane at z={z_ref}. Please check the input file.")
    if len(element_blocks) > 1:
        raise NotImplementedError("This script only supports one block for get_element_id_map.")
    block_id = element_blocks[0]
    block_info = exo.get_element_block(block_id)
    if block_info.elem_type not in ["HEX8", "HEX"]:
        raise ValueError(f"Unsupported element type: {block_info.elem_type}. Only HEX8 or HEX are supported.")

    # --- Step 1: Generate the list of unique quads and set up the quad2hex dictionary
    quads = []  # List of unique quads
    quad2hex = {}  # Map each quad to its corresponding HEX element
    sorted_quads_set = set()  # Set to track sorted quads for uniqueness
    elem_conn = np.array(exo.get_element_conn(block_id))  # Convert to NumPy array for slicing - IDs start at 1
    elem_ids = exo.get_element_id_map()  # Get the true element IDs for this block

    # Pre-filter elements that have at least one node in the z_ref plane
    with Timer('Filtering elements', silent=not profiler):
        mask = np.isin(elem_conn, nodes_id_in_plane).any(axis=1)
        filtered_elem_conn = elem_conn[mask] # Connectivity IDs starts at 1
        filtered_elem_ids = elem_ids[mask]   # Filter the element IDs accordingly, IDs start at 1
        if verbose:
            print(f"Number of hex            : {len(elem_conn)}")
            print(f"Number of hex close to z : {len(filtered_elem_conn)}")

    # Precompute face z-coordinates for all elements
    with Timer('Face coords', silent=not profiler, writeBefore=True):
        face_indices = np.array(HEX_FACES)  # Convert HEX_FACES to a NumPy array
        all_faces = filtered_elem_conn[:, face_indices]  # Shape: (num_elements, num_faces, num_nodes_per_face)
        all_face_z_coords = node_coords[all_faces - 1, 2]  # Get z-coordinates of all face nodes

    with Timer('Identifying faces', silent=not profiler, writeBefore=True):
        # Identify faces that lie on the z_ref plane
        faces_on_plane = np.all(np.isclose(all_face_z_coords, z_ref), axis=2)  # Shape: (num_elements, num_faces)
        # Flatten the data for easier iteration
        elements_with_faces = np.argwhere(faces_on_plane)  # Indices of (element, face) pairs

    # Process each face that lies on the z_ref plane
    with Timer('Processsing faces', silent=not profiler, writeBefore=True):
        nOrientWrong = 0
        for elem_idx, face_idx in elements_with_faces:
            face_nodes_ids = all_faces[elem_idx, face_idx] # IDs start at 1
            sorted_face = tuple(sorted(face_nodes_ids)) # To ensure uniqueness, we sort and create a tuple
            if sorted_face not in sorted_quads_set:
                if not quad_is_positive_about_z(face_nodes_ids, node_coords, ioff=1) and check_zpos:
                    nOrientWrong += 1
                    if debug:
                        print('[WARN] Face is oriented negatively about Z, reorienting it:', face_nodes_ids)
                    face_nodes_ids = [face_nodes_ids[3], face_nodes_ids[2], face_nodes_ids[1], face_nodes_ids[0]]  # Reverse the order
                # Add the original face to the quads list
                quad = {
                    'node_ids': face_nodes_ids,
                    'quad_id': len(quads) + 1,
                    'hex_id': filtered_elem_ids[elem_idx],
                    'hex_node_ids': filtered_elem_conn[elem_idx],
                    'hex_face_index': face_idx
                }
                quads.append(quad)
                sorted_quads_set.add(sorted_face)

                if debug:
                    # Show the node IDs and their coordinates
                    node_coords_str = ', '.join(
                        [f"{node} ({node_coords[node-1][0]:.2f}, {node_coords[node-1][1]:.2f}, {node_coords[node-1][2]:.2f})"
                        for node in face_nodes_ids]
                    )
                    print(f"  Face {face_idx} of element {filtered_elem_ids[elem_idx]}: {node_coords_str}")
        if nOrientWrong > 0:
            print(f"[WARN] {nOrientWrong}/{len(elements_with_faces)} faces were oriented negatively about Z and have been reoriented.")

    #if check_zpos:
    #    conn= force_quad_positive_about_z(conn, coords, verbose=True)

    # --- Step 2: Handle side sets
    with Timer('Matching side sets', silent=not profiler, writeBefore=True):
        side_sets = {}

        # Precompute a mapping of HEX IDs to side set indices
        hex_to_side_set = {}
        for side_set_id in exo.get_side_set_ids():
            old_side_set = exo.get_side_set(side_set_id)
            if exo.get_side_set_name(side_set_id) not in SIDE_SETS_EXCLUDE:
                for hex_id, side in zip(old_side_set.elems, old_side_set.sides):
                    if hex_id not in hex_to_side_set:
                        hex_to_side_set[hex_id] = []
                    hex_to_side_set[hex_id].append((side_set_id, side))

        # Create side sets
        for side_set_id in exo.get_side_set_ids():
            old_side_set = exo.get_side_set(side_set_id)
            if exo.get_side_set_name(side_set_id) not in SIDE_SETS_EXCLUDE:
                side_sets[int(side_set_id)] = {
                    "elements": [],
                    "sides": [],
                    "name": exo.get_side_set_name(side_set_id),
                    "old_id": side_set_id,
                }

        # Group quads by HEX ID
        quads_by_hex_id = {}
        for quad in quads:
            if quad['hex_id'] not in quads_by_hex_id:
                quads_by_hex_id[quad['hex_id']] = []
            quads_by_hex_id[quad['hex_id']].append(quad)

        # Match quads to side sets
        for hex_id, quad_list in quads_by_hex_id.items():
            if hex_id in hex_to_side_set:
                for side_set_id, hex_side in hex_to_side_set[hex_id]:
                    for quad in quad_list:
                        # Determine the side of the quad
                        hex_face_node_ids_in_ss = quad['hex_node_ids'][HEX_FACES[hex_side - 1]]
                        quad_nodes_ids = quad['node_ids']

                        # Find the union of the two sets
                        union_nodes = set(hex_face_node_ids_in_ss).intersection(set(quad_nodes_ids))

                        # Determine the side of the quad using QUAD_SIDES
                        quad_side = None
                        for side_index, side_nodes in enumerate(QUAD_SIDES, start=1):
                            if all(quad_nodes_ids[node] in union_nodes for node in side_nodes):
                                quad_side = side_index
                                break

                        if quad_side is None:
                            raise ValueError(f"Could not determine the side for quad {quad['quad_id']}.")

                        # Add the quad to the side set
                        side_sets[int(side_set_id)]["elements"].append(quad['quad_id'])
                        side_sets[int(side_set_id)]["sides"].append(quad_side)

                        if debug:
                            print(f"  Quad ID {quad['quad_id']} added to side set {side_set_id} with side {quad_side}.")
    if verbose:
        print(f"Number of quads          : {len(quads)}")
        print(f"Number of side sets      : {len(side_sets)}")
    # Display side set information
    for side_set_id, side_set_data in side_sets.items():
        if verbose:
            print(f"   Side Set ID: {side_set_id}, Name: {side_set_data['name']}, nElements: {len(side_set_data['elements'])}, Side: {np.unique(side_set_data['sides'])}")
        if debug:
            print(f"    Elements: {side_set_data['elements'][:5]} ({len(side_set_data['elements'])}, {len(np.unique(side_set_data['elements']))})")
            print(f"    Sides   : {side_set_data['sides'][:5]} ({len(side_set_data['sides'])}, {len(np.unique(side_set_data['sides']))})")
    return quads, side_sets, node_coords, z_ref

def reindex_quads(quads, old_node_coords, side_sets, z_ref, verbose=False, profiler=False):
    # Step 1.1: Re-ID the nodes and update the quads
    with Timer('Re-iding nodes', silent=not profiler, writeBefore=True):
        old_node_ids = np.asarray(sorted(set(int(old_node_id) for quad in quads for old_node_id in quad['node_ids'])))  # Collect all unique nodes
        old_node_id_2_new_node_id = {old_node_id: new_id + 1 for new_id, old_node_id in enumerate(old_node_ids)}  # Assign new IDs
        nodes_coords = old_node_coords[old_node_ids-1,:] 
        # Check that all coords are indeed close to z_ref
        if not np.all(np.isclose(nodes_coords[:, 2], z_ref)):
            raise ValueError(f"Node coordinates do not match the reference Z value {z_ref}. Please check the input file.")
        nodes_coords[:,2] = 0.0  # Set Z coordinate to zero for the plane
        # Step 1.2 - Update the quads with the new node IDs
        for quad in quads:
            quad['node_ids'] = [old_node_id_2_new_node_id[old_node_id] for old_node_id in quad['node_ids']]
        # Delete maps that do now exist anymore
        del old_node_id_2_new_node_id

        # Step 2.1 reindex the side_sets (Optional)
        new_side_sets = {}
        for new_id, (side_set_id, side_set_data) in enumerate(side_sets.items()):   
            new_side_sets[new_id+1] = side_set_data
    return quads, nodes_coords, new_side_sets

def exo_flatten(input_file, output_file=None, z_ref=None, verbose=True, profiler=False, check_zpos=True):
    """
    Extract a plane of HEX elements at a specific Z value, convert to QUADs, and export to a new Exodus file.

    Parameters:
        input_file (str): Path to the input Exodus file.
        output_file (str): Path to the output Exodus file. Defaults to 'plane_quads.exo'.
        z_ref (float): Reference Z value for the plane. If None, the Z value of the first node is used.
        verbose (bool): If True, print detailed information during processing.
    """
    if output_file is None:
        output_file = rename_n(input_file, nSpan=1)

    print( 'Opening Exodus file      :', input_file)
    # Read exodus and extrad quads at z_ref
    with ExodusIIFile(input_file, mode="r") as exo:
        block_id = exo.get_element_block_ids()[0]
        block_info = exo.get_element_block(block_id)
        block_name_in = block_info.name
        quads, side_sets, node_coords, z_ref = exo_hex_to_quads(exo, z_ref=z_ref, verbose=verbose, profiler=profiler, check_zpos=check_zpos)

    # Reindex
    quads, nodes_coords, side_sets = reindex_quads(quads, node_coords, side_sets, z_ref, verbose=verbose, profiler=profiler)

    # Write to file
    title=f"Plane QUADs at z={z_ref}",
    conn = np.array([[node_id for node_id in quad['node_ids']] for quad in quads])
    block_name = 'fluid-quad'
    if block_name_in is not None:
        if block_name_in.lower().endswith('-hex'):
            block_name= block_name_in.replace('-hex','-quad').replace('-HEX','-quad')
    write_exodus_quads(output_file, nodes_coords, conn, title=title, side_sets=side_sets, block_name=block_name, verbose=True, profiler=profiler)
    return output_file


def exo_flatten_CLI():
    """
    Command-line interface for extracting a plane of HEX elements at a specific Z value, converting to QUADs, and exporting to a new Exodus file.
    """
    import argparse
    parser = argparse.ArgumentParser(description="Extract a plane of HEX elements at a specific Z value, convert to QUADs, and export to a new Exodus file.")
    parser.add_argument("input_file", type=str, help="Path to the input Exodus file containing the HEX mesh.")
    parser.add_argument("-o", "--output_file", metavar="Output_file", dest="output", type=str, default=None,
                        help="Path to the output Exodus file. Defaults to '<input_file>_n1.exo'.")
    parser.add_argument("-z", metavar="Z_ref", type=float, default=None,
                        help="Reference Z value for the plane. If None, the Z value of the first node is used.")
    parser.add_argument("-v", "--verbose", action="store_true", help="Enable verbose output.")
    parser.add_argument("--profiler", action="store_true", help="Enable profiler timing .")
    parser.add_argument("--no-zpos-check", action="store_true", help="Do not check and enforce the orientation of quads about Z axis (default: check).")

    args = parser.parse_args()

    with Timer('hex_to_quads', silent = not args.profiler):
        exo_flatten(
            input_file=args.input_file,
            output_file=args.output,
            z_ref=args.z,
            check_zpos= not args.no_zpos_check,
            verbose=args.verbose,
            profiler=args.profiler
        )


if __name__ == "__main__":
    exo_flatten_CLI

    #input_file =  "ffa_w3_211_aoa32.exo"
    #output_file = None
    #z_ref = None
    #with Timer('hex2quads'):
    #    hex_to_quads_plane(input_file, output_file, z_ref, verbose=False)

