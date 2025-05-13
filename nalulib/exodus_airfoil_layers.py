import numpy as np
import matplotlib.pyplot as plt
import sys
import os
# Local
from nalulib.exodus_hex2quads import hex_to_quads_plane  # Import the function to convert HEX to QUADS
from nalulib.meshlib import create_quadrilateral_cells, save_mesh, open_mesh_in_gmsh
from nalulib.exodusii.file import ExodusIIFile
from nalulib.essentials import *

# Define the faces of a HEX8 element
HEX_FACES = [
    [1, 5, 4, 0],  # 1
    [1, 2, 6, 5],  # 2
    [3, 2, 6, 7],  # 3
    [0, 3, 7, 4],  # 4
    [0, 1, 2, 3],  # 5
    [4, 5, 6, 7],  # 6
]

QUAD_SIDES = [
    [0, 1], #1
    [1, 2], #2
    [2, 3], #3
    [3, 0], #4
]

SIDE_SETS_NAMES = ['wing', 'wing_pp', 'airfoil']


def find_neighbor_element(elem_id, side_index, elem_conn, elem_to_sides):
    """
    Given an element ID and a side index, find the neighboring element that shares the side.

    Parameters:
        exo (ExodusIIFile): Opened ExodusIIFile object.
        elem_id (int): The current element ID.
        side_index (int): The side index of the current element.
        elem_conn (np.ndarray): Connectivity array for all elements.
        elem_to_sides (dict): Precomputed mapping of element sides to their node sets.

    Returns:
        tuple: (neighbor_elem_id, neighbor_side_index) or (None, None) if no neighbor is found.
    """
    # Get the nodes of the current side
    current_side_nodes = elem_to_sides[elem_id][side_index-1]

    # Loop through all elements to find the neighbor
    for neighbor_elem_id in range(len(elem_conn)):
        neighbor_elem_id += 1  # Adjust for 1-based indexing
        if neighbor_elem_id == elem_id:
            continue  # Skip the current element

        # Check all sides of the neighbor element
        for neighbor_side_index, neighbor_side_nodes in enumerate(elem_to_sides[neighbor_elem_id]):
            if set(current_side_nodes) == set(neighbor_side_nodes):
                return neighbor_elem_id, neighbor_side_index+1

    return None, None  # No neighbor found


def find_layer_coords(current_layer_elem_ids, current_layer_elem_sides, node_coords, elem_conn, elem_to_sides):
    """
    Compute the coordinates of the current layer.

    Parameters:
        current_layer_elem_ids (list): List of element IDs in the current layer.
        current_layer_elem_sides (list): List of side indices for the current layer elements.
        node_coords (np.ndarray): Array of node coordinates.
        elem_conn (np.ndarray): Connectivity array for all elements.
        elem_to_sides (dict): Precomputed mapping of element sides to their node sets.

    Returns:
        np.ndarray: Coordinates of the current layer.
    """
    current_layer_coords = []
    previous_side_nodes = None

    for i, (elem_id, side_index) in enumerate(zip(current_layer_elem_ids, current_layer_elem_sides)):
        current_side_nodes = elem_to_sides[elem_id][side_index - 1]

        # Sanity check: Ensure the current side shares a node with the previous side
        if i > 0:
            common_nodes = set(previous_side_nodes).intersection(set(current_side_nodes))
            if len(common_nodes) != 1:
                raise ValueError(f"Sanity check failed: Element {elem_id} does not share exactly one node with the previous element.")

        # Add the nodes to the current layer coordinates
        if i == 0:
            current_layer_coords.append(node_coords[current_side_nodes[0] - 1])
            current_layer_coords.append(node_coords[current_side_nodes[1] - 1])
        else:
            unique_node = [node for node in current_side_nodes if node not in previous_side_nodes][0]
            current_layer_coords.append(node_coords[unique_node - 1])

            # Special handling for the second element
            if i == 1:
                common_node = list(set(previous_side_nodes).intersection(set(current_side_nodes)))[0]
                if current_layer_coords[0][0] == node_coords[common_node - 1][0] and current_layer_coords[0][1] == node_coords[common_node - 1][1]:
                    current_layer_coords[0], current_layer_coords[1] = current_layer_coords[1], current_layer_coords[0]

        previous_side_nodes = current_side_nodes

    return np.array(current_layer_coords)


def find_next_layer(current_layer_elem_ids, current_layer_elem_sides, elem_to_sides, side_to_element_map):
    """
    Compute the next layer's element IDs and sides.

    Parameters:
        current_layer_elem_ids (list): List of element IDs in the current layer.
        current_layer_elem_sides (list): List of side indices for the current layer elements.
        elem_to_sides (dict): Precomputed mapping of element sides to their node sets.
        side_to_element_map (dict): Precomputed mapping of sides to elements.

    Returns:
        tuple: (next_layer_elem_ids, next_layer_elem_sides)
    """
    next_layer_elem_ids = []
    next_layer_elem_sides = []

    for elem_id, side_index in zip(current_layer_elem_ids, current_layer_elem_sides):
        # Get the nodes of the opposite side
        opposite_side_index = (side_index - 1 + 2) % 4 + 1
        opposite_side_nodes = elem_to_sides[elem_id][opposite_side_index - 1]
        side_key = tuple(sorted(opposite_side_nodes))

        # Find the neighboring element and side
        neighbors = side_to_element_map.get(side_key, [])
        for neighbor_elem_id, neighbor_side_index in neighbors:
            if neighbor_elem_id != elem_id:  # Skip the current element
                next_layer_elem_ids.append(neighbor_elem_id)
                next_layer_elem_sides.append(neighbor_side_index)
                break

    return next_layer_elem_ids, next_layer_elem_sides


def extract_airfoil_geometry(input_file, side_set_name, num_layers, output_prefix='', verbose=True, write_airfoil=False, plot=False, gmsh_write=False, gmsh_open=False, write_layers=False, write_fig=False, profiler=False):
    """
    Extract airfoil geometry and surrounding layers from an Exodus file.

    Parameters:
        exo (ExodusIIFile): Opened ExodusIIFile object.
        side_set_name (str): Name of the side set defining the airfoil. If None, search for 'airfoil', 'wing', or 'wing_pp'.
        num_layers (int): Number of layers to extract.
        output_prefix (str): Prefix for output files.
        verbose (bool): If True, print detailed information.
        gmsh_open (bool): If True, open the generated .msh file in GMSH.
    """
    basefilename = os.path.splitext(input_file)[0]


    print( 'Opening Exodus file      :', input_file)
    with ExodusIIFile(input_file, mode="r") as exo:
        # Determine the side set name if not provided
        if side_set_name is None:
            for candidate in ['airfoil', 'wing', 'wing_pp']:
                for ss_id in exo.get_side_set_ids():
                    if exo.get_side_set_name(ss_id) == candidate:
                        side_set_name = candidate
                        break
                if side_set_name is not None:
                    break
            if side_set_name is None:
                raise ValueError("No suitable side set found. Tried 'airfoil', 'wing', and 'wing_pp'.")
        #  Check if the mesh contains HEX elements and convert to QUADS if necessary
        element_blocks = exo.get_element_block_ids()
        block_id = element_blocks[0]
        block_info = exo.get_element_block(block_id)
        conversion_needed = block_info.elem_type in ["HEX8", "HEX"]
    if conversion_needed:
        #if verbose:
        print('--------------------------------- HEX2QUADS --------------------------------------')
        print("Converting HEX elements to QUADS...")
        output_file = basefilename + "_quads_tmp.exo"
        input_file = hex_to_quads_plane(input_file, output_file=output_file, z_ref=None, verbose=verbose)
        #if verbose:
        print('----------------------------------------------------------------------------------')
        print( 'Opening Exodus file      :', input_file)
    with ExodusIIFile(input_file, mode="r") as exo:
        # Get basic information
        node_coords = exo.get_coords()

        # Find the side set ID
        side_set_id = None
        for ss_id in exo.get_side_set_ids():
            if exo.get_side_set_name(ss_id) == side_set_name:
                side_set_id = ss_id
                break
        if side_set_id is None:
            raise ValueError(f"Side set '{side_set_name}' not found in the Exodus file.")

        print(f"Side set for airfoil     : '{side_set_name}' (ID: {side_set_id}).")

        side_set = exo.get_side_set(side_set_id)
        element_blocks = exo.get_element_block_ids()
        block_id = element_blocks[0]

        # --- Precomputations
        elem_conn = np.array(exo.get_element_conn(block_id))  # Element connectivity
        node_coords = exo.get_coords()
        # --- DONE WITH EXO
        #exo.close()

    with Timer('Precomputations', silent = not profiler):
        # Precompute element sides and their node sets
        elem_to_sides = {}
        for elem_id, conn in enumerate(elem_conn):
            elem_to_sides[elem_id + 1] = [conn[side] for side in QUAD_SIDES]

        # Precompute a mapping of sides to elements
        side_to_element_map = {}
        for elem_id, sides in elem_to_sides.items():
            for side_index, side_nodes in enumerate(sides, start=1):
                side_key = tuple(sorted(side_nodes))  # Use sorted nodes as the key
                if side_key not in side_to_element_map:
                    side_to_element_map[side_key] = []
                side_to_element_map[side_key].append((elem_id, side_index))

    # --- Extract layers
    next_layer_elem_ids = side_set.elems
    next_layer_elem_sides = side_set.sides
    layers = []

    with Timer('Finding fronts', silent = not profiler):
        print('--- Finding fronts:')
        for layer_idx in range(0, num_layers):
            current_layer_elem_ids = next_layer_elem_ids
            current_layer_elem_sides = next_layer_elem_sides

            # Compute current layer coordinates
            current_layer_coords = find_layer_coords(current_layer_elem_ids, current_layer_elem_sides, node_coords, elem_conn, elem_to_sides)
            layers.append(current_layer_coords)

            # Compute diagnostics for the current layer
            if len(current_layer_coords) > 1:
                distances_between_nodes = np.linalg.norm(np.diff(current_layer_coords, axis=0), axis=1)
                min_spacing = np.min(distances_between_nodes)
                max_spacing = np.max(distances_between_nodes)
                avg_spacing = np.mean(distances_between_nodes)
                print(f"Front {layer_idx:4d}: ds:(min={min_spacing:.6f}, avg={avg_spacing:.6f}, max={max_spacing:.6f})")

            # Compute next layer
            next_layer_elem_ids, next_layer_elem_sides = find_next_layer( current_layer_elem_ids, current_layer_elem_sides, elem_to_sides, side_to_element_map)

            if len(next_layer_elem_ids) == 0:
                print("[WARN] No more layers found. Stopping extraction.")
                break
        
    # --- Add the last layer based on the opposide side
    opposite_side_index = np.mod((np.asarray(current_layer_elem_sides) - 1 + 2) , 4) + 1
    final_layer_coords = find_layer_coords(current_layer_elem_ids, opposite_side_index, node_coords, elem_conn, elem_to_sides)
    layers.append(final_layer_coords)

    # --- Check if layer 0 is looped
    is_looped = np.allclose(layers[0][0], layers[0][-1])
    if is_looped:
        # Remove the last coordinate of each layer
        layers = [layer[:-1] for layer in layers]

    # --- Reorganize the first layer to start at the trailing edge
    trailing_edge_idx = np.argmin(np.linalg.norm(layers[0] - np.array([1, 0]), axis=1))
    layers = [np.roll(layer, -trailing_edge_idx, axis=0) for layer in layers]

    # Ensure the loop is closed after reindexing
    if is_looped:
        layers = [np.vstack([layer, layer[0]]) for layer in layers]

    # --- Compute distances between airfoil and first layer
    with Timer('Diagnostics', silent = not profiler):
        print('--- Layer diagnostics:')
        cumulative_avg_height = 0  # Initialize cumulative average height

        # Compute diagnostics between layers
        for i in range(len(layers) - 1):
            distances_to_next_layer = np.linalg.norm(layers[i + 1] - layers[i], axis=1)
            min_distance = np.min(distances_to_next_layer)
            max_distance = np.max(distances_to_next_layer)
            avg_distance = np.mean(distances_to_next_layer)

            # Compute growth
            if i > 0:
                distances_to_prev_layer = np.linalg.norm(layers[i] - layers[i - 1], axis=1)
            else:
                distances_to_prev_layer = distances_to_next_layer
            growth = distances_to_next_layer / distances_to_prev_layer
            min_growth = np.min(growth)
            max_growth = np.max(growth)
            avg_growth = np.mean(distances_to_next_layer) / np.mean(distances_to_prev_layer)

            # Update cumulative average height
            cumulative_avg_height += avg_distance

            # Compute average radius compared to the origin
            radii = np.linalg.norm(layers[i+1], axis=1)
            avg_radius = np.mean(radii)

            # Print diagnostics
            print(f"Layer {i:4d}: "
                f"r={avg_radius:5.2f} - "
                f"y={cumulative_avg_height:8.5f} - "
                f"dy:(min={min_distance:.6f}, avg={avg_distance:.6f}, max={max_distance:.6f}) - "
                f"Growth=({avg_growth:.3f}, min={min_growth:.3f}, max={max_growth:.3f})")

    with Timer('Outputs', silent = not profiler):
        print(f'--- Outputs:  airfoil: {write_airfoil}, layers: {write_layers}, gmsh: {gmsh_write}, plot: {plot}')
        # --- Write airfoil coordinates to file
        if write_airfoil:
            airfoil_coords = layers[0]
            airfoil_file = basefilename + f"{output_prefix}_airfoil.txt"
            np.savetxt(airfoil_file, airfoil_coords[:, :2], header="X Y", comments="")
            print( 'Written airfoil coords.  :', airfoil_file)

        # --- Write all layers to file
        if write_layers:
            for idx, layer in enumerate(layers):
                layer_file = basefilename + f"{output_prefix}_layer_{idx}.txt"
                np.savetxt(layer_file, layer[:, :2], header="X Y", comments="")
            layer_file = basefilename + f"{output_prefix}_layer_*.txt"
            print( 'Written front coords     :', layer_file)

        # --- Save as .msh format
        points = np.vstack(layers)
        if len(layers) > 1:
            cells = create_quadrilateral_cells(layers, len(layers), len(layers[0]))
            if gmsh_write:
                filename_out = basefilename + f"{output_prefix}_layers.msh"
                save_mesh(points, cells, filename=filename_out)
            if gmsh_open:
                open_mesh_in_gmsh(filename=filename_out)

        # --- Plot layers
        if plot:
            plt.figure(figsize=(5, 5))
            for idx, layer in enumerate(layers):
                plt.plot(layer[:, 0], layer[:, 1], label=f"Layer {idx}")
            plt.legend()
            plt.axis("equal")
            plt.title("Airfoil and Layers")
            plt.xlabel("X")
            plt.ylabel("Y")
            plt.grid(True)
            if write_fig:
                figfilename = basefilename + f"{output_prefix}_layers.pdf"
                plt.savefig(figfilename)
                print( 'Written layer figure     :', figfilename)

    if conversion_needed:
        # Delete output_file
        try:
            os.remove(output_file)
        except OSError as e:
            print(f"[WARN] Error deleting file {output_file} (nc bug?)")


def exo_layers():
    """
    Command-line interface for extracting airfoil geometry and surrounding layers from an Exodus file.
    """
    import argparse
    parser = argparse.ArgumentParser(description="Extract airfoil geometry and surrounding layers from an Exodus file.")
    parser.add_argument("input_file", type=str, help="Path to the input Exodus file.")
    parser.add_argument("-n", "--num_layers", type=int, default=1, help="Number of layers to extract.")
    parser.add_argument("side_set_name", type=str, nargs='?', default=None, help="Name of the side set defining the airfoil. Defaults to None.")
    parser.add_argument("--gmsh_open", action="store_true", help="Open the generated .msh file in GMSH.")
    parser.add_argument("--airfoil", action="store_true", help="Write the airfoil coordinates to *_airfoil.txt.")
    parser.add_argument("--layers", action="store_true", help="Write the layers coordinates to *_airfoil.txt.")
    parser.add_argument("--plot", action="store_true", help="Plot the extracted layers.")
    parser.add_argument("--verbose", action="store_true", help="Enable verbose output.")
    parser.add_argument("--profiler", action="store_true", help="Enable Profiling.")
    args = parser.parse_args()

    extract_airfoil_geometry(
        input_file=args.input_file,
        side_set_name=args.side_set_name,
        num_layers=args.num_layers,
        write_airfoil=args.airfoil,
        write_layers=args.layers,
        verbose=args.verbose,
        gmsh_open=args.gmsh_open,
        plot=args.plot,
        profiler=args.profiler
    )
    #if args.plot:
    #    plt.show()


if __name__ == "__main__":
    exo_layers()
