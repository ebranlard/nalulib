import meshio

import os
import numpy as np
from nalulib.essentials import *
from nalulib.exodusii.file import ExodusIIFile

# Define the faces of a HEX8 element
#hex_faces_old = [
#    [0, 3, 2, 1],  # -X face
#    [4, 5, 6, 7],  # +X face
#    [0, 1, 5, 4],  # -Y face
#    [3, 7, 6, 2],  # +Y face
#    [0, 4, 7, 3],  # -Z face
#    [1, 2, 6, 5],  # +Z face
#]
#hex_faces_old = [
#    [0, 3, 2, 1],  # -X face
#    [4, 5, 6, 7],  # +X face
#    [0, 1, 5, 4],  # -Y face
#    [3, 7, 6, 2],  # +Y face
#    [0, 4, 7, 3],  # -Z face
#    [1, 2, 6, 5],  # +Z face
#]
hex_faces = [
    [1, 5, 4, 0],  # 1 
    [1, 2, 6, 5],  # 2
    [3, 2, 6, 7],  # 3
    [0, 3, 7, 4],  # 4
    [0, 1, 2, 3],  # 5
    [4, 5, 6, 7],  # 6
]


def gmesh_to_exo(filein, fileout=None, verbose=False):
    """
    Convert a 3D GMSH mesh file to Exodus II format.
    
    Parameters:
        filein (str): Input GMSH file name.
        fileout (str): Output Exodus II file name. If None, defaults to 'converted_mesh.e'.
        verbose (bool): If True, print detailed information during conversion.
    """

    # Default output file name
    if fileout is None:
        fileout = os.path.splitext(filein)[0] + ".exo"
    # ------------------------------------------------------------------------------ 
    # --- Read the GMSH mesh file
    # ------------------------------------------------------------------------------ 
    with Timer('Reading'):
        print(f"Reading msh file: {filein}")
        mesh = meshio.read(filein)

    # --------------------------------------------------------------------------------}
    # ---  
    # --------------------------------------------------------------------------------{
    with Timer('Prepping'):
        if verbose:
            print("--- Nodes ---")
            for node_id, node in zip(node_ids_map.values(), nodes):
                print(f"Node {node_id-1}: {node}")
            print("--- Elements ---")
            for element_id, element in zip(element_ids_map.values(), elements):
                print(f"Element {element_id-1}: {element}")

        # Map physical IDs to names (if available)
        surface_names = {}
        if mesh.field_data:
            for name, (physical_id, _) in mesh.field_data.items():
                surface_names[physical_id] = name

        # Extract nodes and elements
        nodes = mesh.points
        num_nodes = len(nodes)
        num_dims = nodes.shape[1]
        elements = []
        quads = mesh.cells_dict["quad"]
        if "hexahedron" in mesh.cells_dict:
            hexs = mesh.cells_dict["hexahedron"]
            elements = hexs

        # Assign unique IDs to nodes and elements (starting at 1)
        node_ids_map = {tuple(node): idx + 1 for idx, node in enumerate(nodes)}
        element_ids_map = {tuple(element): idx + 1 for idx, element in enumerate(elements)}

        if verbose:
            print("--- Nodes ---  0-indexing")
            for node_id, node in zip(node_ids_map.values(), nodes):
                print(f"Node {node_id-1}: {node}")

            print("--- Elements --- 0-indexing")
            for element_id, element in zip(element_ids_map.values(), elements):
                print(f"Element {element_id-1}: {element}")

        # Map physical IDs to names (if available)
        surface_names = {}
        if mesh.field_data:
            for name, (physical_id, _) in mesh.field_data.items():
                surface_names[physical_id] = name

        # Extract physical surface information (quads)
        physical_surfaces = {}
        if "gmsh:physical" in mesh.cell_data_dict:
            physical_data = mesh.cell_data_dict["gmsh:physical"]
            # Create physical_surfaces dictionary
            for surface_id in set(physical_data["quad"]):
                surface_name = surface_names.get(surface_id, f"surface_{surface_id}")
                physical_surfaces[surface_id] = {
                    "ID": surface_id,
                    "name": surface_name,
                    "quads_original": [],
                    "quads_matched": [],
                }

            # Populate quads_original for each physical surface
            for quad, surface_id in zip(quads, physical_data["quad"]):
                physical_surfaces[surface_id]["quads_original"].append(quad)

        # Precompute face node sets for all elements
        print("Precomputing face node sets for all elements...")
        element_faces = {}  # Dictionary to store face node sets for each element
        for hex_id, hex_element in enumerate(elements, start=1):  # 1-based hex ID
            element_faces[hex_id] = {
                face_id: set(hex_element[node] for node in face_nodes)
                for face_id, face_nodes in enumerate(hex_faces, start=1)
            }

        # Create a mapping from node IDs to elements
        print("Creating node-to-element mapping...")
        node_to_elements = {}
        for hex_id, hex_element in enumerate(elements, start=1):  # 1-based hex ID
            for node_id in hex_element:
                if node_id not in node_to_elements:
                    node_to_elements[node_id] = []
                node_to_elements[node_id].append(hex_id)

    # --------------------------------------------------------------------------------}
    # ---  
    # --------------------------------------------------------------------------------{
    with Timer('Surface mapping'):
        for surface_id, surface_data in physical_surfaces.items():
            # Print surface ID and name
            if verbose:
                print('--------------------------------------------------------------------')
                print(f"Surface ID: {surface_id}, Name: {surface_data['name']}")
                print(f"  Original Quads:")
                for quad in surface_data["quads_original"]:
                    print(f"  - Node IDs: {quad}")
                    print(f"    Quad Coordinates:")
                    for node_id in quad:
                        print(f"      Node {node_id}: {nodes[node_id]}")
            # --- 
            print(f"Looking for hex elements matching surface ID: {surface_id} ({surface_data['name']})")
            for quad in surface_data["quads_original"]:
                # Find candidate elements that contain at least one node of the quad
                candidate_elements = set()
                for node_id in quad:
                    if node_id in node_to_elements:
                        candidate_elements.update(node_to_elements[node_id])

                # Check each candidate element
                for hex_id in candidate_elements:
                    for face_id, face_node_set in element_faces[hex_id].items():
                        # Compare the quad nodes with the face nodes
                        if set(quad) == face_node_set:
                            if verbose:
                                print(f"  Match Found!")
                                print(f"    HEX8 Element ID: {hex_id}")
                                print(f"    Matching Face ID: {face_id}")
                                print(f"    Face Node IDs: {face_node_set}")
                            surface_data["quads_matched"].append((hex_id, face_id))  # Store hex ID and face ID
                            break
            # Debugging: Print matched quads
            if verbose:
                print("   --- Matched Quads ---")
                print(f"     Matched Quads (HEX8 ID, Face ID): {surface_data['quads_matched']}")

    # Assign unique IDs to physical surfaces (starting at 1)
    physical_surface_ids_map = {surface_id: idx + 1 for idx, surface_id in enumerate(physical_surfaces.keys())}

    # Display summary of the mesh
    print(f"Number of nodes:    {num_nodes:12d}")
    print(f"Number of elements: {len(elements):12d}")
    print("Physical surfaces:")
    for surface_id, surface_data in physical_surfaces.items():
        hex_ids  = [h for h,i in surface_data["quads_matched"]]
        face_ids = [i for h,i in surface_data["quads_matched"]]
        uface_ids = np.unique(face_ids)
        print(f"  - Surface ID {surface_id} ({surface_data['name']:10s}): {len(surface_data['quads_matched']):10d} quads, uFace ID {uface_ids}")

    # Verbose output
    if verbose:
        print("\n--- Surfaces ---")
        for surface_id, surface_data in physical_surfaces.items():
            print(f"Surface ID {surface_id} ({surface_data['name']}):")
            for hex_id, face_id in surface_data["quads_matched"]:
                print(f"  Hex ID {hex_id}, Face ID {face_id}")

    # ------------------------------------------------------------------------------ 
    # --- Write Exodus II file
    # ------------------------------------------------------------------------------ 
    with Timer('Writting'):
        with ExodusIIFile(fileout, mode="w") as exo:
            # Initialize the Exodus file
            exo.put_init(
                title="Converted Mesh from GMSH",
                num_dim=num_dims,
                num_nodes=num_nodes,
                num_elem=len(elements),
                num_elem_blk=1,
                num_node_sets=0,
                num_side_sets=len(physical_surfaces),
            )

            # Write node coordinates
            exo.put_coord(nodes[:, 0], nodes[:, 1], nodes[:, 2])
            exo.put_coord_names(["X", "Y", "Z"])

            # Write element block
            exo.put_element_block(0, "HEX8", len(elements), 8)
            # Convert element connectivity to 1-based indexing
            element_connectivity = np.asarray([[node_ids_map[tuple(nodes[node_idx])] for node_idx in element] for element in elements])
            exo.put_element_conn(0, element_connectivity)
            exo.put_element_block_names(["fluid-hex"])

            # Write side sets for physical surfaces
            for surface_id, surface_data in physical_surfaces.items():
                side_set_id = physical_surface_ids_map[surface_id]  # Use unique surface ID
                surface_name = surface_data["name"]
                side_set_elems = [hex_id for hex_id, _ in surface_data["quads_matched"]]
                side_set_faces = [face_id for _, face_id in surface_data["quads_matched"]]

                # Debugging output for verification
                if verbose:
                    print(f"side_set_id {side_set_id}")
                    print(f"Surface ID {surface_id} ({surface_name}): {len(side_set_elems)} elements")
                    print("side_set_elems", side_set_elems)
                    print("side_set_faces", side_set_faces)

                # Write side set to Exodus file
                exo.put_side_set_param(side_set_id, len(side_set_elems))
                exo.put_side_set_sides(side_set_id, side_set_elems, side_set_faces)
                exo.put_side_set_name(side_set_id, surface_name)

    print(f"\nExodus II file '{fileout}' created successfully.")


def gmesh2exo():
    """
    Command-line interface to convert a 3D GMSH mesh file to Exodus II format.
    """
    import argparse
    parser = argparse.ArgumentParser(description="Convert a 3D gmesh file to Exodus II format")
    parser.add_argument("input_file", type=str, help="Path to the gmesh file.")
    parser.add_argument("-o", "--output_file", type=str, default=None, help="Path to the output Exodus file. Defaults to '<input_file>.exo'.")
    parser.add_argument("-v", "--verbose", action="store_true", help="Enable verbose output.")

    args = parser.parse_args()

    gmesh_to_exo(args.input_file, args.output_file, verbose=args.verbose)


if __name__ == "__main__":
    import sys
    # Grab the filename from the first system argument or use a default
    filename = sys.argv[1] if len(sys.argv) > 1 else "example.msh"
    with Timer('Total conversion'):
        gmsh2exo(filename)
