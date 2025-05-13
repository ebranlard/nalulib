from exodusii.file import ExodusIIFile
import numpy as np
import sys
import os
from welib.tools.clean_exceptions import *
from welib.essentials import Timer

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
    [0, 1],
    [1, 2],
    [2, 3],
    [3, 0]]

SIDE_SETS_EXCLUDE=['front', 'back','wing-pp']

def quads_to_hex(input_file, output_file=None, nSpan=10, zSpan=1.0, zoffset=0.0, verbose=True, airfoil2wing=True, ss_wing_pp=True, profiler=False):
    """
    Extrude a 2D QUAD mesh into a 3D HEX mesh along the z direction.

    Parameters:
        input_file (str): Path to the input Exodus file containing the 2D QUAD mesh.
        output_file (str): Path to the output Exodus file. Defaults to 'extruded_hex.exo'.
        nSpan (int): Number of layers in the z direction (nSpan-1 cells).
        zSpan (float): Total span in the z direction.
        verbose (bool): If True, print detailed information during processing.
    """
    if output_file is None:
        output_file = os.path.splitext(input_file)[0] + "_nSpan={}.exo".format(nSpan)

    print('Opening Exodus file      :', input_file)
    with ExodusIIFile(input_file, mode="r") as exo:
        # Read basic information
        num_nodes = exo.num_nodes()
        num_elems = exo.num_elems()
        node_coords = exo.get_coords()
        element_blocks = exo.get_element_block_ids()

        if len(element_blocks) > 1:
            raise NotImplementedError("This script only supports one block for extrusion.")
        block_id = element_blocks[0]
        block_info = exo.get_element_block(block_id)
        if block_info.elem_type not in ["QUAD"]:
            raise ValueError(f"Unsupported element type: {block_info.elem_type}. Only QUAD is supported.")

        # Read element connectivity
        quad_conn = np.array(exo.get_element_conn(block_id))  # Convert to NumPy array for slicing
        quad_ids = exo.get_element_id_map()  # Get the true element IDs for this block

        # Read side sets
        side_sets = {}
        side_set_names = []
        for side_set_id in exo.get_side_set_ids():
            side_set = exo.get_side_set(side_set_id)
            side_sets[side_set_id] = {
                "elements": list(map(int,side_set.elems)),
                "sides": list(map(int, side_set.sides)),
                "name": exo.get_side_set_name(side_set_id),
            }
            side_set_names.append(str(side_set.name))

    # --- Extrude the mesh
    with Timer('Extruding mesh', silent=not profiler, writeBefore=True):
        # Compute z-coordinates for each layer
        z_coords = np.linspace(0, zSpan, nSpan)+zoffset

        # Fill new node coordinates
        new_node_coords = np.zeros((num_nodes * nSpan, 3), dtype=np.float64)
        for i, z in enumerate(z_coords):
            start_idx = i * num_nodes
            end_idx = start_idx + num_nodes
            new_node_coords[start_idx:end_idx, 0] = node_coords[:, 0]  # X-coordinates
            new_node_coords[start_idx:end_idx, 1] = node_coords[:, 1]  # Y-coordinates
            new_node_coords[start_idx:end_idx, 2] = z                 # Z-coordinates

        # Preallocate HEX connectivity array
        num_hex = (nSpan - 1) * len(quad_conn)
        hex_conn = np.zeros((num_hex, 8), dtype=int)

        # Create HEX connectivity
        for i in range(nSpan - 1):
            offset = i * num_nodes
            start_idx = i * len(quad_conn)
            end_idx = start_idx + len(quad_conn)
            hex_conn[start_idx:end_idx, :] = np.column_stack([
                quad_conn[:, 0] + offset,
                quad_conn[:, 1] + offset,
                quad_conn[:, 2] + offset,
                quad_conn[:, 3] + offset,
                quad_conn[:, 0] + offset + num_nodes,
                quad_conn[:, 1] + offset + num_nodes,
                quad_conn[:, 2] + offset + num_nodes,
                quad_conn[:, 3] + offset + num_nodes,
            ])
    if verbose:
        # Print mesh information
        print("--- Mesh Information ---")
        print(f"Element block name: {block_info.name} (original)")
        print(f"Number of nodes   : {len(new_node_coords)} (new) vs {num_nodes} (original)")
        print(f"Number of elements: {len(hex_conn)} (new) vs {len(quad_conn)} (original)")
        print(f"z-coords          : [{z_coords[0]:.3f}, {z_coords[1]:.3f}, ..., {z_coords[-1]:.3f}], len={nSpan}")
        print(f"Number of z-cells : {len(z_coords)-1} (zSpan={zSpan:.3f})")

    # Update side sets for extrusion
    wing_pp_elements = []
    wing_pp_sides = []
    with Timer('Update side sets', silent=not profiler, writeBefore=True):
        new_side_sets = {}
        for side_set_id, side_set_data in side_sets.items():
            if side_set_data["name"] in SIDE_SETS_EXCLUDE:
                continue
            if airfoil2wing:
                if side_set_data["name"] == "airfoil":
                    side_set_data["name"] = "wing"

            # Extrude side sets
            elements = side_set_data["elements"]
            sides = side_set_data["sides"]
            extruded_elements = []
            extruded_sides = []


            for i in range(nSpan - 1):
                offset = i * len(quad_ids)
                span_elements = []
                span_sides = []
                for elem, side in zip(elements, sides):
                    span_elements.append(elem + offset)
                    span_sides.append(side)
                if ss_wing_pp and side_set_data["name"] == "wing" and i == (nSpan - 1) // 2:
                    if len(wing_pp_elements) > 0:
                        raise ValueError("wing_pp side set already defined")
                    wing_pp_elements = span_elements
                    wing_pp_sides = span_sides
                else:
                    extruded_elements.extend(span_elements)
                    extruded_sides.extend(span_sides)

            new_side_sets[side_set_id] = {
                "elements": list(map(int,extruded_elements)),
                "sides": list(map(int,extruded_sides)),
                "name": side_set_data["name"],
            }

        if ss_wing_pp:
            # Add the "wing_pp" side set
            new_side_sets[len(new_side_sets) + 1] = {
                "elements": wing_pp_elements,
                "sides": wing_pp_sides,
                "name": "wing-pp",
            }

        # Add front and back side sets
        front_elements = []
        front_sides = []
        back_elements = []
        back_sides = []

        for i, quad in enumerate(quad_conn):
            # Back side set (z=0)
            back_elements.append(i + 1)  # Element IDs are 1-based
            back_sides.append(5)  # Side index for the bottom face of HEX8

            # Front side set (z=zSpan)
            front_elements.append(i + (nSpan - 2) * len(quad_conn) + 1)  # Corrected offset for the top layer
            front_sides.append(6)  # Side index for the top face of HEX8

        # Add to new_side_sets
        new_side_sets[len(new_side_sets) + 1] = {
            "elements": back_elements,
            "sides": back_sides,
            "name": "back",
        }
        new_side_sets[len(new_side_sets) + 1] = {
            "elements": front_elements,
            "sides": front_sides,
            "name": "front",
        }


    new_side_set_names = [str(v['name']) for v in new_side_sets.values()]
    if verbose:
        # Print side set information
        print("--- Side Sets ---")    
        print(f"Side set names:   : {side_set_names} (original)")
        print(f"Side set names:   : {new_side_set_names} (new)")
        for side_set_id, side_set_data in new_side_sets.items():
            print(f"Side Set ID: {side_set_id}, Name: {side_set_data['name']}")
            print(f"  Elements: {side_set_data['elements'][:5]} (len={len(side_set_data['elements'])}, nUnique={len(np.unique(side_set_data['elements']))})")
            print(f"  Sides   : {side_set_data['sides'][:5]} (len={len(side_set_data['sides'])}, nUnique={len(np.unique(side_set_data['sides']))})")

    # Check that no side set is empty
    for side_set_id, side_set_data in new_side_sets.items():
        if len(side_set_data["elements"]) == 0:
            raise ValueError(f"Side set {side_set_id} ({side_set_data['name']}) is empty after extrusion.")

    # --- Write the extruded mesh to a new Exodus file
    with Timer('Writing file', silent=not profiler):
        with ExodusIIFile(output_file, mode="w") as exo_out:
            exo_out.put_init(
                title=f"Extruded HEX mesh from {input_file}",
                num_dim=3,
                num_nodes=len(new_node_coords),
                num_elem=len(hex_conn),
                num_elem_blk=1,
                num_node_sets=0,
                num_side_sets=len(new_side_sets),
                double=True #  Added by Emmanuel
            )

            # Write node coordinates
            exo_out.put_coord(new_node_coords[:, 0], new_node_coords[:, 1], new_node_coords[:, 2])
            exo_out.put_coord_names(["X", "Y", "Z"])

            # Write element block
            exo_out.put_element_block(0, "HEX8", len(hex_conn), 8)
            exo_out.put_element_conn(0, hex_conn)
            exo_out.put_element_block_names(["fluid-hex"])

            # Write side sets
            for side_set_id, side_set_data in new_side_sets.items():
                exo_out.put_side_set_param(side_set_id, len(side_set_data["elements"]))
                exo_out.put_side_set_sides(side_set_id, side_set_data["elements"], side_set_data["sides"])
                exo_out.put_side_set_name(side_set_id, side_set_data["name"])

    print(f"Written Exodus file      : {output_file}")
    return output_file


def exo_quads2hex():
    """
    Command-line interface for converting a 2D QUAD mesh into a 3D HEX mesh.
    """
    import argparse
    parser = argparse.ArgumentParser(description="Extrude a 2D QUAD mesh into a 3D HEX mesh along the z direction.")
    parser.add_argument("input_file", type=str, help="Path to the input Exodus file containing the 2D QUAD mesh.")
    parser.add_argument("-o",'--output_file', metavar='output_file', type=str, default=None, help="Path to the output Exodus file. Defaults to '<input_file>_nSpan=N.exo'.")
    parser.add_argument("-n", metavar='nSpan', type=int, default=10, help="Number of layers in the z direction (nSpan-1 cells). Default is 10.")
    parser.add_argument("-z", metavar='zSpan', type=float, default=1.0, help="Total span in the z direction. Default is 1.0.")
    parser.add_argument("--zoffset", type=float, default=0.0, help="Offset in the z direction. Default is 0.0.")
    parser.add_argument("--verbose", default=False, action="store_true", help="Enable verbose output.")
    parser.add_argument("--no-airfoil2wing", default=False, action="store_true", help="Do not rename 'airfoil' side set to 'wing'.")
    parser.add_argument("--no-ss_wing_pp", default=False, action="store_true", help="Do not split the 'wing' side set into 'wing' and 'wing_pp'.")
    parser.add_argument("--profiler", action="store_true", help="Enable profiling with timers.")

    args = parser.parse_args()

    with Timer('quads_to_hex', silent = not args.profiler):
        quads_to_hex(
            input_file=args.input_file,
            output_file=args.output_file,
            nSpan=args.n,
            zSpan=args.z,
            zoffset=args.zoffset,
            verbose=args.verbose,
            airfoil2wing=not args.no_airfoil2wing,
            ss_wing_pp=not args.no_ss_wing_pp,
            profiler = args.profiler
        )

exo_zextrude = exo_quads2hex

if __name__ == "__main__":
    exo_quads2hex()

    #input_file = sys.argv[1]
    #nSpan = 121
    #zSpan = 4
    #output_file=None
    #zoffset=0
    #quads_to_hex(input_file, output_file, nSpan, zSpan, verbose=True, zoffset=zoffset, airfoil2wing=True, ss_wing_pp=True)





