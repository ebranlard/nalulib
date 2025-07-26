import numpy as np
import sys
import os
# Local
from nalulib.essentials import *
from nalulib.exodus_core import write_exodus, HEX_FACES, QUAD_SIDES
from nalulib.exodusii.file import ExodusIIFile


# TODO
SIDE_SETS_EXCLUDE=['front', 'back','wing-pp']


def exo_zextrude(input_file, output_file=None, nSpan=10, zSpan=4.0, zoffset=0.0, verbose=True, airfoil2wing=True, ss_wing_pp=True, profiler=False, ss_suffix=None):
    """
    Extrude a 2D QUAD mesh into a 3D HEX mesh along the z direction.

    Parameters:
        input_file (str): Path to the input Exodus file containing the 2D QUAD mesh.
        output_file (str): Path to the output Exodus file. Defaults to 'extruded_hex.exo'.
        nSpan (int): Number of layers in the z direction (nSpan-1 cells).
        zSpan (float): Total span in the z direction.
        verbose (bool): If True, print detailed information during processing.
    """
    # --- Default output file - We replace _n1 with _nNSPAN if possible
    if output_file is None:
        output_file = rename_n(input_file, nSpan)

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
        block_name_in = block_info.name

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

    # --- Infer potential ss_suffix
    if ss_suffix is None:
        suffix=[]
        for name in side_set_names: 
            suffix.append(name.split('_')[-1])
        usuffix = np.unique(suffix)
        if len(usuffix)==1:
            ss_suffix = '_'+usuffix[0]
            print(f"[INFO] Side set suffix inferred as: {ss_suffix}.  If this is undesired, use --ss_suffix ''")
        else:
            ss_suffix=''
    else:
        ss_suffix=''

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
            # Add suffix
            if not side_set_data["name"].endswith(ss_suffix):
                side_set_data["name"] += ss_suffix


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

        tmp_side_set_names = [str(v['name']) for v in new_side_sets.values()]
        if ss_wing_pp and ('wing' in tmp_side_set_names):
            if verbose:
                print('Adding wing_pp side set.')
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
            "name": "back"+ss_suffix,
        }
        new_side_sets[len(new_side_sets) + 1] = {
            "elements": front_elements,
            "sides": front_sides,
            "name": "front"+ss_suffix,
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
    delete_keys=[]
    for side_set_id, side_set_data in new_side_sets.items():
        if len(side_set_data["elements"]) == 0:
            if side_set_data['name'].lower().startswith('wing-pp'):
                print(f"[WARN] Side set {side_set_id} ({side_set_data['name']}) is empty after extrusion. Try with --no-ss_wing_pp. Deleting this side set.")
                delete_keys.append(side_set_id)
            else:
                raise ValueError(f"Side set {side_set_id} ({side_set_data['name']}) is empty after extrusion. Try with --no-")
    for key in delete_keys:
        del new_side_sets[key]



    # --- Write the extruded mesh to a new Exodus file
    title=f"Extruded HEX mesh from {input_file}"
    block_name ='fluid-hex'
    if block_name_in is not None:
        if block_name_in.lower().endswith('-quad'):
            block_name= block_name_in.replace('-quad','-hex').replace('-QUAD','-hex')
    write_exodus(output_file, new_node_coords, hex_conn, title=title, side_sets=new_side_sets, block_name=block_name, verbose=True, profiler=profiler)

    return output_file


def exo_zextrude_CLI():
    """
    Command-line interface for converting a 2D QUAD mesh into a 3D HEX mesh.
    """
    import argparse
    parser = argparse.ArgumentParser(description="Extrude a 2D QUAD mesh into a 3D HEX mesh along the z direction.")
    parser.add_argument("input_file", type=str, help="Path to the input Exodus file containing the 2D QUAD mesh.")
    parser.add_argument("-o",'--output_file', metavar='output_file', type=str, default=None, help="Path to the output Exodus file. Defaults to '<input_file>_nSpan=N.exo'.")
    parser.add_argument("-n", metavar='nSpan', type=int, default=10, help="Number of layers in the z direction (nSpan-1 cells). Default is 10.")
    parser.add_argument("-z", metavar='zSpan', type=float, default=4.0, help="Total span in the z direction. Default is 1.0.")
    parser.add_argument("--zoffset", type=float, default=0.0, help="Offset in the z direction. Default is 0.0.")
    parser.add_argument("-v", "--verbose", default=False, action="store_true", help="Enable verbose output.")
    parser.add_argument("--no-airfoil2wing", default=False, action="store_true", help="Do not rename 'airfoil' side set to 'wing'.")
    parser.add_argument("--no-ss_wing_pp", default=False, action="store_true", help="Do not split the 'wing' side set into 'wing' and 'wing_pp'.")
    parser.add_argument("--ss_suffix", default=None, help="Side set suffix (e.g. '_bg')")
    parser.add_argument("--profiler", action="store_true", help="Enable profiling with timers.")

    args = parser.parse_args()

    with Timer('quads_to_hex', silent = not args.profiler):
        exo_zextrude(
            input_file=args.input_file,
            output_file=args.output_file,
            nSpan=args.n,
            zSpan=args.z,
            zoffset=args.zoffset,
            verbose=args.verbose,
            airfoil2wing=not args.no_airfoil2wing,
            ss_wing_pp=not args.no_ss_wing_pp,
            ss_suffix=args.ss_suffix,
            profiler = args.profiler
        )

if __name__ == "__main__":
    exo_zextrude_CLI()

    #input_file = sys.argv[1]
    #nSpan = 121
    #zSpan = 4
    #output_file=None
    #zoffset=0
    #quads_to_hex(input_file, output_file, nSpan, zSpan, verbose=True, zoffset=zoffset, airfoil2wing=True, ss_wing_pp=True)





