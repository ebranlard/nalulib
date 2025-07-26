import numpy as np
# Local
from nalulib.essentials import *
from nalulib.exodus_core import quad_is_positive_about_z, negative_quads
from nalulib.exodusii.file import ExodusIIFile

def exodus_get_names(filename, lower=False):
    """
    Get the names of the blocks, side sets, and variables in an Exodus file.

    :param filename: Path to the Exodus file.
    :return: A dictionary with block names, side set names, and variable names.
    """
    with ExodusIIFile(filename, mode="r") as exo:
        block_names = exo.get_element_block_names()
        side_set_names = exo.get_side_set_names()
        glob_var = exo.get_global_variable_names()
        elem_var = exo.get_element_variable_names()
        node_var = exo.get_node_variable_names()
        edge_var = exo.get_edge_variable_names()
        face_var = exo.get_face_variable_names()

    names = {
      "blocks"            : block_names,
      "side_sets"         : side_set_names,
      "global_variables"  : glob_var,
      "element_variables" : elem_var,
      "node_variables"    : node_var,
      "edge_variables"    : edge_var,
      "face_variables"    : face_var,
    }
    if lower:
        # return lower cases names
        names = {k: [name.lower() for name in v] for k, v in names.items()}
    return names

def print_elements(conn, node_coords, element_ids=None, ne=None, blk_type='quad'):
    if ne is None: 
        ne = len(conn)
    if element_ids is None:
        element_ids = range(conn.shape[0])
    for i, elem_id in enumerate(element_ids[:ne]):
        node_ids = conn[i]
        if blk_type.lower() in ['quad']:
            orientation = {True: 'positive', False:'negative'}[quad_is_positive_about_z(node_ids, node_coords, ioff=1)]
            print(f"Element ID: {elem_id}, Node IDs: {node_ids}, Orientation: {orientation}")
        else:
            print(f"Element ID: {elem_id}, Node IDs: {node_ids}")

def print_nodes(coords, node_ids=None, nn=None):
    if nn is None:
        nn = len(coords)    
    if node_ids is None:
        node_ids = range(1, nn + 1)
    for i, node_id in enumerate(node_ids[:nn]):
        print(f"Node ID: {node_id}, Coordinates: {coords[i]}")


def exo_info(filename, n=5, nss=10):
    print(f"Filename:                 {filename}")
    with ExodusIIFile(filename, mode="r") as exo:
        warnings = explore_exodus(exo, n=n, nss=nss)

    if len(warnings) > 0:
        print("\n------- WARNINGS ------------")
        for warn in warnings:
            print(f"[WARN] {warn}")
        plt.show()
    else:
        print(f"[ OK ] No warnings found analyzing {filename}.")

def explore_exodus(exo, n=5, nss=10):
    warnings=[]
    # Basic information
    block_ids = exo.get_element_block_ids()
    block_info = exo.get_element_block(block_ids[0])
    side_set_ids = exo.get_side_set_ids()
    side_set_names = exo.get_side_set_names()
    block_names = exo.get_element_block_names()
    print("\n--- Basic Information ---")
    print(f"Title                   :`{exo.title()}`")
    print(f"First block element type: {block_info.elem_type}")
    print(f"Number of Dimensions    : {exo.num_dimensions()}")
    print(f"Number of Nodes         : {exo.num_nodes()}")
    print(f"Number of Elements      : {exo.num_elems()}")
    print(f"Number of Blocks        : {exo.num_elem_blk()}")
    print(f"Block Names             : {block_names}")
    print(f"Number of Side Sets     : {exo.num_side_sets()}")
    print(f"Side Sets Names         : {side_set_names}")
    #print(f"Element Block ID: {block_info.id}, Type: {block_info.elem_type}, Name: {block_info.name}, nElem: {block_info.num_block_elems}, Nodes/Block: {block_info.num_elem_nodes}")

    print("Variables:", exo.get_element_variable_names())
    times = exo.get_times()
    print(f"Number of Time Steps    : {len(times)}")
    if len(times) > 1:
        print(f"Times                   : [{times[0]:.5f}, {times[1]:.5f}, ..., {times[-1]:.5f}] (n={len(times)})")
    elif len(times) > 0:
        print(f"Times                   : [{times[0]:.5f}, ..., {times[-1]:.5f}] (n={len(times)})")
    else:
        print(f"Times                   : ", times)
    glob_var = exo.get_global_variable_names()
    elem_var = exo.get_element_variable_names()
    node_var = exo.get_node_variable_names()
    edge_var = exo.get_edge_variable_names()
    face_var = exo.get_face_variable_names()
    print(f"Global Variables        : {glob_var} (n={len(glob_var)})")
    print(f"Element Variables       : {elem_var} (n={len(elem_var)})")
    print(f"Node Variables          : {node_var} (n={len(node_var)})")
    print(f"Edge Variables          : {edge_var} (n={len(edge_var)})")
    print(f"Face Variables          : {face_var} (n={len(face_var)})")


    # --- Display the min/max extent for each dimension
    node_coords = exo.get_coords()
    y_coords = node_coords[:, 1]
    x_coords = node_coords[:, 0]
    if exo.num_dimensions() == 3:
        z_coords = node_coords[:, 2]

    print("\n--- Nodes Dimensions ---")
    print("x: [{:10.6f}, {:10.6f}],  lx: {:10.6f}".format(x_coords.min(), x_coords.max(), x_coords.max() - x_coords.min()))
    print("y: [{:10.6f}, {:10.6f}],  ly: {:10.6f}".format(y_coords.min(), y_coords.max(), y_coords.max() - y_coords.min()))
    if exo.num_dimensions() == 3:
        print("z: [{:10.6f}, {:10.6f}],  lz: {:10.6f}".format(z_coords.min(), z_coords.max(), z_coords.max() - z_coords.min()))
    else:
        print("z: No z-coordinates (2D file)")

    # Display the first 5 nodes
    nn = min(n, exo.num_nodes())
    print("\n--- First {}/{} Nodes ---".format(nn, exo.num_nodes()))
    node_ids = exo.get_node_id_map()
    for i, node_id in enumerate(node_ids[:nn]):
        coords = node_coords[i]
        print(f"Node ID: {node_id}, Coordinates: {coords}")
    print_nodes(node_coords, node_ids=node_ids, nn=nn)


    # --- Elements 
    print("\n--- All Blocks Elements: {} ---".format(exo.num_elems()))
    element_ids = exo.get_element_id_map()
    # TODO warn if element_ids are not consecutive
    print("Element IDs range from: {} to {}, number of elements: {}".format(np.min(element_ids), np.max(element_ids), len(element_ids)-1))
    if np.max(element_ids)!=len(element_ids):
        warnings.append(f"Element IDs are not consecutive. IDs min-max [{np.min(element_ids)} {np.max(element_ids)}], number of elements: {len(element_ids)-1} (are we multiblock?).")
    if np.min(element_ids)!=1:
        warnings.append(f"First Element ID is not 1")

    for block_id in exo.get_element_block_ids():
        block_info = exo.get_element_block(block_id)
        #    info = SimpleNamespace(
        #        id=block_id,
        #        elem_type=elem_type,
        #        name=self.get_element_block_name(block_id),
        #        num_block_elems=self.num_elems_in_blk(block_id),
        #        num_elem_nodes=self.num_nodes_per_elem(block_id),
        #        num_elem_edges=self.num_edges_per_elem(block_id),
        #        num_elem_faces=self.num_faces_per_elem(block_id),
        #        num_elem_attrs=self.num_attr(block_id),
        #    )
        # name=self.get_element_block_name(block_id),
        ne = min(n, exo.num_elems())
        print("\n--- First {}/{} Elements ---".format(ne, exo.num_elems()))
        print(f"Block ID: {block_info.id}, Type: {block_info.elem_type}, Name: {block_info.name}, nElem: {block_info.num_block_elems}, Nodes/Block: {block_info.num_elem_nodes}")
        elem_conn = exo.get_element_conn(block_id)
        print("Element Nodes IDs range from: {} to {}, number of nodes: {}".format(np.min(elem_conn), np.max(elem_conn), len(node_coords)-1))
        if np.min(elem_conn)!=1:
            warnings.append(f"Minimum node ID is not 1 in element connectivity. Connectivity min-max IDs [{np.min(elem_conn)} {np.max(elem_conn)}], number of coords: {len(node_coords)-1}")
        if np.max(elem_conn)!=len(node_coords):
            warnings.append(f"Element connectivity does not cover all nodes exactly. Connectivity min-max IDs [{np.min(elem_conn)} {np.max(elem_conn)}], number of coords: {len(node_coords)-1}")
        print_elements(elem_conn, node_coords, element_ids=element_ids, ne=ne, blk_type=block_info.elem_type)

        # --- Count negative quads
        if block_info.elem_type.lower() in ['quad']:
            block_info = exo.get_element_block(block_id)
            INeg = negative_quads(elem_conn, node_coords, ioff=1, full_coverage_check=True, plot=True)
            nNeg = len(INeg)
            pcent = 100 * nNeg / len(element_ids)
            print("Negative quads: {}/{} ({:.2f}%)".format(nNeg, len(element_ids), pcent))
            if pcent==0 or pcent==100:
                pass
            else:
                warnings.append(f"Negative quads found: {nNeg}/{len(element_ids)} ({pcent:.2f}%)")

    # Display the first n side sets
    nss = min(nss, exo.num_side_sets())
    print("\n--- First {}/{} Side Sets ---".format(nss, exo.num_side_sets()))
    side_set_ids = exo.get_side_set_ids()
    for i, set_id in enumerate(side_set_ids[:nss]):
        try:
            side_set = exo.get_side_set(set_id)
            print(f"Side Set ID: {side_set.id}, Name: {side_set.name}")
            print(f"  Elements: {side_set.elems[:n]} (len={len(side_set.elems)}, nUnique={len(np.unique(side_set.elems))})")
            print(f"  Sides   : {side_set.sides[:n]} (len={len(side_set.sides)}, nUnique={len(np.unique(side_set.sides))})")
        except:
            print(f"Side Set ID: {set_id} not found or empty.")
    return warnings


def exo_info_CLI():
    import argparse
    parser = argparse.ArgumentParser(description="Explore the contents of an Exodus file.")
    parser.add_argument("filename", type=str, help="Path to the Exodus file to explore.")
    parser.add_argument("-n", type=int, default=5, help="Number of elements / nodes to display.")
    parser.add_argument("-nss", type=int, default=10, help="Number of side sets to display.")
    args = parser.parse_args()

    exo_info(args.filename, n=args.n, nss=args.nss)


if __name__ == "__main__":
    import sys
    # Grab the filename from the first system argument or use a default
    filename = sys.argv[1] if len(sys.argv) > 1 else "example.exo"
    exo_info(filename)
