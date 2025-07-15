import os
import glob
from nalulib.gmesh_gmesh2exodus import gmesh_to_exo
from nalulib.plot3D_plot3D2exo import plt3d2exo
from nalulib.exodus_info import explore_exodus_file
from nalulib.exodus_airfoil_layers import extract_airfoil_geometry
from nalulib.exodus_hex2quads import hex_to_quads_plane
from nalulib.exodus_quads2hex import quads_to_hex
from nalulib.exodus_rotate import rotate_exodus
from nalulib.nalu_aseq import nalu_prepare_aseq
from nalulib.nalu_restart import nalu_prepare_restart

def print_section(title, width=76, char='#'):
    title = f' {title} '
    pad = (width - len(title)) // 2
    print(char * pad + title + char * (width - len(title) - pad))

def print_command(msg):
    print(f"# {msg}")

def main(cleanUp=False, verbose=False):
    # For convenience, change to scriptDir
    python_dir = os.getcwd()
    scriptDir = os.path.dirname(__file__)
    os.chdir(scriptDir)

    # --- Convert gmsh mesh to Exodus
    print_section('gmsh2exo')
    print_command('gmsh2exo diamond_n2.msh       # Create exo file from gmesh')
    gmesh_to_exo("diamond_n2.msh", output_file=None, verbose=verbose)

    # --- Convert plot3d mesh to Exodus
    print_section('plt3d2exo')
    print_command('plt3d2exo diamond_n2.fmt     # Create exo file from plot3d')
    plt3d2exo("diamond_n2.fmt", output_file="diamond_n2_2.exo", verbose=verbose)

    # --- Show Exodus file info
    print_section('explore_exodus_file')
    print_command('exo-info  diamond_n2.exo       # Show info')
    explore_exodus_file("diamond_n2.exo")

    # --- Extract layers around airfoil and diagnostics
    print_section('extract_airfoil_geometry')
    print_command('exo-layers diamond_n2.exo -n 2000 --layers # Extract layers around airfoil and diagnostics')
    extract_airfoil_geometry(
        input_file="diamond_n2.exo",
        side_set_name=None,      # or specify if known, e.g. "airfoil"
        num_layers=2000,
        output_prefix="",
        verbose=verbose,
        write_airfoil=False, plot=False, gmsh_write=False, gmsh_open=False, write_layers=True, write_fig=False, profiler=False)

    # --- Create 2D mesh (quads) from 3D mesh (hexs)
    print_section('hex_to_quads_plane / flatten')
    print_command('exo-flatten  diamond_n2.exo -o diamond_n1.exo  # Create 2D mesh (quads) from 3D mesh (hexs)')
    hex_to_quads_plane(input_file="diamond_n2.exo", output_file="diamond_n1.exo", z_ref=None, verbose=verbose, profiler=False)

    # --- Create 3D mesh from 2D mesh
    print_section('quad_to_hex / zextrude')
    print_command('exo-zextrude diamond_n1.exo -z 4 -n 20       # Create 3D mesh from 2D mesh')
    quads_to_hex(input_file="diamond_n1.exo", output_file=None, 
                nSpan=20, zSpan=4, zoffset=0.0,
                verbose=verbose, airfoil2wing=True, ss_wing_pp=True, profiler=False, ss_suffix=None)

    # --- Rotate mesh by 30 deg
    print_section('rotate')
    print_command('exo-rotate   diamond_n20.exo -a 30             # Rotate mesh by 30 deg')
    rotate_exodus(
        input_file="diamond_n20.exo", output_file=None,  
        angle=30, center=(0, 0),
        angle_center=None, inlet_start=None, inlet_span=None, outlet_start=None,
        keep_io_side_set=False, verbose=verbose, profiler=False)

    # ---  Create mesh, yaml, submit for polar
    print_section('nalu_prepare_aseq')
    print_command('nalu-aseq input.yaml -a -30 30 10 -j polar -b submit.sh # Create mesh, yaml, submit for polar')
    nalu_prepare_aseq(input_file="input.yaml",
        aseq= (-30, 30, 10),
        verbose=verbose, debug=False,
        batch_file="submit.sh", cluster="unity",
        aoa_ori=None, jobname="polar")

    # ---  Create restart yaml and submit script
    print_section('nalu_prepare_restart')
    print_command('nalu-restart input.yaml -b submit.sh # Create restart yaml and submit script')
    nalu_prepare_restart(yaml_file="input.yaml",
        it=10, output_file='input_restart.yaml', verbose=verbose, debug=False, nt_max_tol=10, nrun=None,
        batch_file="submit.sh", cluster="unity")

    # --- Clean up
    if cleanUp:
        print_section('cleanup')
        # Remove files matching the specified patterns
        for pattern in [ 'submit-input*.sh', 'input_aoa*.yaml', 'input_restart.yaml', 'diamond_*.exo', 'diamond_*.txt']:
            print_command('rm '+pattern)
            for f in glob.glob(pattern):
                try:
                    os.remove(f)
                    print(f"Deleted: {f}")
                except Exception as e:
                    print(f"Could not delete {f}: {e}")

    # Change back to the original python directory
    os.chdir(python_dir)


if __name__ == "__main__":
    main(cleanUp=True, verbose=False)

if __name__ == "__test__":
    main(cleanUp=True, verbose=False)
