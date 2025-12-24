""" Command line interfaces (CLI)"""
import argparse

from nalulib.exodus import exo_cuboid_CLI
from nalulib.exodus import exo_flatten_CLI
from nalulib.exodus import exo_info_CLI
from nalulib.exodus import exo_layers_CLI
from nalulib.exodus import exo_rotate_CLI
from nalulib.exodus import exo_zextrude_CLI
from nalulib.gmesh import gmsh2exo_CLI
from nalulib.nalu_aseq import nalu_aseq_CLI
from nalulib.nalu_forces import nalu_forces_CLI
from nalulib.nalu_forces_combine import nalu_forces_combine_CLI
from nalulib.nalu_input import nalu_input_CLI
from nalulib.nalu_restart import nalu_restart_CLI
from nalulib.plot3D_plot3D2exo import plt3d2exo_CLI
from nalulib.airfoil_shapes_io import airfoil_plot_CLI
from nalulib.airfoil_shapes_io import convert_airfoil_CLI
from nalulib.airfoil_mesher import mesh_airfoil_CLI
from nalulib.pyhypwrap import pyhyp_CLI
from nalulib.airfoil_shapes import airfoil_info_CLI
