""" Command line interfaces (CLI)"""
import argparse

from nalulib.exodus import exo_info
from nalulib.exodus import exo_layers
from nalulib.exodus import exo_hex2quads, exo_flatten
from nalulib.exodus import exo_quads2hex, exo_zextrude
from nalulib.exodus import exo_rotate
from nalulib.gmesh import gmsh2exo
from nalulib.nalu_input import nalu_input_CLI
from nalulib.nalu_restart import nalu_restart
from nalulib.nalu_aseq import nalu_aseq
from nalulib.nalu_forces import nalu_forces_CLI
from nalulib.plot3D_plot3D2exo import plt3d2exo_CLI
from nalulib.airfoil_shapes_io import airfoil_plot_CLI
from nalulib.airfoil_shapes_io import convert_airfoil_CLI
from nalulib.airfoil_mesher import mesh_airfoil_CLI
from nalulib.pyhypwrap import pyhyp_cmdline_CLI
from nalulib.airfoil_shapes import airfoil_info_CLI
