from setuptools import setup, find_packages

with open('requirements.txt') as f:
    requirements = f.read().splitlines()

setup(
    name='nalulib',
    version='0.1',
    packages=find_packages(),
    install_requires=requirements,
    entry_points={
        'console_scripts': [
            'exo-info=nalulib.cli:exo_info_CLI',
            'exo-flatten=nalulib.cli:exo_flatten_CLI',
            'exo-zextrude=nalulib.cli:exo_zextrude_CLI',
            'exo-rotate=nalulib.cli:exo_rotate_CLI',
            'exo-layers=nalulib.cli:exo_layers_CLI',
            'nalu-input=nalulib.cli:nalu_input_CLI',
            'nalu-restart=nalulib.cli:nalu_restart',
            'nalu-aseq=nalulib.cli:nalu_aseq_CLI',
            'nalu-forces = nalulib.nalu_forces:nalu_forces_CLI',
            'gmsh2exo=nalulib.cli:gmsh2exo_CLI',
            'plt3d2exo=nalulib.cli:plt3d2exo_CLI',
            'arf-info=nalulib.airfoil_shapes:airfoil_info_CLI',
            'arf-plot=nalulib.cli:airfoil_plot_CLI',
            'arf-convert=nalulib.cli:convert_airfoil_CLI',
            'arf-mesh=nalulib.cli:mesh_airfoil_CLI',
            'pyhyp=nalulib.cli:pyhyp_CLI',
        ],
    },
)
