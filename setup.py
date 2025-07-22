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
            'exo-info=nalulib.cli:exo_info',
            'exo-flatten=nalulib.cli:exo_flatten',
            'exo-zextrude=nalulib.cli:exo_zextrude',
            'exo-rotate=nalulib.cli:exo_rotate',
            'exo-layers=nalulib.cli:exo_layers',
            'nalu-restart=nalulib.cli:nalu_restart',
            'nalu-aseq=nalulib.cli:nalu_aseq',
            'nalu-input=nalulib.cli:nalu_input_CLI',
            'gmsh2exo=nalulib.cli:gmsh2exo',
            'plt3d2exo=nalulib.cli:plt3d2exo_CLI',
            'arf-plot=nalulib.cli:airfoil_plot_CLI',
            'arf-convert=nalulib.cli:convert_airfoil_CLI',
            'arf-mesh=nalulib.cli:mesh_airfoil_CLI',
            'pyhyp=nalulib.cli:pyhyp_cmdline_CLI',
        ],
    },
)
