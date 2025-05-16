# nalulib

Small tools to work with airfoils in nalu-wind and 2D/3D airfoil meshes.

## Installation

```bash
git clone https://github.com/ebranlard/nalulib
cd nalulib
pip install -e .
```

## Command line tools 
After performing the pip install, the following tools should be accessible from the command line:

 - `exo-info`: print info about an exodus file
 - `exo-flatten`: flatten a 3D mesh into a 2D mesh (along z, preserves side-sets)
 - `exo-zextrude`: extrude a 2D mesh into a 3D mesh (along z, preserve side-sets)
 - `exo-rotate`: rotate a 2D/3D mesh about a point, can change or preserve side sets
 - `exo-layers`: extract and display information (thickness, growth) for layers about the wing sideset
 - `nalu-restart`: write a new yaml file and slurm script based on latest time found in restart file
 - `nalu-aseq`: write, meshes, yaml files and slurm scripts  for a sequence of angle of attack
 - `gmesh2exo`: convert a 3D gmesh file to exodus format (use physical surfaces as side-sets).


Typical usage (minimal runnable example):
```bash
cd examples
gmesh2exo -h # Display help
gmesh2exo diamond_n2.msh       # Create exo file from gmesh
exo-info  diamond_n2.exo       # Show info
exo-layers diamond_n2.exo -n 2000 --layers # Extract layers around airfoil and diagnostics
exo-flatten  diamond_n2.exo -o diamond_n1.exo  # Create 2D mesh (quads) from 3D mesh (hexs)
exo-zextrude diamond_n1.exo -z 4 -n 120       # Create 3D mesh from 2D mesh
exo-rotate   diamond_n120.exo -a 30             # Rotate mesh by 30 deg
nalu-aseq input.yaml -a -30 30 10 -j polar -b submit.sh # Create mesh, yaml, submit for polar
nalu-restart input.yaml -b submit.sh # Create restart yaml and submit script
```

Typical optional flags:

 - `-h` : display help and available flags
 - `-v` : verbose print to terminal
 - `-o` : output file (if omitted default file name are used)


## Exodus library
The exodus reader/writer was taken from: https://github.com/sandialabs/exodusii . See license in nalulib/exodusii.
