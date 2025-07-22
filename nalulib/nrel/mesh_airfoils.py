import argparse
import pandas as pd
import os
#from nalulib.plot3D_plot3D2exo import plt3d_to_exo
from nalulib.nrel.plt3d2exo import plt3d2exo
import numpy as np


def get_s0(nu, reynolds, yplus):
    """
    Compute the first layer height based on Reynolds number and desired y+ using a turbulent flat plate approximation
    """
    U_inf = reynolds * nu
    Cf = 0.027 / reynolds ** (1 / 7)
    u_tau = np.sqrt(0.5 * Cf * U_inf**2)
    s0 = yplus * nu / u_tau

    return s0


def create_plot3d_surface(coords_file, surface_file):
    """
    Write out a 2D airfoil surface in 3D (one element in z direction)
    """
    base, ext = os.path.splitext(coords_file)
    print('ext',ext)
    if ext.lower()=='.csv':
        df = pd.read_csv(coords_file)
        x = df.iloc[:,0].values
        y = df.iloc[:,1].values
    else:
        coords = np.loadtxt(coords_file)
        x = coords[:, 0]
        y = coords[:, 1]
    print('Number of coordinates:', len(x))

    with open(surface_file, "w") as f:
        f.write("1\n")
        f.write(f"{len(x)} 2 1\n")
        for dim in range(3):
            for j in range(2):
                for i in range(len(x)):
                    if dim == 0:
                        f.write(f"{x[i]}\n")
                    elif dim == 1:
                        f.write(f"{y[i]}\n")
                    else:
                        f.write(f"{float(j)}\n")


def extrude_airfoil(coords_file, N, s0, marchDist):
    base, ext = os.path.splitext(coords_file)
    surface_file = base+"_temp_surface.fmt"
    create_plot3d_surface(coords_file, surface_file)

    options = {
        # ---------------------------
        #   General options
        # ---------------------------
        "inputFile": surface_file,
        "fileType": "PLOT3D",
        "unattachedEdgesAreSymmetry": False,
        "BC": {1: {"jLow": "zSymm", "jHigh": "zSymm"}},
        "outerFaceBC": "farfield",
        "families": "wall",
        # ---------------------------
        #   Grid parameters
        # ---------------------------
        "N": N,
        "s0": s0,
        "marchDist": marchDist,
        # ---------------------------
        #   Smoothing parameters
        # ---------------------------
        "epsE": 1.0,
        "epsI": 2.0,
        "theta": 3.0,
        "volCoef": [[0, 0.25], [0.1, 0.5], [1.0, 0.8]],
        "volBlend": [[0, 1e-7], [0.1, 1e-6], [1.0, 1e-3]],
        "volSmoothIter": [[0, 0], [0.1, 500], [1.0, 1000]],
        # ---------------------------
        #   Solution parameters
        # ---------------------------
        "cMax": 3.0,
        "kspRelTol": 1e-10,
        "kspMaxIts": 1500,
        "kspSubspaceSize": 50,
    }

    from pyhyp import pyHyp
    hyp = pyHyp(options=options)
    hyp.run()

    airfoil_tag = os.path.basename(coords_file).split(".")[0]
    plot3d_volume_file = os.path.join("volume", f"{airfoil_tag}.xyz")
    hyp.writePlot3D(plot3d_volume_file)

    plt3d2exo(plot3d_volume_file)

    # Remove temporary files
    os.remove(surface_file)
    os.remove(plot3d_volume_file)


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("coords_file", type=str)
    parser.add_argument("--nu", type=float, default=1.46071e-5)
    parser.add_argument("--reynolds", type=float, default=5e6)
    parser.add_argument("--yplus", type=float, default=0.1)
    parser.add_argument("--N", type=int, default=199)
    parser.add_argument("--marchDist", type=float, default=300.0)
    args = parser.parse_args()

    if not os.path.exists("volume"):
        os.makedirs("volume")

    s0 = get_s0(args.nu, args.reynolds, args.yplus)
    extrude_airfoil(args.coords_file, args.N, s0, args.marchDist)
