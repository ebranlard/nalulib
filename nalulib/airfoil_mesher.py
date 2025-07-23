import os
import numpy as np
import matplotlib.pyplot as plt
# Local

import nalulib.essentials
from nalulib.airfoil_shapes import StandardizedAirfoilShape
from nalulib.airfoillib import airfoil_get_xy, _DEFAULT_REL_TOL



def mesh_airfoil(airfoil_coords_wrap, method='auto', n=100, n_te=None, check=True, respline=True, Re=1e6, 
                 outputfile=None, format=None, output_format=None, verbose=False, plot=False, thick=True, 
                 a_hyp=2.5, **kwargs):
    # Get airfoil coordinates based on multiple types of inputs for convenience
    x, y = airfoil_get_xy(airfoil_coords_wrap, format=format)

    #print(airfoil_coords)

    #print('x', x)
    #print('y', y)       
    #print('d', d)
    #x, y, d = normalize_airfoil_coords(x, y, d=d, reltol=_DEFAULT_REL_TOL, verbose=verbose)

    arf = StandardizedAirfoilShape(x, y, name='', reltol=_DEFAULT_REL_TOL, verbose=verbose)
    if verbose:
        print(arf)
    if plot:
        arf.plot(title='Original')

    if respline and method != 'refine' and method != 'none':
        arf = arf.resample_spline(n_surf=500, inplace=True)
        if plot:
            arf.plot(title='Resplined')
        #if verbose:
        #    print(arf)
    else:
        print('[INFO] No respline before meshing, using original coordinates.')

    arf.resample(method=method, n=n, n_te=n_te, inplace=True, verbose=verbose, a_hyp=a_hyp, **kwargs)

    if check:
        arf.check_mesh(Re=Re)

    if verbose:
        print(arf)

    if outputfile is not None:
        arf.write(outputfile, format=output_format, verbose=verbose, thick=thick)

    if plot:
        arf.plot(title='Remeshed')
    plt.show()



    #print(arf)
    #ax = arf.plot(title= 'Original', simple=True, label='Original')
    #arf.copy().resample_refine(inplace=True, factor_surf=2, factor_te=1).plot(label='Refined', ax=ax, simple=True, sty='o')

    #arf_hr = arf.copy()
    ##arf_hr.resample_te(n_te=10, inplace=True)
    #arf_hr.plot(title='High res')
    #arf_hr.check_mesh(Re=1e6)
    #print(arf_hr)
    #arf_cos = arf_hr.resample_cosine(n_surf=11)
    #arf_cos.check_mesh(Re=1e6)
    #arf_cos.plot(title='cosine')
    #arf_cos = arf_cos.resample_te(n_te=30)
    #arf_cos.plot(title='cosine with TE resampling')
    #print(arf_cos)
    #arf_hyp = arf_hr.resample_hyperbolic(n_surf=101)
    #arf_hyp.plot(title='hyperbolic')
    #print(arf_hyp)





def mesh_airfoil_CLI():
    """
    Command-line interface for meshing an airfoil.
    """
    import argparse
    parser = argparse.ArgumentParser(description="Mesh an airfoil and output the result in various formats.")
    parser.add_argument("input", type=str, help="Airfoil input (file path or airfoil name, e.g. 'naca0018' or './file.csv').")
    parser.add_argument("-o", "--output", type=str, default=None, help="Output file path.")
    parser.add_argument("-n", "--n", type=int, default=100, help="Number parameter (e.g. per surface, default: 100).")
    parser.add_argument("--n_te", type=int, default=None, help="Number of trailing edge points (if blunt TE).")
    parser.add_argument("--method", type=str, default="auto", choices=["auto", "cosine", "hyperbolic", "equi_n", "refine","none"], help="Meshing method.")
    parser.add_argument("--a_hyp", type=float, default=2.5, help="Hyperbolic clustering parameter (used if method='hyperbolic').")
    parser.add_argument("--format", type=str, default=None, help="Input format (e.g. 'csv', 'pointwise') if input file provided. If omitted, the fileformat is inferred from the extension.")
    parser.add_argument("--output-format", type=str, default=None, help="Output format (e.g. 'csv', 'pointwise') if output file provided.")
    parser.add_argument("--no-respline", action="store_true", help="Do not respline before meshing.")
    parser.add_argument("--check", action="store_true", help="Check mesh quality.")
    parser.add_argument("--no-thick", action="store_true", help="No thick outputs for pointwise output format.")
    parser.add_argument("--plot", action="store_true", help=" Plot mesh.")
    parser.add_argument("--Re", type=float, default=1e6, help="Reynolds number for mesh checks.")
    parser.add_argument("--verbose", action="store_true", help="Enable verbose output.")

    args = parser.parse_args()

    mesh_airfoil(
        airfoil_coords_wrap=args.input,
        method=args.method,
        n=args.n,
        n_te=args.n_te,
        check=args.check,
        plot=args.plot,
        thick=not args.no_thick,
        respline=not args.no_respline,
        Re=args.Re,
        outputfile=args.output,
        format=args.format,
        output_format=args.output_format,
        verbose=args.verbose,
        a_hyp=args.a_hyp
    )


if __name__ == '__main__':
    #mesh_airfoil_CLI()

    #import matplotlib.pyplot as plt
    #from nalulib.airfoil_shapes_naca import naca_shape
    #scriptDir = os.path.dirname(os.path.abspath(__file__))

    #x,y = naca_shape('0024', chord=1, n=151, sharp=False, pitch=0, xrot=0.25)
    #x,y = naca_shape('0024', chord=1, n=5, sharp=True, pitch=0, xrot=0.25)

    #mesh_airfoil((x,y))
    mesh_airfoil('naca0012', method='hyperbolic', n=10, n_te=4, check=False, respline=True, verbose=True, plot=True, thick=True)
    #mesh_airfoil(os.path.join(scriptDir, '../data/FFA_211_Re=10M_AoA0_nSpan=120_airfoil.txt'), format='csv')
    #mesh_airfoil(os.path.join(scriptDir, '../data/FFA_211_Re=10M_AoA0_nSpan=120_airfoil.txt'), format='csv')
    #mesh_airfoil(os.path.join(scriptDir, '../data/airfoils/ffa_w3_211_coords.pwise'), format='pointwise', verbose=True)
    #mesh_airfoil(os.path.join(scriptDir, '../data/airfoils/S809.csv'), format='csv', verbose=True)


    plt.show()
