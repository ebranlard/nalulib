import os
import numpy as np
import matplotlib.pyplot as plt
# Local

import nalulib.essentials

from nalulib.airfoillib import *
from nalulib.airfoil_shapes_io import write_airfoil
from nalulib.airfoillib import _DEFAULT_REL_TOL

class NormalizedAirfoilShape():
    """ 
    Store a normalized representation of an airfoil shape. 
    The stored convention is such that:
      - the first point is the upper trailing edge (TE) point 
      - the contour is closed
      - the contour is counterclockwise
    The class stores the origianl starting point, orientation, and closed status of the airfoil.
    The class can output in a different convention as the internal representation (TODO)
    """
    def __init__(self, x, y, name='', reltol=_DEFAULT_REL_TOL, verbose=False):
        # Store original info
        self._ori_start_point = (x[0], y[0])
        self._ori_orientation = contour_orientation(x, y)
        self._ori_closed = contour_is_closed(x, y, reltol=reltol)

        # Normalized airfoil coordinates
        x, y = normalize_airfoil_coords(x, y, reltol=reltol, verbose=verbose)
        self.name = name
        self._x = x
        self._y = y
        self.reltol = reltol
        self.verbose=verbose

        # Possibility to output in different conventions
        self.output_closed = True
        self.output_counterclockwise = True

        self._TE_type = airfoil_TE_type(self._x, self._y)
        self._split_surfaces()

    def copy(self):
        return NormalizedAirfoilShape(x=self._x.copy(), y=self._y.copy(), name=self.name+'_copy', reltol=self.reltol)

    def return_new_or_copy(self, x_new, y_new, inplace=False, label='_new'):
        """ Return the normalized airfoil shape."""
        x_new, y_new = normalize_airfoil_coords(x_new, y_new, reltol=self.reltol, verbose=self.verbose)
        if inplace:
            self._x, self._y = x_new, y_new
            return self
        else:
            return NormalizedAirfoilShape(x_new, y_new, name=self.name+label, reltol=self.reltol, verbose=self.verbose)
    
    # --------------------------------------------------------------------------------
    # --- Properties 
    # --------------------------------------------------------------------------------
    @property
    def x(self):
        if self.output_closed and self.output_counterclockwise:
            return self._x.copy()
        else:
            raise NotImplementedError()
        #if self._closed:
        #    return np.append(self._x, self._x[0])
        #else:
        #    return self._x
    @property
    def y(self):
        if self.output_closed and self.output_counterclockwise:
            return self._y.copy()
        else:
            raise NotImplementedError()
        #if self._closed:
        #    return np.append(self._y, self._y[0])
        #else:
        #    return self._y

    @property
    def chord(self):
        return np.max(self.x) - np.min(self.x)

    @property
    def thickness_max(self):
        return np.max(self.y) - np.min(self.y)

    @property
    def t_rel_max(self):
        return self.thickness_max/self.chord

    @property
    def orientation(self):
        return contour_orientation(self.x, self.y)

    @property
    def is_closed(self):
        return contour_is_closed(self.x, self.y, reltol=self.reltol)

    # --------------------------------------------------------------------------------
    # --- Geometry
    # --------------------------------------------------------------------------------
    def _split_surfaces(self):
        self._IUpper, self._ILower, self._ITE, self._iLE = airfoil_split_surfaces(self._x, self._y, reltol=self.reltol, verbose=self.verbose)
        self._xu,  self._yu  = self._x[self._IUpper], self._y[self._IUpper]      
        self._xl,  self._yl  = self._x[self._ILower], self._y[self._ILower]
        self._xTE, self._yTE = self._x[self._ITE]   , self._y[self._ITE]
        self._xLE, self._yLE = self._x[self._iLE]   , self._y[self._iLE]

    def resample_te(self, n_te=None, inplace=False):
        self._split_surfaces()
        if n_te is not None:
            interp_te = lambda x_, y_ : curve_interp(x_, y_, n=n_te)
        else:
            raise NotImplementedError("n_te should be specified for resampling TE.")
        x_new, y_new = resample_airfoil_ul(self._x, self._y, self._IUpper, self._ILower, self._ITE, interp_ul=None, interp_te=interp_te)
        return self.return_new_or_copy(x_new, y_new, inplace=inplace, label='_te')

    def resample_spline(self, n_surf=1000, inplace=False, kind='cubic'):
        self._split_surfaces()
        x_new, y_new = resample_airfoil_spline(self._x, self._y, self._IUpper, self._ILower, self._ITE, n_surf=n_surf, kind=kind)
        return self.return_new_or_copy(x_new, y_new, inplace=inplace, label='_spline')

    def resample_cosine(self, n_surf=80, inplace=False):
        self._split_surfaces()
        x_new, y_new = resample_airfoil_cosine(self._x, self._y, self._IUpper, self._ILower, self._ITE, n_surf=n_surf)
        return self.return_new_or_copy(x_new, y_new, inplace=inplace, label='_cosine')

    def resample_hyperbolic(self, n_surf=80, a_hyp=2.5, inplace=False):
        self._split_surfaces()
        x_new, y_new = resample_airfoil_hyperbolic(self._x, self._y, self._IUpper, self._ILower, self._ITE, n_surf=n_surf, a_hyp=a_hyp)
        return self.return_new_or_copy(x_new, y_new, inplace=inplace, label='_hyper')

    def resample_refine(self, factor_surf=2, factor_te=1, inplace=False):
        x_new, y_new = resample_airfoil_refine(self._x, self._y, self._IUpper, self._ILower, self._ITE, factor_surf=factor_surf, factor_te=factor_te)
        return self.return_new_or_copy(x_new, y_new, inplace=inplace, label='_refine')

    def resample(self, method='', inplace=False, n=None, n_te=None, verbose=False, a_hyp=2.5, **kwargs):

        arf = self
        # --- Auto method
        if method == 'auto':
            method = 'hyperbolic'
            n = n if n is not None else GUIDELINES_N_DS_REC
            if self._TE_type == 'blunt':
                n_te = GUIDELINES_N_TE_BLUNT_REC
        if verbose:
            print('Resampling airfoil with method: {}, n: {}, n_te: {}'.format(method, n, n_te))

        if n_te is not None:
            arf = arf.resample_te(n_te=n_te, inplace=inplace)
            #arf.plot(title='Resampled TE')

        if method == 'equi_n':
            # TODO CUBIC OR LINEAR
            arf = arf.resample_spline(n_surf=n, inplace=inplace, kind='linear')
        elif method == 'cosine':
            arf = arf.resample_cosine(n_surf=n, inplace=inplace)
        elif method == 'hyperbolic':
            arf = arf.resample_hyperbolic(n_surf=n, inplace=inplace, a_hyp=a_hyp)
        elif method == 'refine':
            arf = arf.resample_refine(factor_surf=n, factor_te=1, inplace=inplace)
        elif method == 'none':
            pass
        else:
            raise ValueError("Unknown resampling method: {}".format(method))
        return arf


    def check_mesh(self, Re=1e6):
        self._split_surfaces()
        check_airfoil_mesh(self._x, self._y, self._IUpper, self._ILower, self._ITE, Re=Re)

    # --------------------------------------------------------------------------------
    # --- IO
    # --------------------------------------------------------------------------------
    def write(self, filename, format=None, verbose=False, thick=False):
        write_airfoil(self._x, self._y, filename, format=format, thick=thick)

    def __repr__(self):
        s='<{} object>:\n'.format(type(self).__name__)
        s+='|Main attributes:\n'
        s+='| - name: {}\n'.format(self.name)
        s+='| - x[0]={:.3f} x[-1]={:.3f}, xmax={:.3f}, xmin={:.3f},  dx~: {:.3f}, n: {} \n'.format(self.x[0], self.x[-1],np.max(self.x), np.min(self.x), np.mean(np.abs(self.x)), len(self.x))
        s+='| - y[0]={:.3f} y[-1]={:.3f}, ymax={:.3f}, ymin={:.3f},  dy~: {:.3f}, n: {} \n'.format(self.y[0], self.y[-1],np.max(self.y), np.min(self.y), np.mean(np.abs(self.y)), len(self.y))
        # Original attributes
        s+='| - _ori_closed: {}\n'.format(self._ori_closed)
        s+='| - _ori_orientation: {}\n'.format(self._ori_orientation)   
        s+='| - _ori_start_point: ({:.3f}, {:.3f})\n'.format(self._ori_start_point[0], self._ori_start_point[1])
        s+='| * TE_type    : {}\n'.format(self._TE_type)
        s+='| * orientation: {}\n'.format(self.orientation)
        s+='| * start_point: ({:.3f}, {:.3f})\n'.format(self.x[0], self.y[0])
        s+='| * chord: {}\n'.format(self.chord)
        s+='| * thickness_max: {:.6f}\n'.format(self.thickness_max)
        s+='| * t_rel_max:     {:.6f}\n'.format(self.t_rel_max)
        s+='| * ITE: {}\n'.format(self._ITE)
        s+='| * iLE: {}\n'.format(self._iLE)

        # ds for upper+lower
        xul = np.concatenate([self._xu, self._xl[1:]])
        yul = np.concatenate([self._yu, self._yl[1:]])
        sul = curve_coord(xul, yul, normalized=False)
        ds_ul = np.diff(sul)
        s+='| * ds upper+lower: min={:.4g}, max={:.4g}, mean={:.4g}\n'.format(np.min(ds_ul), np.max(ds_ul), np.mean(ds_ul))

        # ds for TE
        xte = self._xTE
        yte = self._yTE
        ste = curve_coord(xte, yte, normalized=False)
        ds_te = np.diff(ste)
        s+='| * ds TE         : min={:.4g}, max={:.4g}, mean={:.4g}\n'.format(np.min(ds_te), np.max(ds_te), np.mean(ds_te))

        # Output attributes
        s+='| * output_closed: {}\n'.format(self.output_closed)
        s+='| * output_counterclockwise: {}\n'.format(self.output_counterclockwise)
        return s

    # --------------------------------------------------------------------------------
    # --- Plot
    # --------------------------------------------------------------------------------
    def plot_surfaces(self):
        #xu, yu, xl, yl, Iu, Il = self.split_surfaces()
        #x0, y0, y0u, y0l = self.camberline()
        fig,ax = plt.subplots(1, 1, sharey=False, figsize=(6.4,4.8)) # (6.4,4.8)
        #fig.subplots_adjust(left=0.12, right=0.95, top=0.95, bottom=0.11, hspace=0.20, wspace=0.20)
        #ax.plot(self.x, self.y, 'k.', label='All')
        #ax.plot(xu, yu, 'o'         , label='Upper', markerfacecolor='none')
        #ax.plot(xl, yl, 'd'         , label='Lower', markerfacecolor='none')
        #ax.plot(x0, y0, '--' , label='Camber line' )
        ##ax.plot(x0, y0u, ':'  )
        ##ax.plot(x0, y0l, '-.'  )
        #ax.legend()
        #ax.set_xlabel('x')
        #ax.set_ylabel('y')
        #ax.set_title(self.name)
        #plt.axis ( 'equal' )
        return ax

    def plot(self, **kwargs):
        return plot_normalized(self._x, self._y, **kwargs)


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

    arf = NormalizedAirfoilShape(x, y, name='', reltol=_DEFAULT_REL_TOL, verbose=verbose)
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
