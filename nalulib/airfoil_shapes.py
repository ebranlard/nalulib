""" 

Trailing edge types:
 - Blunt: | 
 - Sharp:  > 
 - Cusp:  = same as sharp, but the tangent are the same


Standardized convention:
 - anticlockwise
 - first and last points are the same and repeated
 - first point is the upper point of the trailing edge

Normalized (should not be used yet):
 - (chord is 1 and trailing edge is at x=1, leading edge is at x=0)

if Blunt: 
- Upper surface include the upper point of the trailing edge and the leading edge
- Lower surface include the lower point of the trailing edge and the leading edge
- Trailing edge contains all the points at x=xmax 

if Sharp: 
- Upper surface include the trailing edge and the leading edge
- Lower surface include the trailing edge and the leading edge


TODO: closed should be an "output property"

"""
import os
import numpy as np
import pandas as pd
import argparse
import nalulib.pyplot as plt
# NOTE: consider using shapely for more operations
# try:
#     import shapely
# except:
#     print('[WARN] shape: shapely not available')

# Local 
from nalulib.airfoillib import *
from nalulib.airfoil_shapes_io import write_airfoil, read_airfoil
from nalulib.airfoillib import _DEFAULT_REL_TOL

# Used by lecay AirfoilShape class
from nalulib.curves import contour_is_closed
from nalulib.curves import contour_orientation
from nalulib.curves import contour_remove_duplicates
from nalulib.curves import find_closest
from nalulib.curves import opposite_contour
from nalulib.curves import reloop_contour


class StandardizedAirfoilShape():
    """ 
    Store a standardized representation of an airfoil shape. 
    The stored convention is such that:
      - the first point is the upper trailing edge (TE) point 
      - the contour is closed
      - the contour is counterclockwise
    The class stores the origianl starting point, orientation, and closed status of the airfoil.
    The class can output in a different convention as the internal representation (TODO)
    """
    def __init__(self, x, y, name='', reltol=_DEFAULT_REL_TOL, verbose=False):
        # Store original info

        self._ori_standardized = airfoil_is_standardized(x, y, raiseError=False)
        self._ori_start_point = (x[0], y[0])
        self._ori_orientation = contour_orientation(x, y)
        self._ori_closed = contour_is_closed(x, y, reltol=reltol)
        self._ori_xy = np.column_stack((x, y))

        # Normalized airfoil coordinates
        x, y = standardize_airfoil_coords(x, y, reltol=reltol, verbose=verbose)
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
        return StandardizedAirfoilShape(x=self._x.copy(), y=self._y.copy(), name=self.name+'_copy', reltol=self.reltol)

    def return_new_or_copy(self, x_new, y_new, inplace=False, label='_new'):
        """ Return the normalized airfoil shape."""
        x_new, y_new = standardize_airfoil_coords(x_new, y_new, reltol=self.reltol, verbose=self.verbose)
        if inplace:
            self._x, self._y = x_new, y_new
            return self
        else:
            return StandardizedAirfoilShape(x_new, y_new, name=self.name+label, reltol=self.reltol, verbose=self.verbose)
    
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
        self._split_surfaces()
        s='<{} object>:\n'.format(type(self).__name__)
        s+='|Basic properties:\n'
        s+='| - name       : {}\n'.format(self.name)
        s+='| * TE_type    : {}\n'.format(self._TE_type)
        s+='| * chord      : {}\n'.format(self.chord)
        s+='| * thickness_max: {:.6f}\n'.format(self.thickness_max)
        s+='| * t_rel_max  :     {:.6f}\n'.format(self.t_rel_max)
        # Original attributes
        s+='|Properties of the original airfoil (before standardization):\n'
        s+='| > (original) number of points : {:4d}\n'.format(len(self._ori_xy))
        s+='| > (original) standardized     : {}\n'.format(self._ori_standardized)
        s+='| > (original) closed           : {}\n'.format(self._ori_closed)
        s+='| > (original) orientation      : {}\n'.format(self._ori_orientation)   
        s+='| > (original) start_point      : ({:.3f}, {:.3f})\n'.format(self._ori_start_point[0], self._ori_start_point[1])
        s+='|Properties of the airfoil after standardization:\n'
        s+='| - x[0]={:.3f} x[-1]={:.3f}, xmax={:.3f}, xmin={:.3f},  dx~: {:.3f}, n: {} \n'.format(self.x[0], self.x[-1],np.max(self.x), np.min(self.x), np.mean(np.abs(self.x)), len(self.x))
        s+='| - y[0]={:.3f} y[-1]={:.3f}, ymax={:.3f}, ymin={:.3f},  dy~: {:.3f}, n: {} \n'.format(self.y[0], self.y[-1],np.max(self.y), np.min(self.y), np.mean(np.abs(self.y)), len(self.y))
        #s+='| * orientation: {}\n'.format(self.orientation)
        s+='| * start_point  : ({:.3f}, {:.3f})\n'.format(self.x[0], self.y[0])
        s+='| * ITE: {}\n'.format(self._ITE)
        s+='| * iLE: {}\n'.format(self._iLE)
        s+='| > number of TE points   : {:4d}\n'.format(len(self._ITE))
        s+='| > number of upper points: {:4d}\n'.format(len(self._IUpper))
        s+='| > number of lower points: {:4d}\n'.format(len(self._ILower))
        s+='| > total number of points: {:4d}\n'.format(len(self.x))

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

        # Output attributes FOR LATER
        if not self.output_closed:
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
        return plot_standardized(self._x, self._y, **kwargs)


# ---------------------------------------------------------------------------
# --- Airfoil shape (LEGACY)
# ---------------------------------------------------------------------------


def normalize(x,y):
    c  = np.max(x)-np.min(x)
    xc = (x-np.min(x))/c # between 0 and 1
    yc = (y)/c
    return xc, yc

class AirfoilShape():
    def __init__(self, x=None, y=None, name=None, filename=None, blunt=None, 
                 reltol=_DEFAULT_REL_TOL):
        """ 
         - reltol: relative tolerance to figure out if points are close. Relative to chord
        """
        # 
        if filename is not None:
            import welib.weio as weio
            df = weio.read(filename).toDataFrame()
            x = df.iloc[:,0]
            y = df.iloc[:,1]
            if name is None:
                name = os.path.splitext(os.path.basename(filename))[0]
        if name is None:
            name =''
        self._blunt=blunt # blunt, sharp, other?
        x = np.asarray(x)
        y = np.asarray(y)
        self.name = name
        self.filename = filename
        #self._x_bkp = x.copy()
        #self._y_bkp = y.copy()

        self.reltol = reltol
        self.closed = contour_is_closed(x, y, reltol=self.reltol)
        # --- 
        # We only store the unclosed contour in self._x and self._y
        self._x, self._y = self._removeDuplicates(x, y)

    def copy(self):
        return AirfoilShape(x=self.x.copy(), y=self.y.copy(), name=self.name+'_copy', reltol=self.reltol)

    # --------------------------------------------------------------------------------
    # --- Properties 
    # --------------------------------------------------------------------------------
    @property
    def x(self):
        if self._closed:
            return np.append(self._x, self._x[0])
        else:
            return self._x
    @property
    def y(self):
        if self._closed:
            return np.append(self._y, self._y[0])
        else:
            return self._y

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
    def startPoint(self):
        iTE = self.iTE
        iLE = self.iLE
        if iTE==0:
            return 'TE'
        if iLE==0:
            return 'LE'
        else:
            return 'Unknown'

    @property
    def closed(self):
        return self._closed

    @closed.setter
    def closed(self, val):
        self._closed = (val==True)

    @property
    def is_closed(self):
        # NOTE: the internal contour _x, _y should never be closed
        return contour_is_closed(self.x, self.y, reltol=self.reltol)

    @property
    def iTE(self):
        # TODO consider interpolating and returning upper and lower neighbors 
        xc, yc = normalize(self._x, self._y)
        xcTE, ycTE, iTE = find_closest(xc, yc, (1,0))
        return iTE

    @property
    def iLE(self):
        # TODO consider interpolating and returning upper and lower neighbors 
        xc, yc = normalize(self._x, self._y)
        xcLE, ycLE, iLE = find_closest(xc, yc, (0,0))
        return iLE

    # --------------------------------------------------------------------------------}
    # --- Accessor / Read Only routines
    # --------------------------------------------------------------------------------{
    def leading_edge(self):
        i = self.iLE
        return self._x[i], self._y[i], i

    def trailing_edge(self):
        i = self.iTE
        return self._x[i], self._y[i], i

    def split_surfaces(self):
        """ 
        - Iu: indices such that xu = x[Iu] (note: xu is increasing)
        """
        # TODO TODO Lot more work do be done here
        xLE, yLE, iLE = self.leading_edge ()
        xTE, yTE, iTE = self.trailing_edge()
        # Naive implementation
        if iLE==0:
            I1 = np.arange(iLE+1, iTE)
            I2 = np.arange(iTE+1, len(self._x))
        elif iTE==0:
            I1 = np.arange(iTE+1, iLE)
            I2 = np.arange(iLE+1, len(self._x))
        else:
            raise NotImplementedError()
        y1 =np.mean(self.y[I1])
        y2 =np.mean(self.y[I2])
        if y1>y2:
            Iu=I1
            Il=I2
        else:
            Il=I1
            Iu=I2
        xu = self._x[Iu]
        xl = self._x[Il]
        # Making sure x is increasing
        IIu = np.argsort(xu)
        IIl = np.argsort(xl)
        Iu = Iu[IIu]
        Il = Il[IIl]
        xu = self._x[Iu]
        xl = self._x[Il]
        yu = self._y[Iu]
        yl = self._y[Il]

        return xu, yu, xl, yl, Iu, Il

    def camberline(self):
        xu, yu, xl, yl, Iu, Il = self.split_surfaces()
        x1 = min(np.min(xu), np.min(xl))
        x2 = max(np.max(xu), np.max(xl))
        x0 = np.linspace(x1, x2, int(len(self._x)/2)+1)
        y0u = np.interp(x0, xu, yu)
        y0l = np.interp(x0, xl, yl)

        y0 = (y0u+y0l)/2
        return x0, y0, y0u, y0l
    
    # --------------------------------------------------------------------------------
    # --- Functions that change order / start point but not the data
    # --------------------------------------------------------------------------------
    def setStartPoint(self, name='LE', i=None, P=None):
        if name=='LE':
            i = self.iLE
            self._x, self._y = reloop_contour(self._x, self._y, i)
        elif name=='TE':
            i = self.iTE
            self._x, self._y = reloop_contour(self._x, self._y, i)
        else:
            raise NotImplementedError()

    def setOrientation(self, orientation='clockwise'):
        allowed = ['clockwise','counterclockwise']
        if orientation not in allowed:
            raise Exception('orientation should be: {}'.format(allowed))
        # Compute current orientation
        ori = self.orientation
        if ori not in allowed:
            raise Exception('Orientation cannot be determined')
        if ori==orientation:
            pass
        else:
            self._x, self._y = opposite_contour(self._x, self._y)
        return None

    def setClosed(self):
        """ Close contour """
        self.closed = True

    def setClosed(self):
        """ Close contour """
        self.closed = False

    # --------------------------------------------------------------------------------
    # --- Functions that potentially change the data
    # --------------------------------------------------------------------------------
    def _removeDuplicates(self, x=None, y=None, inPlace=False, verbose=False):
        """ remove duplicates in list of points """
        if x is None:
            x = self._x
            y = self._y
        n1 = len(x)
        xout, yout, duplicate_points = contour_remove_duplicates(x, y, reltol=self.reltol)
        n2 = len(xout)
        if n2!=n1:
            if self.closed and n2-n1==1:
                pass # Normal case
            else:
                #if verbose:
                print('[WARN] AirfoilShape: {} duplicate(s) removed: {}'.format(n1-n2, duplicate_points))
        if inPlace:
            self._x = xout
            self._y = yout
        return xout, yout

    # --------------------------------------------------------------------------------
    # --- IO 
    # --------------------------------------------------------------------------------
    def write(self, filename, format='csv', delim=' ', header=True, comment_char='#', xlabel='x', ylabel='y', units=''):
        if format=='csv':
            s=''
            if header:
                s+= comment_char+'{}{}{}{}{}\n'.format(xlabel, units, delim, ylabel, units)
            for xx,yy in zip(self.x, self.y):
                s += '{:12.7f}{:s}{:12.7f}\n'.format(xx, delim, yy)
            with open ( filename, 'w' ) as f:
                  f.write ( s )
        else:
            NotImplementedError()

    def __repr__(self):
        s='<{} object>:\n'.format(type(self).__name__)
        s+='|Main attributes:\n'
        s+='| - name: {}\n'.format(self.name)
        s+='| - x[0]={:.3f} x[-1]={:.3f}, xmax={:.3f}, xmin={:.3f},  dx~: {:.3f}, n: {} \n'.format(self.x[0], self.x[-1],np.max(self.x), np.min(self.x), np.mean(np.abs(self.x)), len(self.x))
        s+='| - y[0]={:.3f} y[-1]={:.3f}, ymax={:.3f}, ymin={:.3f},  dy~: {:.3f}, n: {} \n'.format(self.y[0], self.y[-1],np.max(self.y), np.min(self.y), np.mean(np.abs(self.y)), len(self.y))
        s+='| - closed     : {}\n'.format(self.closed)
        #s+='| - blunt      : {}\n'.format(self.closed)
        s+='| * orientation: {}\n'.format(self.orientation)
        s+='| * startPoint:  {}\n'.format(self.startPoint)
        s+='| * chord: {}\n'.format(self.chord)
        s+='| * thickness_max: {:.6f}\n'.format(self.thickness_max)
        s+='| * t_rel_max:     {:.6f}\n'.format(self.t_rel_max)
        #s+='| * thickness: {}\n'.format(self.thickness)
        s+='| * iTE: {}\n'.format(self.iTE)
        s+='| * iLE: {}\n'.format(self.iLE)
        return s

    # --------------------------------------------------------------------------------
    # --- Plot
    # --------------------------------------------------------------------------------
    def plot_surfaces(self):
        xu, yu, xl, yl, Iu, Il = self.split_surfaces()
        x0, y0, y0u, y0l = self.camberline()
        fig,ax = plt.subplots(1, 1, sharey=False, figsize=(6.4,4.8)) # (6.4,4.8)
        fig.subplots_adjust(left=0.12, right=0.95, top=0.95, bottom=0.11, hspace=0.20, wspace=0.20)
        ax.plot(self.x, self.y, 'k.', label='All')
        ax.plot(xu, yu, 'o'         , label='Upper', markerfacecolor='none')
        ax.plot(xl, yl, 'd'         , label='Lower', markerfacecolor='none')
        ax.plot(x0, y0, '--' , label='Camber line' )
        #ax.plot(x0, y0u, ':'  )
        #ax.plot(x0, y0l, '-.'  )
        ax.legend()
        ax.set_xlabel('x')
        ax.set_ylabel('y')
        ax.set_title(self.name)
        plt.axis ( 'equal' )
        return ax

    def plot(self, first=True, orient=True, label=None, internal=False):
        if internal:
            x = self._x
            y = self._y
        else:
            x = self.x
            y = self.y
        fig,ax = plt.subplots(1, 1, sharey=False, figsize=(6.4,4.8)) # (6.4,4.8)
        fig.subplots_adjust(left=0.12, right=0.95, top=0.95, bottom=0.11, hspace=0.20, wspace=0.20)
        ax.plot(self.x, self.y, '.-', label=label)
        
        if first:
            ax.plot(self.x[0], self.y[0], 's', label='First point')

        if orient:
            c = self.chord
            scale=0.01 * c
            dx = self.x[1] - self.x[0]
            dy = self.y[1] - self.y[0]
            ax.arrow(self.x[0], self.y[0], dx, dy, head_width=scale, head_length=scale, fc='black', ec='black')

        ax.set_xlabel('x')
        ax.set_ylabel('y')
        ax.legend()
        ax.set_aspect( 'equal' )
        return ax

def debug_slope():

    Up=np.array([[0.9947532, 0.0019938],
                 [0.9976658, 0.0015870],
                 [0.9994161, 0.0013419],
                 [1.0000000, 0.0012600]])

    Do=np.array([ [0.9947532, -.0019938],
                  [0.9976658, -.0015870],
                  [0.9994161, -.0013419],
                  [1.0000000, -.0012600]])

    ##X/c_[-]   Y/c_[-]
    #1.000000   0.001260
    #0.992704   0.002274
    #0.979641   0.004079
    #0.964244   0.006169
    #
    #0.964244  -0.006169
    #0.979641  -0.004079
    #0.992704  -0.002274
    #1.000000  -0.001260

    DxU = Up[:,0]-Up[-1,0]
    DyU = Up[:,1]-Up[-1,1]

    DxD = Do[:,0]-Do[-1,0]
    DyD = Do[:,1]-Do[-1,1]

    aU = np.arctan(DyU/DxU)*180/np.pi
    aD = np.arctan(DyD/DxD)*180/np.pi
    print('aU',aU[:-1])
    print('aD',aD[:-1])
    alpha=np.mean(-aU[:-1]+aD[:-1])
    print(alpha)

    fig,ax = plt.subplots(1, 1, sharey=False, figsize=(6.4,4.8)) # (6.4,4.8)
    fig.subplots_adjust(left=0.12, right=0.95, top=0.95, bottom=0.11, hspace=0.20, wspace=0.20)
    ax.plot(Up[:,0], Up[:,1]    , label='')
    ax.plot(Do[:,0],Do[:,1]    , label='')


    x_=np.linspace(-1,0,10)
    ax.plot(x_+1,  np.tan(alpha/2*np.pi/180)*x_)
    ax.plot(x_+1, -np.tan(alpha/2*np.pi/180)*x_)


    ax.set_xlabel('')
    ax.set_ylabel('')
    ax.legend()
    plt.show()

if __name__ == '__main__':
    from welib.airfoils.naca import naca_shape

    digits='0022'
    n=5
    x, y = naca_shape(digits, chord=1, n=n)
    print('x',x)
    print('y',y)

    arf = AirfoilShape(x=x, y=y, name='Naca'+digits)
    print('x',arf.x)
    print('y',arf.y)
    arf.write('_Naca{}.csv'.format(digits), format='csv', delim=' ')
    #arf.plot(digits)
    arf.plot_surfaces()

    plt.show()


def airfoil_info_CLI():
    parser = argparse.ArgumentParser(description="Show info about an airfoil file using StandardizedAirfoilShape.")
    parser.add_argument('input', type=str, help='Input airfoil file path')
    parser.add_argument('--plot', action='store_true', help='Plot the airfoil shape')
    parser.add_argument('--name', type=str, default='', help='Optional name for the airfoil')
    parser.add_argument('--verbose', action='store_true', help='Verbose output')
    args = parser.parse_args()

    x, y, _ = read_airfoil(args.input)

    arf = StandardizedAirfoilShape(x, y, name=args.name or args.input, verbose=args.verbose)

    print(arf)

    if args.plot:
        arf.plot()
        plt.show()

if __name__ == '__main__':

    airfoil_info_CLI()

