import os
import matplotlib.pyplot as plt
import numpy as np
import inspect
from scipy.interpolate import interp1d
from scipy.interpolate import splprep, splev

# Local
from nalulib.curves import contour_is_closed
from nalulib.curves import contour_is_clockwise
from nalulib.curves import contour_is_counterclockwise
from nalulib.curves import counterclockwise_contour
from nalulib.curves import close_contour
from nalulib.curves import open_contour
from nalulib.curves import contour_orientation
from nalulib.curves import contour_remove_duplicates
from nalulib.curves import find_closest
from nalulib.curves import opposite_contour
from nalulib.curves import reloop_contour
from nalulib.curves import curve_interp_s
from nalulib.curves import curve_interp
from nalulib.curves import curve_coord
from nalulib.curves import curve_enforce_superset

# Optional
try:
    from nalulib.airfoil_shapes_naca import naca_shape
    from nalulib.airfoil_shapes_io import read_airfoil
except:
    pass


_DEFAULT_REL_TOL=0.000001

# --- GUIDELINES FOR AIRFOIL MESHING
GUIDELINES_N_DS_HIGH = 3000  # ds=~0.00033c
GUIDELINES_N_DS_REC  = 200   # 
GUIDELINES_N_DS_LOW  = 100   # ds=~0.005  c

GUIDELINES_N_TE_BLUNT_HIGH = 20 # Number of points high res
GUIDELINES_N_TE_BLUNT_REC = 10 # Number of points recommended
GUIDELINES_N_TE_BLUNT_LOW = 5  # Number of points low res

GUIDELINES_TANGENTIAL_EXPANSION_RATIO_MAX = 1.2


# --------------------------------------------------------------------------------}
# --- Airfoil coordinates using various methods 
# --------------------------------------------------------------------------------{
def airfoil_get_xy(airfoil_coords_wrap, format=None, **kwargs):
    coords = np.asarray(airfoil_coords_wrap)
    if isinstance(airfoil_coords_wrap, str):
        basename = airfoil_coords_wrap
        name, ext = os.path.splitext(basename)
        #print(f'name {name} ext{ext}')
        if len(ext)==0:
            return airfoil_get_xy_by_string(basename, **kwargs)
        else:
            return airfoil_get_xy_by_file(basename, format=format)

    elif len(coords.shape)==2 and coords.shape[0] == 2:
        x = coords[0]
        y = coords[1]
    elif len(coords.shape)==2 and coords.shape[1] == 2:
        x = coords[:,0]
        y = coords[:,1]
    else:
        raise NotImplementedError()

    return x, y

def airfoil_get_xy_by_string(name, **kwargs):
    name = name.lower()
    if name.startswith('naca'):
        # extract digits from the name
        digits = ''.join(filter(str.isdigit, name))
        naca_params = inspect.signature(naca_shape).parameters
        filtered_kwargs = {k: v for k, v in kwargs.items() if k in naca_params}
        if len(digits) == 4:
            x, y = naca_shape(digits, **filtered_kwargs)
        else:
            raise NotImplementedError(f"NACA shape with {len(digits)} digits is not supported: {name}") 
    elif name.startswith('diamond'):
        x = np.array([1, 0.5, 0, 0.5, 1])
        y = np.array([0, 1, 0, -1, 0])
        return x, y
    else:
        raise NotImplementedError()
    return x, y

def airfoil_get_xy_by_file(filename, format=None):
    x, y, _ = read_airfoil(filename, format=format)
    return x, y

# --------------------------------------------------------------------------------}
# --- Airfoil Geometry
# --------------------------------------------------------------------------------{
def normalize_airfoil_coords(x, y, reltol=_DEFAULT_REL_TOL, verbose=False):
    x = np.asarray(x)
    y = np.asarray(y)

    # Printing info to screen
    if verbose:
        print('[INFO] input airfoil is closed:           ', contour_is_closed(x, y, reltol=reltol))
        print('[INFO] input airfoil is counterclockwise: ', contour_is_counterclockwise(x, y))

    # At first remove last point if same as the first point
    x, y = open_contour(x, y, reltol=reltol, verbose=verbose)

    # Remove duplicates
    x, y, duplicates = contour_remove_duplicates(x, y, reltol=reltol, verbose=verbose)

    # Ensure counterclockwise order
    x, y = counterclockwise_contour(x, y, verbose=verbose)

    # reloop so that first point is upper TE point
    IXmax = np.where(x == np.max(x))[0]
    iiymax = np.argmax(y[IXmax])
    iUpperTE = IXmax[iiymax]
    x, y = reloop_contour(x, y, iUpperTE, verbose=verbose)

    # At the end, close the contour
    x, y = close_contour(x, y, force=True, verbose=verbose)

    return x, y

def airfoil_is_normalized(x, y, reltol=_DEFAULT_REL_TOL, verbose=False, raiseError=True):
    """
    Checks if airfoil coordinates x, y are properly normalized according to normalized_airfoil_coords convention:
      - Contour is closed
      - Contour is counterclockwise
      - First point is the upper TE point (max x, max y)
    Returns True if all checks pass, False otherwise.
    """
    x = np.asarray(x)
    y = np.asarray(y)
    checks = []
    #if not contour_is_closed(x, y, reltol=reltol):
    checks.append(contour_is_closed(x, y, reltol=reltol))
    if not checks[-1]:
        if raiseError:
            raise Exception("Airfoil contour should be closed, but it is not. Use close_contour() to close it.")
        if verbose:
            print("[is_normalized_airfoil_coords] Contour is not closed.")

    checks.append(contour_is_counterclockwise(x, y))
    if not checks[-1]:
        if raiseError:
            raise Exception("Airfoil contour should be counterclockwise, but it is not. Use counterclockwise_contour() to fix it.")     
        if verbose:
            print("[is_normalized_airfoil_coords] Contour is not counterclockwise.")

    # First point is upper TE (max x, max y among max x)
    IXmax = np.where(x == np.max(x))[0]
    iiymax = np.argmax(y[IXmax])
    iUpperTE = IXmax[iiymax]
    checks.append(iUpperTE == 0)
    if not checks[-1]:
        if raiseError:
            raise Exception(f"First point is not upper TE (index {iUpperTE}).")
        if verbose:
            print(f"[is_normalized_airfoil_coords] First point is not upper TE (index {iUpperTE}).")

    return all(checks)


def airfoil_TE_type(x, y, closed=True):
    """ Detects the type of trailing edge (TE) of an airfoil contour."""
    if closed:
        nSharp=2
    else:
        nSharp=1
    # --- Find TE points
    IXmax = np.where(x == np.max(x))[0]
    if len(IXmax) == nSharp:
        # TODO cusps, based on y
        return 'sharp'
    elif len(IXmax) >= nSharp+1:
        return 'blunt'
    else:
        raise ValueError("No trailing edge found in the airfoil coordinates.")

def airfoil_split_surfaces(x, y, reltol=_DEFAULT_REL_TOL, verbose=False):
    """
    Split an airfoil contour into upper and lower surfaces, and find leading edge (LE) and trailing edge (TE) indices.
    This assumes that the contour is closed and counterclockwise.
    """
    # Ensure contour is open and counterclockwise
    if not contour_is_closed(x, y, reltol=reltol):
        raise Exception("Airfoil contour should be closed, but it is not. Use close_contour() to close it.")
    if not contour_is_counterclockwise(x, y):
        raise Exception("Airfoil contour should be counterclockwise, but it is not. Use counterclockwise_contour() to fix it.")     
    
    # --- TE indices
    # From lower TE to upper TE (if blunt). Upper TE should always be 0
    # 2 points if sharp TE, 3 or more if blunt TE
    IXmax = np.where(x == np.max(x))[0]
    ITE = np.concatenate((IXmax[1:], [IXmax[0]]))
    #print('>>> ITE indices:', ITE)
    assert ITE[-1] == 0, "Last TE index should be 0, but got {}".format(ITE[-1])
    assert ITE[-2] == len(x)-1, "Second last TE index should be the last point, but got {}".format(ITE[-2])

    # --- Find LE point
    IXmin = np.where(x == np.min(x))[0]
    if len(IXmin) == 1:
        iLE = IXmin[0]
    elif len(IXmin) == 2:
        print('[WARN] Two leading edge points found, using the first one.')
        iLE = IXmin[0]
    else:
        raise ValueError("{} leading edge found in the airfoil coordinates.".format(len(IXmin)))

    # --- Surfaces
    IUpper = np.arange(0, iLE+1) # Counterclockwise, include TE and LE
    ILower = np.arange(iLE, ITE[0]+1) # Counterclockwise, include LE and TE

    IAll = np.concatenate((IUpper[:-1], ILower[:-1], ITE[:-1]))
    assert np.all(IAll == np.arange(len(x))), "Indices do not match the length of x and y arrays."

    return IUpper, ILower, ITE, iLE


def airfoil_trailing_edge_angle(x, y, plot=False):
    """
    Computes the trailing edge angle using the first two upper points and the last two lower points.
    Returns a dictionary with:
        - a_up, b_up: slope and intercept for upper tangent line at TE
        - a_low, b_low: slope and intercept for lower tangent line at TE
        - angle_deg: total trailing edge angle in degrees
        - angle_up_deg: upper tangent angle (deg, from x-axis)
        - angle_low_deg: lower tangent angle (deg, from x-axis)
        - x_fit, y_fit_up, y_fit_low: points for plotting the tangent lines (optional)
    """
    IUpper, ILower, ITE, iLE = airfoil_split_surfaces(x, y)
    # Upper: first two points
    x1u, y1u = x[IUpper[0]], y[IUpper[0]]
    x2u, y2u = x[IUpper[1]], y[IUpper[1]]
    # Lower: last two points
    x1l, y1l = x[ILower[-1]], y[ILower[-1]]
    x2l, y2l = x[ILower[-2]], y[ILower[-2]]

    # Fit lines: y = a*x + b
    a_up = (y2u - y1u) / (x2u - x1u) if (x2u - x1u) != 0 else np.inf
    b_up = y1u - a_up * x1u

    a_low = (y2l - y1l) / (x2l - x1l) if (x2l - x1l) != 0 else np.inf
    b_low = y1l - a_low * x1l

    # Angles (in radians)
    theta_up = np.arctan(a_up)
    theta_low = np.arctan(a_low)
    # Trailing edge angle (between tangents)
    angle_rad = np.abs(theta_up - theta_low)
    angle_deg = np.degrees(angle_rad)

    result = {
        'a_up': a_up,
        'b_up': b_up,
        'a_low': a_low,
        'b_low': b_low,
        'angle_deg': angle_deg,
        'angle_up_deg': np.degrees(theta_up),
        'angle_low_deg': np.degrees(theta_low),
    }
    # For plotting: fit lines over a small x-range near TE
    chord = np.max(x) - np.min(x)
    x_te = x[IUpper[0]]
    x_fit = np.linspace(x_te - 0.05*chord, x_te + 0.05*chord, 3)
    y_fit_up = a_up * x_fit + b_up
    y_fit_low = a_low * x_fit + b_low
    result['x_fit'] = x_fit
    result['y_fit_up'] = y_fit_up
    result['y_fit_low'] = y_fit_low
    if plot:
        fig,ax = plt.subplots(1, 1, sharey=False, figsize=(12.8,4.0)) # (6.4,4.8)
        fig.subplots_adjust(left=0.12, right=0.95, top=0.95, bottom=0.11, hspace=0.20, wspace=0.20)
        ax.plot(x, y, 'k-', label='Airfoil contour')
        ax.plot(x_fit, y_fit_up, 'r--', label='Upper tangent line')
        ax.plot(x_fit, y_fit_low, 'b--', label='Lower tangent line')
        ax.scatter([x1u, x2u], [y1u, y2u], color='red', label='Upper TE points')
        ax.scatter([x1l, x2l], [y1l, y2l], color='blue', label='Lower TE points')
        ax.set_xlabel('x')
        ax.set_ylabel('y')
        ax.set_title(f'Trailing Edge Angle: {angle_deg:.2f}°')
        ax.legend()
        ax.set_aspect('equal')
        ax.grid()

    return angle_deg, result

def airfoil_leading_edge_radius(x, y, verbose=False, plot=False):
    """
    Estimate the leading edge radius of an airfoil by fitting a circle to points around the LE.
    The points are automatically selected as those closest to the LE (min x), up to the inflection
    (where curvature changes sign) or a reasonable arc length threshold.
    Returns a dictionary with:
        - radius: estimated leading edge radius
        - xc, yc: center of fitted circle
        - x_fit, y_fit: points used for fitting
        - x_circle, y_circle: points on the fitted circle (for plotting)
    If plot=True, also plots the airfoil, fit points, and fitted circle.

    NOTE: TODO TODO TODO LEADING EDGE RADIUS IS NOT FINISHED
    """
    x = np.asarray(x)
    y = np.asarray(y)
    _, _, _, iLE = airfoil_split_surfaces(x, y)

    # Find points near the LE (min x)
    xLE, yLE = x[iLE], y[iLE]
    s = curve_coord(x, y, normalized=False)
    sLE = s[iLE]

    # Compute distance from LE for all points
    dLE = np.sqrt((x - xLE)**2 + (y - yLE)**2)

    # Heuristic: select points within a small arc length from LE (e.g., 5% of chord)
    chord = np.max(x) - np.min(x)
    arc_thresh = 0.05 * chord
    idx_near = [iLE]
    for direction in [-1, 1]:
        i = iLE
        arc = 0.0
        prev_sign = None
        while True:
            i_next = (i + direction) % len(x)
            ds = np.sqrt((x[i_next] - x[i])**2 + (y[i_next] - y[i])**2)
            arc += ds
            if arc > arc_thresh:
                break
            # Compute curvature sign
            if 1 < i_next < len(x) - 1:
                dx1 = x[i_next] - x[i]
                dy1 = y[i_next] - y[i]
                dx2 = x[(i_next + direction) % len(x)] - x[i_next]
                dy2 = y[(i_next + direction) % len(y)] - y[i_next]
                cross = dx1 * dy2 - dy1 * dx2
                sign = np.sign(cross)
                if prev_sign is not None and sign != prev_sign and sign != 0:
                    break  # Inflection point detected
                prev_sign = sign
            idx_near.append(i_next)
            i = i_next

    idx_near = np.unique(idx_near)
    x_fit = x[idx_near]
    y_fit = y[idx_near]

    # Fit a circle to these points
    def fit_circle(xs, ys):
        # Algebraic circle fit (Kasa method)
        A = np.c_[2*xs, 2*ys, np.ones(xs.shape)]
        b = xs**2 + ys**2
        c, resid, rank, s = np.linalg.lstsq(A, b, rcond=None)
        xc, yc, d = c
        r = np.sqrt(xc**2 + yc**2 + d)
        return xc, yc, r

    xc, yc, r = fit_circle(x_fit, y_fit)

    # For plotting: points on the fitted circle
    theta = np.linspace(0, 2 * np.pi, 100)
    x_circle = xc + r * np.cos(theta)
    y_circle = yc + r * np.sin(theta)

    result = {
        'radius': r,
        'xc': xc,
        'yc': yc,
        'x_fit': x_fit,
        'y_fit': y_fit,
        'x_circle': x_circle,
        'y_circle': y_circle,
    }

    if verbose:
        print(f"[airfoil_leading_edge_radius] Fitted circle center: ({xc:.5f}, {yc:.5f}), radius: {r:.5f}, using {len(x_fit)} points.")

    if plot:
        fig, ax = plt.subplots(1, 1, figsize=(8, 4))
        ax.plot(x, y, 'k-', label='Airfoil contour')
        ax.scatter(x_fit, y_fit, color='red', label='Fit points')
        ax.plot(x_circle, y_circle, 'b--', label='Fitted circle')
        ax.scatter([xc], [yc], color='blue', marker='x', label='Circle center')
        ax.scatter([x[iLE]], [y[iLE]], color='green', marker='o', label='LE point')
        ax.set_aspect('equal')
        ax.set_xlabel('x')
        ax.set_ylabel('y')
        ax.set_title(f'Leading Edge Radius: {r:.5f}')
        ax.legend()
        ax.grid()

    return r, result


# --- OLD LEGACY FUNCTIONS
def reindex_starting_from_te(coords, TE_indices):
    raise Exception('reindex_starting_from_te is deprecated, use airfoil_TE_type() instead.')
    x, y = reloop_contour(coords[:,0], coords[:,1], TE_indices[0])
    return 

def detect_airfoil_te_type(coords):
    raise Exception('detect_airfoil_te_type() is deprecated, use airfoil_TE_type() instead.')
    x,y = coords[:,0], coords[:,1]
    IXmax = np.where(x == np.max(x))[0]
    return airfoil_TE_type(x, y, closed=True), IXmax

def detect_airfoil_features(coords):
    raise Exception('detect_airfoil_features() is deprecated, use airfoil_split_surfaces instead.')
    indices = { "trailing_edge": ITE, "leading_edge": iLE, "upper_surface": IUpper, "lower_surface": ILower}
    return trailing_edge_type, indices


# --------------------------------------------------------------------------------}
# --- Airfoil Remeshing
# --------------------------------------------------------------------------------{
def cosine_spacing_s(n, reverse=False):
    """
    Returns n indices spaced according to cosine spacing (0 at LE, 1 at TE).
    If reverse=True, flips the order.
    """
    beta = np.linspace(0, np.pi, n)
    s = 0.5 * (1 - np.cos(beta)) # Cosine spacing from 0 to 1
    if reverse:
        s = s[::-1]
    return s

def hyperbolic_tangent_spacing_s(n, a=2.5, reverse=False):
    """
    Returns n indices spaced according to hyperbolic tangent spacing (0 at LE, 1 at TE).
    'a' controls clustering: higher a = more clustering at ends.
    If reverse=True, flips the order.
    """
    t = np.linspace(0, 1, n)
    s = 0.5 * (1 + np.tanh(a * (t - 0.5)) / np.tanh(a / 2))
    if reverse:
        s = s[::-1]
    return s

def resample_airfoil(x, y, IUpper, ILower, ITE, interp_upper, interp_lower, interp_te=None):
    """
    Generic airfoil resampling function.
    Allows custom interpolation functions for upper, lower, and TE surfaces.
    Each interp_* function should have the signature:
        interp(x, y) -> x_new, y_new
    """
    # Upper surface
    xu, yu = x[IUpper], y[IUpper]
    xu2, yu2 = interp_upper(xu, yu)
    xu2[0], yu2[0] = xu[0], yu[0]
    # Lower surface
    xl, yl = x[ILower], y[ILower]
    xl2, yl2 = interp_lower(xl, yl)
    # TE 
    x_te, y_te = x[ITE][:-1], y[ITE][:-1]
    if interp_te is not None and len(ITE) > 2:
        x_te_new, y_te_new = interp_te(x_te, y_te)

    else:
        x_te_new, y_te_new = x_te, y_te # No interpolation for TE if sharp

    x_new = np.concatenate([xu2[:-1], xl2[:-1], x_te_new[:-1], [xu2[0]] ])
    y_new = np.concatenate([yu2[:-1], yl2[:-1], y_te_new[:-1], [yu2[0]] ])

    return x_new, y_new

def resample_airfoil_ul(x, y, IUpper, ILower, ITE, interp_ul, interp_te=None):
    """
    Generic airfoil resampling function.
    Allows custom interpolation functions for upper+lower, and TE surfaces.
    Each interp_* function should have the signature:
        interp(x, y) -> x_new, y_new
    """
    # Upper + Lower surface
    xu, yu = x[IUpper], y[IUpper]
    xl, yl = x[ILower], y[ILower]
    xul = np.concatenate([xu, xl[1:]])
    yul = np.concatenate([yu, yl[1:]])
    if interp_ul is not None:
        x_new, y_new = interp_ul(xul, yul)
        # Safety
        x_new[0], y_new[0] = xul[0], yul[0]
        x_new[-1], y_new[-1] = xul[-1], yul[-1]
    else:
        x_new, y_new = xul, yul
    
    # TE 
    x_te, y_te = x[ITE][:-1], y[ITE][:-1]
    if interp_te is not None and len(ITE) > 2:
        x_te_new, y_te_new = interp_te(x_te, y_te)
    else:
        x_te_new, y_te_new = x_te, y_te  # No interpolation for TE if sharp

    x_new = np.concatenate([x_new[:-1], x_te_new[:-1], [x_new[0]] ])
    y_new = np.concatenate([y_new[:-1], y_te_new[:-1], [y_new[0]] ])

    return x_new, y_new

def resample_airfoil_refine(x, y, IUpper, ILower, ITE, factor_surf=2, factor_te=1):
    """
    Refine the airfoil grid by increasing the number of points by a given factor,
    preserving the original points (all original points are included in the output).
    The new points are inserted between the originals using cubic spline interpolation
    for the main contour and linear interpolation for the TE.
    Returns new x, y arrays (closed, counterclockwise).
    """

    if factor_surf >1:
        def interp_ul(xul, yul):
            # Parameterize by cumulative chordwise distance
            s = curve_coord(xul, yul, normalized=True)
            n_orig = len(xul)
            # Subdivide each segment into factor_surf parts
            s_new = []
            for i in range(n_orig - 1):
                s_sub = np.linspace(s[i], s[i+1], factor_surf + 1)[:-1]  # exclude last to avoid duplicates
                s_new.extend(s_sub)
            s_new.append(s[-1])  # add the last point
            s_new = np.array(s_new)
            tck, _ = splprep([xul, yul], u=s, s=0, k=min(3, n_orig-1))
            x_new, y_new = splev(s_new, tck) # return x_new, y_new
            x_new, y_new = curve_enforce_superset(xul, yul, x_new, y_new, verbose=False, raiseError=False)
            return x_new, y_new
    else:
        interp_ul = None

    if factor_te > 1:
        def interp_te(xte, yte):
            if len(xte) < 2:
                return xte, yte
            s = curve_coord(xte, yte, normalized=True)
            n_orig = len(xte)
            # Subdivide each segment into factor_surf parts
            s_new = []
            for i in range(n_orig - 1):
                s_sub = np.linspace(s[i], s[i+1], factor_te + 1)[:-1]  # exclude last to avoid duplicates
                s_new.extend(s_sub)
            s_new.append(s[-1])  # add the last point
            s_new = np.array(s_new)
            # Linear interpolation
            fx = interp1d(s, xte, kind='linear')
            fy = interp1d(s, yte, kind='linear')
            x_new, y_new = fx(s_new), fy(s_new)
            x_new, y_new = curve_enforce_superset(xte, yte, x_new, y_new, verbose=True, raiseError=True)
            return x_new, y_new
    else:
        interp_te = None

    x_new, y_new = resample_airfoil_ul(x, y, IUpper, ILower, ITE, interp_ul=interp_ul, interp_te=interp_te)
    return x_new, y_new


# KEEP ME FOR NOW:

def resample_airfoil_hyperbolic(x, y, IUpper, ILower, ITE, n_surf=80, a_surf=2.5, interp_te=None):
    """
    Resample upper, lower, and TE surfaces with hyperbolic tangent and constant spacing.
    Returns new x, y arrays (closed, counterclockwise).
    """
    interp_upper = lambda x_, y_: curve_interp_s(x_, y_, s_new=hyperbolic_tangent_spacing_s(n_surf, a=a_surf), normalized=True)
    interp_lower = interp_upper
    x_new, y_new = resample_airfoil(x, y, IUpper, ILower, ITE, interp_upper, interp_lower, interp_te)  

    return x_new, y_new

def resample_airfoil_cosine(x, y, IUpper, ILower, ITE, n_surf=80, interp_te=None):
    """
    Resample upper, lower, and TE surfaces with cosine and constant spacing.
    Returns new x, y arrays (closed, counterclockwise).
    """
    interp_upper = lambda x_, y_: curve_interp_s(x_, y_, s_new=cosine_spacing_s(n_surf), normalized=True)
    interp_lower = interp_upper
    x_new, y_new = resample_airfoil(x, y, IUpper, ILower, ITE, interp_upper, interp_lower, interp_te)  

    return x_new, y_new

def resample_airfoil_spline(x, y, IUpper, ILower, ITE, n_surf=1000, kind='cubic', interp_te=None):
    """
    Resample upper and lower surfaces using spline interpolation (default cubic).
    The TE is not modified: it is appended at the end (constant spacing or original).
    Returns new x, y arrays (closed, counterclockwise).
    """

    def interp_ul(xul, yul):
        # Parameterize by cumulative chordwise distance
        u = curve_coord(xul, yul, normalized=True)
        # Spline fit
        tck, _ = splprep([xul, yul], u=u, s=0, k=3 if kind == 'cubic' else 1)
        u_new = np.linspace(0, 1, 2 * n_surf)
        return splev(u_new, tck)

    x_new, y_new = resample_airfoil_ul(x, y, IUpper, ILower, ITE, interp_ul, interp_te)  

    return x_new, y_new






# --------------------------------------------------------------------------------}
# --- CFD checks
# --------------------------------------------------------------------------------{
def check_airfoil_mesh(x, y, IUpper, ILower, ITE, Re=1e6):
    """
    Checks if the airfoil surface mesh satisfies IDDES surface resolution criteria.

    Parameters:
        x, y: arrays of surface coordinates (ordered CCW, first point upper TE)
        IUpper: indices of upper surface
        ILower: indices of lower surface
        ITE: indices defining the blunt trailing edge
        Re: Reynolds number (default 1e6)
    """
    c = np.max(x)-np.min(x) # chord length

    # Rule of thumb criteria 
    ds_min = c / GUIDELINES_N_DS_HIGH  # ~0.00033
    ds_max = c / GUIDELINES_N_DS_LOW   # ~0.005

    # ------------- Helper for surface checks -------------
    def check_surface(ds, label):
        min_ds = ds.min()
        max_ds = ds.max()
        print(f"{label}: min ds = {min_ds:.5f}, max ds = {max_ds:.5f}")
        if min_ds >= ds_min:
            print(f"✅ {label} minimum spacing OK.")
        else:
            print(f"❌ {label} minimum spacing too small.")
        if max_ds <= ds_max:
            print(f"✅ {label} maximum spacing OK.")
        else:
            print(f"❌ {label} maximum spacing too large.")

        # Expansion ratio check:
        ratios = ds[1:] / ds[:-1]
        max_ratio = np.max(ratios)
        min_ratio = np.min(ratios)
        print(f"{label}: max expansion ratio = {max_ratio:.3f}, min expansion ratio = {min_ratio:.3f}")
        if max_ratio <= GUIDELINES_TANGENTIAL_EXPANSION_RATIO_MAX:
            print(f"✅ {label} expansion ratio OK.")
        else:
            print(f"❌ {label} expansion ratio too large, consider smoothing.")

    # ---------------- Upper/Lower surfaces --------------
    xu, yu = x[IUpper], y[IUpper]
    xl, yl = x[ILower], y[ILower]
    dsu = np.sqrt(np.diff(xu)**2 + np.diff(yu)**2)
    dsl = np.sqrt(np.diff(xl)**2 + np.diff(yl)**2)
    check_surface(dsu, "Upper surface")
    check_surface(dsl, "Lower surface")

    # ---------------- Blunt Trailing Edge ----------------
    if len(ITE) >= 3:
        xTE = x[ITE]
        yTE = y[ITE]
        h_TE = np.abs(yTE.max() - yTE.min())
        ds_TE_crit_min = h_TE / GUIDELINES_N_TE_BLUNT_HIGH
        ds_TE_crit_max = h_TE / GUIDELINES_N_TE_BLUNT_LOW
        dsTE = np.sqrt(np.diff(xTE)**2 + np.diff(yTE)**2)
        min_dsTE = dsTE.min()
        max_dsTE = dsTE.max()
        print(f"Blunt TE: min ds = {min_dsTE:.5f}, max ds = {max_dsTE:.5f}, h_TE = {h_TE:.5f}")
        if min_dsTE <= ds_TE_crit_min:
            print("✅ Blunt TE minimum spacing OK.")
        else:
            print("❌ Blunt TE minimum spacing too large.")
        if max_dsTE <= ds_TE_crit_max:
            print("✅ Blunt TE maximum spacing OK.")
        else:
            print("❌ Blunt TE maximum spacing too large.")

        # Expansion ratio for TE:
        ratios_TE = dsTE[1:] / dsTE[:-1]
        max_ratio_TE = np.max(ratios_TE)
        min_ratio_TE = np.min(ratios_TE)
        print(f"Blunt TE: max expansion ratio = {max_ratio_TE:.3f}, min expansion ratio = {min_ratio_TE:.3f}")
        if max_ratio_TE <= GUIDELINES_TANGENTIAL_EXPANSION_RATIO_MAX:
            print("✅ Blunt TE expansion ratio OK.")
        else:
            print("❌ Blunt TE expansion ratio too large, consider smoothing.")
    else:
        print("🔹 Sharp TE")














# ---------------------------------------------------------------------------
# --- Plot  Airfoil library
# ---------------------------------------------------------------------------
def plot_normalized(x, y, first=True, orient=True, label=None, title='', ax=None, simple=False, sty='k.-'):
    """ Plot airfoil coordinates if normalized using airfoil_normalize_coords()."""
    airfoil_is_normalized(x, y, reltol=_DEFAULT_REL_TOL, verbose=False, raiseError=True)

    IUpper, ILower, ITE, iLE = airfoil_split_surfaces(x, y, reltol=_DEFAULT_REL_TOL, verbose=False)
    xu,  yu  = x[IUpper], y[IUpper]      
    xl,  yl  = x[ILower], y[ILower]
    xTE, yTE = x[ITE]   , y[ITE]
    xLE, yLE = x[iLE]   , y[iLE]

    """ Plot coordinates of an airfoil contour in normalized coordinates."""
    if ax is None:
        fig,ax = plt.subplots(1, 1, sharey=False, figsize=(12.8,4.0)) # (6.4,4.8)
        fig.subplots_adjust(left=0.12, right=0.95, top=0.95, bottom=0.11, hspace=0.20, wspace=0.20)

    # Main plot
    ax.plot(x, y, sty, label=label)

    if not simple:
        ax.plot(xu , yu,   '-o', label='Upper surface', markerfacecolor='none', markersize=8)
        ax.plot(xl , yl,   '-d', label='Lower surface', markerfacecolor='none', markersize=7)     
        ax.plot(xTE, yTE, '-x', label='Trailing edge', markerfacecolor='none', markersize=6)
        ax.plot(xLE, yLE, '^',  label='Leading edge', markerfacecolor='none', markersize=6)


    if first and not simple:
        ax.plot(x[0], y[0], 's', label='First point', markerfacecolor='none', markersize=10)

    if orient and not simple:
        c = np.max(x) - np.min(x)  # Chord length
        scale=0.01 * c
        dx = x[1] - x[0]
        dy = y[1] - y[0]
        ax.arrow(x[0], y[0], dx, dy, head_width=scale, head_length=scale, fc='black', ec='black')

    ax.set_xlabel('x')
    ax.set_ylabel('y')
    ax.legend()
    ax.set_aspect( 'equal' )
    ax.set_title(title)
    return ax


# ---------------------------------------------------------------------------
# --- OLD Airfoil library
# ---------------------------------------------------------------------------
def load_airfoil_coords(filename):
    coords = np.loadtxt(filename)
    coords = coords[:, :2] # Keep only x and y coordinates
    # Remove duplicates
    _, unique_indices = np.unique(coords, axis=0, return_index=True)
    if len(unique_indices) < coords.shape[0]:
        print(f"Removed {coords.shape[0] - len(unique_indices)} duplicate points from airfoil coordinates.")
        coords = coords[np.sort(unique_indices)]
    # Ensure airfoil is closed
    if not np.allclose(coords[0], coords[-1]):
        print("Adding closing point to airfoil coordinates to form closed contour.")
        coords = np.vstack([coords, coords[0]])
    # Check if the points are ordered clockwise or counterclockwise
    if contour_is_clockwise(coords):
        print("Reversing airfoil coordinates to ensure counterclockwise order.")
        coords = coords[::-1]
    return coords



def debug_airfoil_plot(coords, normals, indices, normals2=None):
    """
    Debug function to visualize the airfoil with different zoom levels.
    Creates one plot at the top (spanning two columns) and two zoomed plots at the bottom.
    Optionally plots a second set of normals (normals2).
    """
    trailing_edge_indices = indices["trailing_edge"]
    leading_edge_index = indices["leading_edge"]
    upper_indices = indices["upper_surface"]
    lower_indices = indices["lower_surface"]

    # Create a figure with a custom grid layout
    fig = plt.figure(figsize=(12, 8))
    fig.subplots_adjust(left=0.05, right=0.99, top=0.99, bottom=0.05, hspace=0.04, wspace=0.04)
    grid = fig.add_gridspec(2, 2, height_ratios=[1, 1])

    # Full airfoil plot (spanning two columns)
    full_ax = fig.add_subplot(grid[0, :])
    zoom_te_ax = fig.add_subplot(grid[1, 0])
    zoom_le_ax = fig.add_subplot(grid[1, 1])

    # Define zoom levels
    zooms = [
        {"ax": full_ax, "title": "Full Airfoil", "indices": np.arange(len(coords))},
        {"ax": zoom_te_ax, "title": "Trailing Edge Zoom",
         "indices": np.unique(np.concatenate([trailing_edge_indices,
                                               np.clip(trailing_edge_indices - 1, 0, len(coords) - 1),
                                               np.clip(trailing_edge_indices + 1, 0, len(coords) - 1)]))},
        {"ax": zoom_le_ax, "title": "Leading Edge Zoom",
         "indices": np.unique(np.concatenate([[leading_edge_index],
                                               np.clip([leading_edge_index - 1, leading_edge_index + 1],
                                                       0, len(coords) - 1)]))},
    ]

    for zoom in zooms:
        ax = zoom["ax"]
        zoom_indices = zoom["indices"]
        zoom_coords = coords[zoom_indices]

        # Plot airfoil points
        ax.plot(coords[:, 0], coords[:, 1], 'k-', label="Airfoil")

        # Plot upper and lower surface points with larger markers
        ax.scatter(coords[upper_indices, 0], coords[upper_indices, 1],
                   color="green", label="Upper Surface", s=50, zorder=5)
        ax.scatter(coords[lower_indices, 0], coords[lower_indices, 1],
                   color="red", label="Lower Surface", s=50, zorder=5)

        # Plot trailing edge points with smaller markers
        ax.scatter(coords[trailing_edge_indices, 0], coords[trailing_edge_indices, 1],
                   color="orange", label="Trailing Edge", s=20, zorder=5)

        # Plot leading edge point with smaller marker
        ax.scatter(coords[leading_edge_index, 0], coords[leading_edge_index, 1],
                   color="blue", label="Leading Edge", s=20, zorder=5)

        # Plot normals
        ax.quiver(zoom_coords[:, 0], zoom_coords[:, 1],
                  normals[zoom_indices, 0], normals[zoom_indices, 1],
                  color="purple", scale=10, label="Normals")

        # Optionally plot the second set of normals (normals2)
        if normals2 is not None:
            ax.quiver(zoom_coords[:, 0], zoom_coords[:, 1],
                      normals2[zoom_indices, 0], normals2[zoom_indices, 1],
                      color="cyan", scale=10, alpha=0.6, label="Original Normals")

        # Indicate orientation
        ax.annotate("Start", (coords[0, 0], coords[0, 1]), color="black", fontsize=10,
                    xytext=(-15, 15), textcoords="offset points",
                    arrowprops=dict(arrowstyle="->", color="black"))
        ax.annotate("End", (coords[-1, 0], coords[-1, 1]), color="black", fontsize=10,
                    xytext=(-15, -15), textcoords="offset points",
                    arrowprops=dict(arrowstyle="->", color="black"))

        # Set zoom level dynamically
        x_min, x_max = zoom_coords[:, 0].min(), zoom_coords[:, 0].max()
        x_center = (x_min + x_max) / 2
        x_range = max((x_max - x_min) * 2, 0.05)  # Set xlim to twice the width
        ax.set_xlim(x_center - x_range / 2, x_center + x_range / 2)

        y_min, y_max = zoom_coords[:, 1].min(), zoom_coords[:, 1].max()
        y_center = (y_min + y_max) / 2
        y_range = max((y_max - y_min) * 2, 0.05)  # Set ylim to twice the height
        ax.set_ylim(y_center - y_range / 2, y_center + y_range / 2)

        ax.set_title(zoom["title"])
        ax.set_aspect('equal')
        ax.legend()



