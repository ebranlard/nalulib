""" 
Tools to manipulate 2D curves

In particular: 

 - tools to interpolate a curve at equidistant curvilinear spacing.
 - improved sream Quiver compare to native pyplot tool, for better handling of the arrows placements

"""

import numpy as np
import unittest
_DEFAULT_REL_TOL=1e-12

def get_line(x, y, close_it=False):
    if close_it:
        x, y = close_contour(x, y)

    line = np.zeros((len(x), 2))
    line[:,0] = x
    line[:,1] = y
    return line
    #return np.array([x,y]).T

def curve_coord(x=None, y=None, line=None, normalized=False):
    """ return curvilinear coordinate """
    if line is not None:
        x = line[:,0]
        y = line[:,1]
    x = np.asarray(x)
    y = np.asarray(y)
    s     = np.zeros(x.shape)
    s[1:] = np.sqrt((x[1:]-x[0:-1])**2+ (y[1:]-y[0:-1])**2)
    s     = np.cumsum(s)                                  
    if normalized:
        s = s/s[-1]  # Normalizing to [0,1]
    return s


def curve_extract(line, spacing, offset=None):
    """ Extract points at equidistant space along a curve
    NOTE: curve_extract and curve_interp do basically the same, one uses "spacing", the other uses n
    """
    x=line[:,0]
    y=line[:,1]
    if offset is None:
        offset=spacing/2
    # Computing curvilinear length
    s = curve_coord(line=line)
    offset=np.mod(offset,s[-1]) # making sure we always get one point
    # New (equidistant) curvilinear coordinate
    sExtract=np.arange(offset,s[-1],spacing)
    # Interpolating based on new curvilinear coordinate
    xx=np.interp(sExtract,s,x);
    yy=np.interp(sExtract,s,y);
    return np.array([xx,yy]).T

def curve_interp_s(x, y, s_new, normalized=False):
    s_old = curve_coord(x, y, normalized=normalized)
    xx = np.interp(s_new, s_old, x)
    yy = np.interp(s_new, s_old, y)
    return xx, yy

def curve_interp(x=None, y=None, n=None, s=None, ds=None, line=None, keepOri=False):
    """ Interpolate a curves to equidistant curvilinear space between points

    INPUTS:
     - either:
       -  x,y : 1d arrays 
        or
       -  line: (n_in x 2) array
     - either
        - n : number of points to interpolate
         or
        - s : array of curvilinear coordinates where we'll interpolate
         or
        - ds : equispacing 
      - keepOri: keep the original datapoints
    OUTPUTS:
      - x_new, y_new 
    """
    # --- Sanitization
    MatOut=False
    if line is None:
        line = get_line(x,y)
    else:
        x=line[:,0]
        y=line[:,1]
        MatOut=True
    if type(x) is not np.ndarray:
        x=np.array(x)
        y=np.array(y)
    if len(x)!=len(y):
        raise Exception('Inputs should have the same length')
    if len(x)<2 or len(y)<2:
        if MatOut:
            return get_line(x,y)
        else:
            return x,y

    # Computing curvilinear length
    s_old = curve_coord(line=line)
    if n is not None:
        # --- Interpolate based on n, equidistant curvilinear coordinates
        # New (equidistant) curvilinear coordinate
        s_new = np.linspace(0, s_old[-1], n);
    elif s is not None:
        # Use new curvilinear coordinate
        if s[0]<s_old[0] or s[-1]>s_old[-1]:
            raise Exception('curve_interp: Curvilinear coordinate needs to be between 0 and {}, currently it is between {} and {}'.format(s_old[-1], s[0], s[-1]))
        s_new = s
    elif ds is not None:
        s_new = np.arange(0, s_old[-1]+ds/2, ds)
    else:
        raise NotImplementedError()

    # --- Preseve original points if requested
    if keepOri:
        if ds is not None:
            # Call dedicated function, will maintain equispacing in between original points
            s_new = equispace_preserving(s_old, ds)
        else:
            # Insert old points, and spread neightbors to make the insertion look uniform
            s_new = insert_uniformly(s_new, s_old)
        # Ensure all original s_old are in s_new
        try:
            assert np.allclose(np.isin(np.around(s_old,5), np.around(s_new,5)), True), "Some original points lost."
        except:
            print(s_old)
            print(s_new)
            print('[WARN] Some original points lost.')

    # --- Interpolating based on new curvilinear coordinate
    xx = np.interp(s_new, s_old, x)
    yy = np.interp(s_new, s_old, y)

    if MatOut:
        return get_line(xx, yy) 
    else:
        return xx,yy

def equispace_preserving(s_old, ds, tol=1e-8):
    """ 
    Attempts to keep a constant ds, but preserves original datapoint
    """
    s_new = [s_old[0]]
    for i in range(len(s_old) - 1):
        s0, s1 = s_old[i], s_old[i + 1]
        seg_len = s1 - s0
        s_new.append(s1)  # always keep original points
        if seg_len > ds + tol:
            ds_approx = seg_len/(int(seg_len/ds)+0)
            inserted = np.arange(s0, s1+ds_approx/2, ds_approx)
            #n_insert = int(np.floor(seg_len / ds))
            #inserted = np.linspace(s0, s1, n_insert+1)
            s_new += (inserted.tolist())[1:-1] 
    s_new = np.array(sorted(set(s_new)))  # avoid duplicates, ensure increasing
    return s_new

def insert_uniformly(s, s_old, tol=1e-10, thresh=0.2):
    """
    Insert points from s_old into s, adjusting surrounding points to maintain local uniformity.
    
    Parameters:
      - s: desired initial curvilinear grid (1D array)
      - s_old: required curvilinear points to insert (1D array)
    
    Returns:
      - s_new: adjusted curvilinear grid with all s_old inserted
    """
    s = list(np.sort(np.unique(s)))
    s_old = np.sort(np.unique(s_old))

    np.set_printoptions(linewidth=300, precision=3)
    
    for so in s_old:
        if any(np.isclose(so, s, atol=tol)):
            continue  # already present
        
        # Find where to insert
        for j in range(len(s) - 1):
            if s[j] < so < s[j + 1]:
                break
        else:
            raise ValueError(f"s_old value {so} is outside the range of s.")

        panel_length = s[j+1] - s[j]
        threshold = thresh * panel_length

        # Case 1: very close to s[j]
        if abs(so - s[j]) < threshold:
            s[j] = so
            continue
        # Case 2: very close to s[j+1]
        if abs(so - s[j+1]) < threshold:
            s[j+1] = so
            continue
         # Case 3: insert and shift up to 4 points if room
        if j >= 2 and j + 2 < len(s):
            d = (s[j+2] - s[j-1]) / 5
            dr = (s[j+3] - so) / 3
            dl = (so - s[j-2]) / 3
            new_pts = [so - 2*dl, so - dl, so, so + dr, so + 2*dr]
            s = s[:j-1] + new_pts + s[j+3:]
        else:
            if j < 1 or j + 2 >= len(s):
                # Not enough neighbor to adjust..
                s = s[:j+1] + [so] + s[j:] 
            else:
                # adjusting two neighbors
                dr = (s[j + 2] - so) / 2
                dl = (so - s[j-1]) / 2
                new_pts = [so - dl, so, so + dr]
                s = s[:j] + new_pts + s[j+2:] 
        s = list(np.sort(np.unique(s)))  # clean up and sort
    return np.array(s)



# --------------------------------------------------------------------------------
# --- Contour querry
# --------------------------------------------------------------------------------
def contour_is_closed(x, y, reltol=_DEFAULT_REL_TOL):
    """ Return true if contour is closed """
    l = contour_length_scale(x, y)
    return np.abs(x[0]-x[-1])<l*reltol or np.abs(y[0]-y[-1])<l*reltol

def contour_is_clockwise(coords, y=None):
    if y is None:
        area = 0.5 * np.sum(coords[:-1, 0] * coords[1:, 1] - coords[1:, 0] * coords[:-1, 1])
    else:
        x = coords
        area = 0.5 * np.sum(x[:-1] * y[1:] - x[1:] * y[:-1])
    return area < 0  # Clockwise if the area is negative

def contour_is_counterclockwise(coords, y=None):
    return not contour_is_clockwise(coords, y=y)

def contour_length_scale(x, y):
    """ return representative length scale of contour """
    lx = np.max(x)-np.min(x)
    ly = np.max(y)-np.min(y)
    return  max(lx, ly)

def contour_orientation(x, y):
    """
    Determine if a contour is clockwise or counterclockwise.
    
    INPUTS:
    - x: 1D array containing the x-coordinates of the contour nodes.
    - y: 1D array containing the y-coordinates of the contour nodes.
    
    OUTPUTS:
    - 'clockwise' if the contour is clockwise, 'counterclockwise' if it's counterclockwise.
    """
    # Compute the signed area
    signed_area = np.sum(x[:-1] * y[1:] - x[1:] * y[:-1])
    
    # Determine orientation
    if signed_area > 0:
        return 'counterclockwise' # Positive about z
    elif signed_area < 0:
        return 'clockwise' # Negative about z
    else:
        return 'undetermined'

def closest_point(x0, y0, x, y):
    """ return closest point to curve and index"""
    i = np.argmin((x - x0)**2 + (y - y0)**2)
    return x[i], y[i], i

def find_closest(X, Y, point, xlim=None, ylim=None):
    """Return closest point(s), using norm2 distance 
    if xlim and ylim is provided, these are used to make the data non dimensional.
    """
    # NOTE: this will fail for datetime
    if xlim is not None:
        x_scale = (xlim[1]-xlim[0])**2
        y_scale = (ylim[1]-ylim[0])**2
    else:
        x_scale = 1
        y_scale = 1

    norm2 = ((X-point[0])**2)/x_scale + ((Y-point[1])**2)/y_scale
    ind = np.argmin(norm2, axis=0)
    return X[ind], Y[ind], ind

def point_in_contour(X, Y, contour, method='ray_casting'):
    """
    Checks if a point is inside a closed contour.
 
    INPUTS:
      - X: scalar or array of point coordinates
      - Y: scalar or array of point coordinates
      - contour: A numpy array shape (n x 2), of (x, y) coordinates representing the contour.
            [[x1, y1]
               ...
             [xn, yn]]
         or [(x1,y1), ... (xn,yn)]
 
    OUTPUTS:
      - logical or array of logical: True if the point is inside the contour, False otherwise.
    """
    def __p_in_c_ray(x, y, contour):
        # --- Check if a point is inside a polygon using Ray Casting algorithm.
        n = len(contour)
        inside = False
        p1x, p1y = contour[0]
        for i in range(n+1):
            p2x, p2y = contour[i % n]
            if y > min(p1y, p2y):
                if y <= max(p1y, p2y):
                    if x <= max(p1x, p2x):
                        if p1y != p2y:
                            xinters = (y - p1y) * (p2x - p1x) / (p2y - p1y) + p1x
                        if p1x == p2x or x <= xinters:
                            inside = not inside
            p1x, p1y = p2x, p2y
        return inside

    def __p_in_c_cv2(x, y, contour):
       # --- CV2
       import cv2 # pip install opencv-python
       # NOTE: not working well...
       contour = contour.reshape((-1,1,2))
       dist = cv2.pointPolygonTest(contour.T, P, True)
       if dist > 0:
           return True
       else:
           return False
    # --- End subfunctions

    # --- Sanity
    contour = np.asarray(contour)
    assert(contour.shape[1]==2)

    if np.isscalar(X):
        # Check if the point is inside the bounding box of the contour.
        if X > np.max(contour[:,0]) or X < np.min(contour[:,0]) or Y > np.max(contour[:,1]) or Y < np.min(contour[:,1]):
            return False
        if method=='cv2':
            return __p_in_c_cv2(X, Y, contour)
        elif method=='ray_casting':
            return __p_in_c_ray(X, Y, contour)
        else:
            raise NotImplementedError()
    # --- For arrays
    #assert(X.shape==Y.shape)
    shape_in = X.shape
    Xf = np.array(X).flatten()
    Yf = np.array(Y).flatten()
    Bf = np.zeros(Xf.shape, dtype=bool)
    if len(Xf)!=len(Yf):
        raise Exception('point_in_contour: when array provided, X and Y must have the same length')
    # Quickly eliminate pints outside of bounding box (vectorial calculation)
    bbbox = (Xf <= np.max(contour[:,0])) & (Xf >= np.min(contour[:,0])) & (Yf <= np.max(contour[:,1])) & (Yf >= np.min(contour[:,1]))
    Bf[bbbox] = True
    for i, (x,y,b) in enumerate(zip(Xf, Yf, Bf)):
        if b: # If in Bounding box, go into more complicated calculation
            Bf[i] = __p_in_c_ray(x, y, contour)
    B =  Bf.reshape(shape_in)
    return B


def contour_angles(x, y):
    """
    Computes the angle at each point of a closed contour (in degrees).
    For a closed contour, each point uses its previous and next neighbors.
    Returns an array of angles (degrees).
    For a square, it should return 90, 90, 90, 90, 90
    For points on a line, returns 180
    If the contour is closed (first and last point are the same), ignores the last point for angle calculation.
    """
    x = np.asarray(x)
    y = np.asarray(y)
    n = len(x)
    # Check for closed contour (first and last point repeated)
    is_closed = np.allclose([x[0], y[0]], [x[-1], y[-1]])
    if is_closed:
        x = x[:-1]
        y = y[:-1]
        n = len(x)
    angles = np.zeros(n)
    for i in range(n):
        i_prev = (i - 1) % n
        i_next = (i + 1) % n
        v1 = np.array([x[i_prev] - x[i], y[i_prev] - y[i]])
        v2 = np.array([x[i_next] - x[i], y[i_next] - y[i]])
        v1_norm = v1 / np.linalg.norm(v1)
        v2_norm = v2 / np.linalg.norm(v2)
        dot = np.clip(np.dot(v1_norm, v2_norm), -1.0, 1.0)
        angle_rad = np.arccos(dot)
        angle_deg = np.degrees(angle_rad)
        angles[i] = angle_deg
    if is_closed:
        angles = np.append(angles, angles[0])
    return angles

# --------------------------------------------------------------------------------
# --- Contour Actions
# --------------------------------------------------------------------------------
def contour_remove_duplicates(x, y, reltol=_DEFAULT_REL_TOL, verbose=False):
    l = contour_length_scale(x, y)
    unique_points = []
    duplicate_points = []
    for x,y in zip(x, y):
        if all(np.sqrt((x-p[0])**2 + (y-p[1])**2) > reltol*l for p in unique_points):
            unique_points.append((x,y))
        else:
            duplicate_points.append((x,y))
    x = np.array([p[0] for p in unique_points])
    y = np.array([p[1] for p in unique_points])

    if verbose:
        if len(duplicate_points)>0:
            print('[INFO] curves: {} duplicate(s) removed: {}'.format(len(duplicate_points), duplicate_points))

    return x, y, duplicate_points


def open_contour(x, y, reltol=_DEFAULT_REL_TOL, verbose=False):
    """ Open contour, removing last point if it is the same as the first one, unless it's already open."""
    if contour_is_closed(x, y, reltol=reltol):
        if verbose:
            print('[INFO] curves: Contour closed, removing last point to open it.')
        x = x[:-1]
        y = y[:-1]
    return x, y

def close_contour(x, y, reltol=_DEFAULT_REL_TOL, force=False, verbose=False):
    """ Close contour, unless it's already closed, always do it if force is True"""
    x = np.asarray(x)
    y = np.asarray(y)
    isClosed = contour_is_closed(x, y, reltol=reltol)
    if isClosed or force:
        x = np.append(x, x[0])
        y = np.append(y, y[0])
        if verbose:
            print('[INFO] curves: Contour was open, closing it.')
    return x, y
    
def reloop_contour(x, y, i, verbose=False):
    """
    Reloop a contour array so that it starts at a specific index.
    NOTE: duplicates should preferably be removed
    INPUTS:
    - contour_array: Numpy array of shape (n, 2) representing the contour of point coordinates.
    - i: Index where the relooped contour should start.
    OUTPUTS:
    - Relooped contour array.
    """
    #relooped_contour = np.concatenate((contour_array[i:], contour_array[:i]))
    if i!=0:
        if verbose:
            print("[INFO] curves: Relooping contour to start from index {}".format(i))
        x = np.concatenate((x[i:], x[:i]))
        y = np.concatenate((y[i:], y[:i]))
    return x, y

def opposite_contour(x, y, reltol = _DEFAULT_REL_TOL):
    """
    Make a clockwise contour counterclockwise and vice versa
    INPUTS:
    - contour_array: Numpy array of shape (n, 2) representing the contour of point coordinates.
    OUTPUTS:
    - opposite contour
    """
    isClosed = contour_is_closed(x, y, reltol=reltol)
    if not isClosed:
        # we close the contour
        x, y = close_contour(x, y, force=True, reltol=reltol)
    xopp=x[-1::-1]
    yopp=y[-1::-1]
    # If it was not closed, we remove duplicates
    if not isClosed:
        xopp, yopp, dupli = contour_remove_duplicates(xopp, yopp, reltol=reltol)
    return xopp, yopp

def counterclockwise_contour(x, y, verbose=False):
    if contour_orientation(x, y) != 'counterclockwise':
        x = x[::-1]
        y = y[::-1]
        if verbose:
            print("[INFO] curves: Making contour counterclockwise.")
        return x, y
    else:
        return x, y

def clockwise_contour(x, y, verbose=False):
    if contour_orientation(x, y) != 'clockwise':
        x = x[::-1]
        y = y[::-1]
        if verbose:
            print("[INFO] curves: Making contour clockwise.")
        return x, y
    else:
        return x, y

def curve_enforce_superset(x_orig, y_orig, x_new, y_new, verbose=False, raiseError=False, reltol=_DEFAULT_REL_TOL):
    """
    Ensure that all original points (x_orig, y_orig) are present in the refined grid (x_new, y_new).
    If a point is missing (not within tolerance), replace the closest refined point with the original.
    Returns corrected x_new, y_new arrays.
    """
    ds = np.sqrt(np.diff(x_orig)**2 + np.diff(y_orig)**2)
    tol = np.min(ds) * reltol if len(ds) > 0 else 1e-12
    n_fixed = 0
    for i, (xo, yo) in enumerate(zip(x_orig, y_orig)):
        dists = np.sqrt((x_new - xo)**2 + (y_new - yo)**2)
        min_idx = np.argmin(dists)
        if dists[min_idx] > tol:
            if verbose:
                print(f"[WARN] curves: Point {i} (x={xo:.5f}, y={yo:.5f}) not found in refined grid (min dist={dists[min_idx]:.2e}). Replacing closest point.")
            x_new[min_idx] = xo
            y_new[min_idx] = yo
            n_fixed += 1
    if verbose:
        if n_fixed>0:
            if verbose:
                print(f"[WARN] curves: Total points replaced: {n_fixed} / {len(x_orig)}")
            if raiseError:
                raise ValueError(f"Some original points were not found in the refined grid. {n_fixed} points replaced.")
    return x_new, y_new


def curve_check_superset(x_orig, y_orig, x_new, y_new, reltol=_DEFAULT_REL_TOL, verbose=False, raiseError=False):
    """
    Check whether all points (x_orig, y_orig) are present in (x_new, y_new) within a given tolerance.
    Returns True if all original points are found in the new curve, False otherwise.
    """
    x_orig = np.asarray(x_orig)
    y_orig = np.asarray(y_orig)
    x_new = np.asarray(x_new)
    y_new = np.asarray(y_new)

    ds = np.sqrt(np.diff(x_orig)**2 + np.diff(y_orig)**2)
    tol = np.min(ds) * reltol if len(ds) > 0 else 1e-12

    all_found = True
    for i, (xo, yo) in enumerate(zip(x_orig, y_orig)):
        dists = np.sqrt((x_new - xo)**2 + (y_new - yo)**2)
        min_dist = np.min(dists)
        if min_dist > tol:
            all_found = False
            if verbose:
                print(f"[FAIL] curves: Point {i} (x={xo:.5f}, y={yo:.5f}) not found in new curve (min dist={min_dist:.2e})")
    if not all_found:
        if verbose:
            print("[FAIL] curves: Some original points were not found in the new curve.")
        if raiseError:
            raise ValueError("Some original points were not found in the new curve.")
    return all_found   

# --------------------------------------------------------------------------------}
# --- Contours normals
# --------------------------------------------------------------------------------{
def compute_normals_looped(coords):
    input_is_looped = np.allclose(coords[0], coords[-1])
    if input_is_looped:
        open_loop = coords[:-1]  # Exclude the last point to avoid duplication
    else:
        open_loop = coords

    prev = np.roll(open_loop, 1, axis=0)  # Shifted back by 1
    curr = np.roll(open_loop, 0, axis=0)  
    next = np.roll(open_loop, -1, axis=0)  # Shifted forward by 1

    # Compute tangents at midpoints
    tangents_mid = next - curr
    tangents_mid /= np.linalg.norm(tangents_mid, axis=1)[:, np.newaxis]  # Normalize tangents

    # Rotate tangents by -90 degrees to get outward normals at midpoints
    normals_mid = np.zeros_like(tangents_mid)
    normals_mid[:, 0] = tangents_mid[:, 1]
    normals_mid[:, 1] = -tangents_mid[:, 0]

    # Initialize normals at each point
    normals = np.zeros_like(coords)
    normals[0] = 0.5 * (normals_mid[0] + normals_mid[-1])
    if input_is_looped:
        # Average normals between midpoints
        normals[1:-1] = 0.5 * (normals_mid[:-1] + normals_mid[1:])
        normals[-1] = normals[0]
    else:
        normals[1:] = 0.5 * (normals_mid[:-1] + normals_mid[1:])

    epsilon = 1e-10  # Small value to prevent division by zero
    normals /= (np.linalg.norm(normals, axis=1)[:, np.newaxis] + epsilon)

    return normals, normals_mid

def compute_normals(coords, is_loop=False):
    """
    Compute the normals at each point of the airfoil and the midpoints.
    Normals are averaged between midpoints to ensure smooth transitions.
    """
    if is_loop:
        return compute_normals_looped(coords)
    
    # --- Normals for non-looped
    # Compute tangents at midpoints
    tangents_mid = coords[1:] - coords[:-1]
    tangents_mid /= np.linalg.norm(tangents_mid, axis=1)[:, np.newaxis]  # Normalize tangents

    # Rotate tangents by -90 degrees to get outward normals at midpoints
    normals_mid = np.zeros_like(tangents_mid)
    normals_mid[:, 0] = tangents_mid[:, 1]
    normals_mid[:, 1] = -tangents_mid[:, 0]

    # Initialize normals at each point
    normals = np.zeros_like(coords)

    # Average normals between midpoints
    normals[1:-1] = 0.5 * (normals_mid[:-1] + normals_mid[1:])

    # Handle the first and last points
    normals[0] = normals_mid[0]  # First point normal
    normals[-1] = normals_mid[-1]  # First point normal

    # Normalize the averaged normals
    epsilon = 1e-10  # Small value to prevent division by zero
    normals /= (np.linalg.norm(normals, axis=1)[:, np.newaxis] + epsilon)

    return normals, normals_mid

def smooth_normals(normals, iterations=5, alpha=0.5, boundary_weight=0.8, is_loop=True):
    """
    Smooth normals using a weighted average scheme.
    At the boundaries (trailing edge), apply a forward and backward scheme
    to lean the normals towards the first normal.
    Args:
        normals (np.ndarray): Array of normals to smooth.
        iterations (int): Number of smoothing iterations.
        alpha (float): Smoothing factor (0 < alpha < 1).
        boundary_weight (float): Weight for the boundary normals (0 < boundary_weight <= 1).
    Returns:
        np.ndarray: Smoothed normals.
    """
    if not is_loop:
        raise ValueError("Smoothing is only implemented for looped normals.")
    smoothed = normals.copy()
    for _ in range(iterations):
        prev = np.roll(smoothed, -1, axis=0)  # Shifted back by 1
        next = np.roll(smoothed, 1, axis=0)  # Shifted back by 1
        smoothed = alpha * (prev+next) + (1 - 2 * alpha) * smoothed
    #    # Smooth interior points
    #    smoothed[1:-1] = alpha * (smoothed[:-2] + smoothed[2:]) + (1 - 2 * alpha) * smoothed[1:-1]
    #    # Forward scheme at the first point (trailing edge)
    #    #smoothed[0] = boundary_weight * smoothed[0] + (1 - boundary_weight) * smoothed[1]
    #    smoothed[0] = boundary_weight * smoothed[0] + (1 - boundary_weight) * smoothed[1]
    #    # Backward scheme at the last point (trailing edge)
    #    smoothed[-1] = boundary_weight * smoothed[-1] + (1 - boundary_weight) * smoothed[-2]
    #    # Ensure the first and last normals remain consistent
    #    smoothed[0] = smoothed[-1] = 0.5 * (smoothed[0] + smoothed[-1])
    return smoothed



# --------------------------------------------------------------------------------}
# --- Streamlines and quiver 
# --------------------------------------------------------------------------------{
def lines_to_arrows(lines, n=5, offset=None, spacing=None, normalize=True):
    """ Extract "streamlines" arrows from a set of lines 
    Either: `n` arrows per line
        or an arrow every `spacing` distance
    If `normalize` is true, the arrows have a unit length
    """
    if spacing is None:
        # if n is provided we estimate the spacing based on each curve lenght)
        if type(n) is int:
            n=[n]*len(lines)
        n=np.asarray(n)
        spacing = [ curve_coord(line=l)[-1]/nn for l,nn in zip(lines,n)]
    try:
        len(spacing)
    except:
        spacing=[spacing]*len(lines)
    if offset is None:
        lines_s=[curve_extract(l, spacing=sp, offset=sp/2)         for l,sp in zip(lines,spacing)]
        lines_e=[curve_extract(l, spacing=sp, offset=sp/2+0.01*sp) for l,sp in zip(lines,spacing)]
    else:
        lines_s=[curve_extract(l, spacing=sp, offset=offset)         for l,sp in zip(lines,spacing)]
        lines_e=[curve_extract(l, spacing=sp, offset=offset+0.01*sp) for l,sp in zip(lines,spacing)]
    arrow_x  = [l[i,0] for l in lines_s for i in range(len(l))]
    arrow_y  = [l[i,1] for l in lines_s for i in range(len(l))]
    arrow_dx = [le[i,0]-ls[i,0] for ls,le in zip(lines_s,lines_e) for i in range(len(ls))]
    arrow_dy = [le[i,1]-ls[i,1] for ls,le in zip(lines_s,lines_e) for i in range(len(ls))]

    if normalize:
        dn = [ np.sqrt(ddx**2 + ddy**2) for ddx,ddy in zip(arrow_dx,arrow_dy)]
        arrow_dx = [ddx/ddn for ddx,ddn in zip(arrow_dx,dn)] 
        arrow_dy = [ddy/ddn for ddy,ddn in zip(arrow_dy,dn)] 
    return  arrow_x,arrow_y,arrow_dx,arrow_dy 


def seg_to_lines(seg):
    """ Convert list of segments to list of continuous lines"""
    def extract_continuous(i):
        x=[]
        y=[]
        # Special case, we have only 1 segment remaining:
        if i==len(seg)-1:
            x.append(seg[i][0,0])
            y.append(seg[i][0,1])
            x.append(seg[i][1,0])
            y.append(seg[i][1,1])
            return i,x,y
        # Looping on continuous segment
        while i<len(seg)-1:
            # Adding our start point
            x.append(seg[i][0,0])
            y.append(seg[i][0,1])
            # Checking whether next segment continues our line
            #print('cur start',seg[i][0,:]  , ' end ',seg[i][1,:])
            Continuous= all(seg[i][1,:]==seg[i+1][0,:])
            #print('nxt start',seg[i+1][0,:], ' end ',seg[i+1][1,:],'Conti:',Continuous)
            if not Continuous:
                # We add our end point then
                x.append(seg[i][1,0])
                y.append(seg[i][1,1])
                break
            elif i==len(seg)-2:
                # we add the last segment
                x.append(seg[i+1][0,0])
                y.append(seg[i+1][0,1])
                x.append(seg[i+1][1,0])
                y.append(seg[i+1][1,1])
            i=i+1
        return i,x,y
    lines=[]
    i=0
    while i<len(seg):
        iEnd,x,y=extract_continuous(i)
        lines.append(np.array( [x,y] ).T)
        i=iEnd+1
    return lines

def streamQuiver(ax, sp, n=5, spacing=None, offset=None, normalize=True, **kwargs):
    """ Plot arrows from streamplot data  
    The number of arrows per streamline is controlled either by `spacing` or by `n`.
    See `lines_to_arrows`.
    """
    # --- Main body of streamQuiver
    # Extracting lines
    seg   = sp.lines.get_segments() 
    if seg[0].shape==(2,2):
        #--- Legacy)
        # list of (2, 2) numpy arrays
        # lc = mc.LineCollection(seg,color='k',linewidth=0.7)
        lines = seg_to_lines(seg) # list of (N,2) numpy arrays
    else:
        lines = seg

    # Convert lines to arrows
    ar_x, ar_y, ar_dx, ar_dy = lines_to_arrows(lines, offset=offset, spacing=spacing, n=n, normalize=normalize)
    # Plot arrows
    qv=ax.quiver(ar_x, ar_y, ar_dx, ar_dy, **kwargs)
    return qv


# --------------------------------------------------------------------------------}
# --- TEST 
# --------------------------------------------------------------------------------{
class TestCurves(unittest.TestCase):
    def test_seg_to_lines(self):
        # --- Useful variables
        Seg01=np.array([[0,0],[1,1]])
        Seg12=np.array([[1,1],[2,2]])
        Seg21=np.array([[2,2],[1,1]])
        Seg02=np.array([[0,0],[2,2]])

        # --- Empty segment > empty line
        lines= seg_to_lines([])
        self.assertEqual(seg_to_lines([]),[])
        # --- One segment >  one line
        lines= seg_to_lines([Seg01])
        np.testing.assert_equal(lines,[Seg01])
        # --- One continuous line
        lines= seg_to_lines([Seg01,Seg12,Seg21])
        np.testing.assert_equal(lines[0],np.array([ [0,1,2,1],[0,1,2,1]]).T)
        # --- One segment and one continuous lines
        lines= seg_to_lines([Seg02,Seg01,Seg12])
        np.testing.assert_equal(lines[0],Seg02)
        np.testing.assert_equal(lines[1],np.array([ [0,1,2],[0,1,2]]).T)
        # --- One continuous lines, one segment
        lines= seg_to_lines([Seg01,Seg12,Seg02])
        np.testing.assert_equal(lines[1],Seg02)
        np.testing.assert_equal(lines[0],np.array([ [0,1,2],[0,1,2]]).T)
        # --- Two continuous lines
        lines= seg_to_lines([Seg01,Seg12,Seg02,Seg21])
        np.testing.assert_equal(lines[0],np.array([ [0,1,2],[0,1,2]]).T)
        np.testing.assert_equal(lines[1],np.array([ [0,2,1],[0,2,1]]).T)

    def test_curv_interp(self):
        # --- Emtpy or size 1
#         np.testing.assert_equal(curve_interp([],[],5),([],[]))
#         np.testing.assert_equal(curve_interp([1],[0],5),([1],[0]))
#         np.testing.assert_equal(curve_interp([1],[0],5),([1],[0]))
        # --- Interp along x
        x=[0,1,2]
        y=[0,0,0]
        xx,yy=curve_interp(x,y,5)
        np.testing.assert_equal(xx,[0,0.5,1,1.5,2])
        np.testing.assert_equal(yy,xx*0)
        # --- Interp diag
        x=[0,1]
        y=[0,1]
        xx,yy=curve_interp(x,y,3)
        np.testing.assert_equal(xx,[0,0.5,1])
        np.testing.assert_equal(yy,[0,0.5,1])
        # --- Interp same size
        xx,yy=curve_interp(x,y,2)
        np.testing.assert_equal(xx,x)
        np.testing.assert_equal(yy,y)
        # --- Interp lower size
        xx,yy=curve_interp(x,y,1)
        np.testing.assert_equal(xx,x[0])
        np.testing.assert_equal(yy,y[0])
    
    def test_contour_querries(self):
        # diamond counter clockwise
        x = np.array([1, 0.5, 0, 0.5, 1])
        y = np.array([0, 1, 0, -1, 0])
        # --- Closed
        self.assertTrue(contour_is_closed(x, y))    
        # --- Counter Clockwise
        self.assertEqual(contour_orientation(x, y), "counterclockwise")
        self.assertTrue(contour_is_counterclockwise(x, y))
        self.assertFalse(contour_is_clockwise(x, y))

    def test_contour_manip(self):
        # diamond counter clockwise
        x = np.array([1, 0.5, 0, 0.5, 1])
        y = np.array([0, 1, 0, -1, 0])

        x1, y1 = clockwise_contour(x, y)
        self.assertEqual(contour_orientation(x1, y1), "clockwise")
        x2, y2 = counterclockwise_contour(x1, y1)
        self.assertEqual(contour_orientation(x2, y2), "counterclockwise")
        np.testing.assert_array_equal(x, x2)
        np.testing.assert_array_equal(y, y2)


    def test_contour_angle_square_closed(self):
        # Square, 5 points (closed)
        x = np.array([0, 1, 1, 0, 0])
        y = np.array([0, 0, 1, 1, 0])
        angles = contour_angles(x, y)
        expected = np.full(5, 90.0)
        np.testing.assert_allclose(angles, expected, atol=1e-6)
        # same test for a non-closed contour
        x = x[:-1]
        y = y[:-1]
        angles = contour_angles(x, y)
        expected = expected[:-1]  # Last angle is not defined for non-closed contour
        np.testing.assert_allclose(angles, expected, atol=1e-6) 

    def test_contour_angle_square_with_midpoints(self):
        # Square with midpoints on each edge, 9 points (closed)
        x = np.array([0, 0.5, 1, 1, 1, 0.5, 0, 0, 0])
        y = np.array([0, 0, 0, 0.5, 1, 1, 1, 0.5, 0])
        angles = contour_angles(x, y)
        expected = np.array([90, 180, 90, 180, 90, 180, 90, 180, 90])
        np.testing.assert_allclose(angles, expected, atol=1e-6)
        # same test for a non-closed contour
        x = x[:-1]
        y = y[:-1]
        angles = contour_angles(x, y)
        expected = expected[:-1]  # Last angle is not defined for non-closed contour
        np.testing.assert_allclose(angles, expected, atol=1e-6) 

    def test_contour_angle_triangle(self):
        # Equilateral triangle, 4 points (closed)
        x = np.array([0, 1, 0.5, 0])
        y = np.array([0, 0, np.sqrt(3)/2, 0])
        angles = contour_angles(x, y)
        expected = np.array([60, 60, 60, 60])
        np.testing.assert_allclose(angles, expected, atol=1e-6)
        # same test for a non-closed contour
        x = x[:-1]
        y = y[:-1]
        angles = contour_angles(x, y)
        expected = expected[:-1]  # Last angle is not defined for non-closed contour
        np.testing.assert_allclose(angles, expected, atol=1e-6) 

    def test_contour_angle_line(self):
        # Straight line, 3 points (closed)
        x = np.array([0, 1, 2, 0])
        y = np.array([0, 0, 0, 0])
        angles = contour_angles(x, y)
        expected = np.array([0, 180, 0, 0])
        np.testing.assert_allclose(angles, expected, atol=1e-6)
        # same test for a non-closed contour
        x = x[:-1]
        y = y[:-1]
        angles = contour_angles(x, y)
        expected = expected[:-1]  # Last angle is not defined for non-closed contour
        np.testing.assert_allclose(angles, expected, atol=1e-6) 

    def test_contour_angle_pentagon(self):
        # Regular pentagon, 6 points (closed)
        theta = np.linspace(0, 2*np.pi, 6)
        x = np.cos(theta)
        y = np.sin(theta)
        angles = contour_angles(x, y)
        expected_angle = 108.0  # Internal angle of regular pentagon
        expected = np.full(len(x), expected_angle)
        np.testing.assert_allclose(angles, expected, atol=1e-6)
        # same test for a non-closed contour
        x = x[:-1]
        y = y[:-1]
        angles = contour_angles(x, y)
        expected = expected[:-1]  # Last angle is not defined for non-closed contour
        np.testing.assert_allclose(angles, expected, atol=1e-6) 



# --------------------------------------------------------------------------------}
# --- Examples 
# --------------------------------------------------------------------------------{
def example_streamquiver():
    import numpy as np
    import matplotlib.pyplot as plt

    x = np.linspace(-1,1,100)
    y = np.linspace(-1,1,100)
    X,Y=np.meshgrid(x,y)
    u = -np.sin(np.arctan2(Y,X))
    v =  np.cos(np.arctan2(Y,X))

    xseed=np.linspace(0.1,1,4)

    fig=plt.figure()
    ax=fig.add_subplot(111)
    sp = ax.streamplot(x,y,u,v,color='k',arrowstyle='-',start_points=np.array([xseed,xseed*0]).T,density=30)
    qv = streamQuiver(ax,sp,spacing=0.5, scale=50)
    plt.axis('equal')
    plt.show()





if __name__ == "__main__":
    #TestCurves().test_curv_interp()
    unittest.main()


