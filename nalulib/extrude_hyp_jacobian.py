import os
import matplotlib.pyplot as plt
import numpy as np
import meshio
from airfoillib import load_airfoil_coords, detect_airfoil_te_type, reindex_starting_from_te, detect_airfoil_features, compute_normals, smooth_normals
from meshlib import create_quadrilateral_cells, save_mesh, open_mesh_in_gmsh
from extrudelib import front_extrude_base, front_extrude_TMHW, front_extrude_gem
from welib.essentials import * # NOTE: Do Not Remove
from quads import *
from scipy.optimize import least_squares

# ---------------------------------------------------------------------------
# --- Meshing library
# ---------------------------------------------------------------------------
def compute_chord_lengths(front):
    """
    Compute chord lengths for a closed loop using centered finite differences.
    
    Args:
        front (np.ndarray): Array of points representing the front.
    Returns:
        np.ndarray: Array of chord lengths.
    """
    is_loop = np.allclose(front[0], front[-1])
    if is_loop:
        front = front.copy()
        front = front[:-1]

    front_prev = np.roll(front, 1, axis=0)  # Shifted back by 1
    front_next = np.roll(front, -1, axis=0)  # Shifted forward by 1
    chord_lengths = 0.5* (np.linalg.norm(front_next - front, axis=1) + (np.linalg.norm(front - front_prev, axis=1)))
    return chord_lengths


def compute_Zparams(r_ref, normals, volSmooth=0.3, dy=0.1):
    p = dict()
    p['r_ref'] = r_ref
    p['dy'] = dy
    p['ds'] = dy/6
    p['normals'] = normals
    p['tangents'] = np.zeros_like(normals)
    p['tangents'][:,0] = normals[:, 1]
    p['tangents'][:,1] = -normals[:, 0]
    printMat('Tangents', p['tangents'], digits=2, nchar=8)
    p['c_cur'] = compute_chord_lengths(r_ref)
    p['C_cur'] = np.sum(p['c_cur']) 
    p['c_bar_cur'] = p['C_cur'] / (len(r_ref)-1)
    # l_desired = (C_new / C_cur) * ((1 - volSmooth) * c_cur + volSmooth * c_bar_cur)
    z_new = r_ref.flatten() + normals.flatten() * dy  # Initial guess for z_new
    c_new = compute_chord_lengths(z_new.reshape((-1, 2)))
    C_new = np.sum(c_new)
    p['C_new0'] = C_new
    p['kl_desired'] = 1/p['C_cur'] * (  (1 - volSmooth)* p['c_cur'] + volSmooth *  p['c_bar_cur'] )
    p['l_desired'] = C_new/p['C_cur'] * (  (1 - volSmooth)* p['c_cur'] + volSmooth *  p['c_bar_cur'] )
    # --- Quad
    p['p0'] = r_ref
    p['p1'] = np.roll(r_ref, 1, axis=0) 
    p['base_length'], p['base_unit'] , p['eta_unit'] = calculate_quad_base_params(p['p0'], p['p1'])
    return p


def Zeq(z_new, params, debug=False, fig=None, iteration=[0]):
    """
    Args:
        z_new (np.ndarray): Flattened array of new front points (x1, y1, x2, y2, ..., xn, yn).
        z_ref (np.ndarray): Flattened array of reference front points (x1, y1, x2, y2, ..., xn, yn).
        l_desired (np.ndarray): Desired chord lengths for the new front.

    Returns:
        np.ndarray: Constraint vector of length 2*n.
    
     p3  ---- p2   <- r_new (new front)
     |        |
     p0  ---- p1  <- r_ref (old front)
    """
    n = len(z_new) // 2  # Number of points
    r_new = z_new.reshape((n, 2))



    # Compute tangents for the reference front
    r_ref    = params['r_ref']
    tangents = params['tangents']

    # Compute scalar product constraint
    dr = r_new - r_ref
    scalar_product = dr[:,0] * tangents[:,0] + dr[:,1] * tangents[:,1]

    # Compute quad area constraint
    #p_mid = (p0 + p1)/2
    p2 = np.roll(r_new, 1, axis=0)  # Shifted back by 1
    p3 = r_new
    #area_actual = smoothed_area(params['p0'], params['p1'], p2, p3, base_params=params)
    area_actual = penalized_area(params['p0'], params['p1'], p2, p3, base_params=params)
    #area_actual = quad_area(params['p0'], params['p1'], p2, p3)
    #printMat('p0', params['p0'], digits=2, nchar=8)
    #printMat('p1', params['p1'], digits=2, nchar=8)
    #printMat('p2',p2, digits=2, nchar=8)
    #printMat('p3',p3, digits=2, nchar=8)
    #printMat('Area', area_actual)


    # Compute chord lengths constraint
    c_new = compute_chord_lengths(r_new)
    C_new = np.sum(c_new)
    c_bar_new = C_new / (len(r_new)-1)
    l_desired = C_new * params['kl_desired']
    length_constraint =  params['l_desired'] - c_new
    #length_constraint = c_new - c_bar_new
    #area_desired = params['dy'] * params['l_desired']  # NOTE this is approximate
    area_desired = params['dy'] * l_desired
    #area_desired = params['dy'] * params['c_bar_cur'] 
    area_constraint = area_desired - area_actual

    # Compute growth constraint
    growth_constraint = np.einsum('ij,ij->i', dr, params['normals'])  # Dot product with normals

    # Combine constraints into a single vector
    Z_vector = np.empty(3 * n)
    Z_vector[0:n] = scalar_product  # Scalar product constraint
    Z_vector[n:2*n] = area_constraint  # Area constraint
    Z_vector[2*n:] = length_constraint  # Length constraint
    #Z_vector[2*n:] = growth_constraint  # Growth constraint

    if debug:
        print('')
        print('l_des', l_desired)
        print('l_act', c_new)
        print('dl  ', l_desired-c_new)
        print('a_des', area_desired)
        print('a_act', area_actual)
        print('da  ', area_desired-area_actual)

    res = np.linalg.norm(Z_vector)
    if fig is not None and res<0.1:
        new_front = z_new.reshape((-1, 2))  # Reshape to 2D points
        new_front2 = np.vstack([new_front, new_front[0]])  # Close the loop for plotting
        fig.gca().plot(new_front2[:, 0], new_front2[:, 1], '-o', label=f'{iteration[0]}')
        iteration[0] += 1

    return Z_vector


def compute_jacobian(z0, params):
    """
    Compute the Jacobian of the constraint function Z with respect to z0 using central differences.

    Args:
        z0 (np.ndarray): Flattened array of new front points (x1, y1, x2, y2, ..., xn, yn).
        params (dict): Parameters required for the constraint function Z.

    Returns:
        np.ndarray: Jacobian matrix of shape (2*n, 2*n).
    """
    n = len(z0)
    Z0 = Zeq(z0, params)  # Evaluate Z at the current z0
    jacobian = np.zeros((2 * (n // 2), n))  # Jacobian matrix
    epsilon = params['ds']  # Perturbation size

    for i in range(n):
        # Perturb the i-th component of z0 in both positive and negative directions
        z_perturbed_plus = z0.copy()
        z_perturbed_minus = z0.copy()
        z_perturbed_plus[i] += epsilon
        z_perturbed_minus[i] -= epsilon

        # Compute Z at the perturbed points
        Z_plus = Zeq(z_perturbed_plus, params)
        Z_minus = Zeq(z_perturbed_minus, params)

        # Compute the central difference
        jacobian[:, i] = (Z_plus - Z_minus) / (2 * epsilon)

    return jacobian


def solve_extrusion(r_ref, normals, dy, volSmooth=0.3, max_iterations=10, tol=1e-6, debug=False, fig=None):
    """
    Solve for the new front points using least-squares minimization.

    Args:
        r_ref (np.ndarray): Reference front points (n x 2).
        normals (np.ndarray): Normals at each point of the reference front (n x 2).
        dy (float): Initial extrusion distance along the normals.
        volSmooth (float): Volume smoothing factor.
        max_iterations (int): Maximum number of iterations for convergence.
        tol (float): Convergence tolerance for the constraints.
        debug (bool): If True, print debug information.
        fig (matplotlib.figure.Figure): Optional figure for plotting.

    Returns:
        np.ndarray: New front points (n x 2).
    """
    params = compute_Zparams(r_ref, normals, volSmooth=volSmooth, dy=dy)

    z_ref = r_ref.flatten()  # Flatten the reference front points
    z_new = z_ref + normals.flatten() * dy  # Initial guess for z_new

    def objective_function(z):
        """
        Objective function for least-squares minimization.
        Args:
            z (np.ndarray): Flattened array of new front points.
        Returns:
            np.ndarray: Constraint vector to minimize.
        """
        return Zeq(z, params, debug=debug, fig=fig)

    # Use least-squares minimization to solve for z_new
    result = least_squares(
        objective_function,
        z_new,
        method='lm',  # Levenberg-Marquardt method
        max_nfev=max_iterations,
        xtol=tol,
        verbose=2 if debug else 0
    )

    # Reshape the result back to 2D
    r_new = result.x.reshape((-1, 2))

    if debug:
        print("Optimization Result:")
        print(result)

    return r_new


def front_extrude_manu(front, dy, kbSmooth=3, volSmooth=0.3, max_iterations=10, tol=1e-6, debug=False, fig=None):
    """
    Extrude a 2D line (front) to generate a new front for mesh generation,
    ensuring progressive convergence toward uniform chord lengths and cell areas.
    
    Args:
        front (np.ndarray): Current front (layer) of points.
        dy (float): Layer spacing for the extrusion.
        kbSmooth (int): Number of Kinsey-Barth smoothing iterations (default: 3).
        volSmooth (float): Volume smoothing factor (0 to 1, default: 0.3).
        max_iterations (int): Maximum number of iterations for convergence.
        tol (float): Convergence tolerance for chord lengths and areas.
        debug (bool): If True, print debug information.
    Returns:
        np.ndarray: New extruded front.
    """
    # Compute initial normals for the current front
    normals, normal_mid = compute_normals(front)

    # Remove the last point if the front is a closed loop
    is_loop = np.allclose(front[0], front[-1])
    if is_loop:
        front = front.copy()
        front = front[:-1]
        normals = normals[:-1]

    new_front =  solve_extrusion(front, normals, dy=dy, volSmooth=volSmooth, max_iterations=max_iterations, debug=debug, fig=fig)

    if is_loop:
        new_front = np.vstack([new_front, new_front[0]])

    return new_front

# ---------------------------------------------------------------------------
# --- Main
# ---------------------------------------------------------------------------
def main():
    from curves import curve_interp
    np.set_printoptions(precision=3)
    # Example front (closed loop)
    front = np.array([
        [0.0, 0.0],
        [1.0, 0.0],
        [1.0, 1.0],
        [0.0, 1.0],
        [0.0, 0.0]  # Closed loop
    ])

    #front =  curve_interp(ds=0.1, line=front)
    #print(front)

    R = 2
    theta=np.linspace(0, 2*np.pi, 30)
    theta=np.concatenate([theta[:3], theta[4:]])
    x = np.cos(theta) * R
    y = np.sin(theta) * R
    front = np.vstack([x, y]).T

    max_iteration=6


    fig,ax = plt.subplots(1, 1, sharey=False, figsize=(6.4,4.8))
    fig.subplots_adjust(left=0.12, right=0.95, top=0.95, bottom=0.11, hspace=0.20, wspace=0.20)
    ax.plot(front[:,0], front[:,1], '+-', label='old')

    # Extrude the front
    new_front = front_extrude_manu(front, dy=0.1, volSmooth=0.3, debug=True, max_iterations=max_iteration, fig=fig)
    ax.plot(new_front[:,0], new_front[:,1], 'd-', label='new')

    # Print the new front
    print("New Front:")
    print(new_front)

    ax.set_xlabel('')
    ax.set_ylabel('')
    ax.legend()
    plt.show()
    raise Exception()



    # User inputs
    nBody = 200
    h0 = 0.00000278  # first layer height
    gr_near = 1.04   # growth rate near the body
    gr_away = 1.09   # growth rate away from the body
    hmax_near = 0.4  # max height near the body
    hmax_away = 1   # max height away from the body
    kbSmooth_near = 3     # Kinsey-Barth smoothing near the body
    volSmooth_near = 0.3  # volume smoothing near the body
    kbSmooth_away  = 0     # Kinsey-Barth smoothing away from the body
    volSmooth_away = 1.0  # volume smoothing away from the body

    h0 = 0.01  # first layer height
    gr_near = 1.00   # growth rate near the body
    #hmax_near = 0.1  # max height near the body
    #hmax_away = 1   # max height away from the body
    #kbSmooth_near = 3     # Kinsey-Barth smoothing near the body
    #volSmooth_near = 0.5  # volume smoothing near the body
    #kbSmooth_away  = 0     # Kinsey-Barth smoothing away from the body
    #volSmooth_away = 1.0  # volume smoothing away from the body

    filename_in = 'ffa_w3_211_coords_all.dat'
    filename_out = "airfoil_omesh.msh"
    gmsh = True

    # Load airfoil coordinates
    coords = load_airfoil_coords(filename_in)

    # Detect trailing edge type and reindex airfoil coordinates
    te_type, te_indices = detect_airfoil_te_type(coords)
    coords = reindex_starting_from_te(coords, te_indices)

    nPoints = coords.shape[0]

    # Detect airfoil features
    te_type, indices = detect_airfoil_features(coords)

    # Generate hyperbolic extrusion near the body
    layers_near, dys_near = hyp_extrude(coords, h0, gr_near, hmax_near, kbSmooth=kbSmooth_near, volSmooth=volSmooth_near, method='manu')
    print(f"Near: n={len(layers_near)} - dy={h0}..{dys_near[-1]} -h={np.sum(dys_near)} ({hmax_near}).")
    # Set the first layer height for the second phase
    if False:
        h0_away = dys_near[-1] * gr_near
        # Generate hyperbolic extrusion away from the body
        layers_away, dys_away = hyp_extrude(layers_near[-1], h0_away, gr_away, hmax_away, kbSmooth=kbSmooth_away, volSmooth=volSmooth_away)
        print(f"Away: n={len(layers_away)} - dy={h0_away}..{dys_away[-1]} -h={np.sum(dys_away)} ({hmax_away}).")
        # Combine layers
        layers_near = layers_near[:-1]  # Remove the last layer to avoid duplication
        layers = np.vstack([layers_near, layers_away])
    else:
        layers = layers_near
    print(f"Total layers combined: {layers.shape[0]}")
    #plot_layers(layers_near, layers_away)
    #plt.show()
    #raise Exception()
    # Create quadrilateral cells
    points = layers.reshape(-1, 2)
    cells = create_quadrilateral_cells(layers, layers.shape[0], nPoints)
    # Save mesh
    save_mesh(points, cells, filename=filename_out)
    # View mesh in GMSH
    if gmsh:
        open_mesh_in_gmsh(filename=filename_out)


if __name__ == "__main__":
    main()
    plt.show()