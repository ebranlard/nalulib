import os
import matplotlib.pyplot as plt
import numpy as np
import meshio
from scipy.optimize import least_squares
from nalulib.airfoillib import load_airfoil_coords, detect_airfoil_te_type, reindex_starting_from_te, detect_airfoil_features, compute_normals, smooth_normals
from nalulib.meshlib import create_quadrilateral_cells
from nalylib.gmesh import save_mesh, open_mesh_in_gmsh
from nalulib.extrudelib import front_extrude_base, front_extrude_TMHW, front_extrude_gem, normal_growth, consistent_growth
from nalulib.quads import *
from nalulib.essentials import * 


def hyp_extrude(coords, h0, gr, hmax, kbSmooth=3, volSmooth=0.3, explicit=0.5, debug=False, method='gem', is_loop=None, max_layers=100):
    """
    """
    layers = [coords]  # Start with the airfoil surface as the first layer
    dy = h0
    cumulative_height = 0
    dys = []  # Store delta y values for each layer

    if is_loop is None:
        is_loop = np.allclose(coords[0], coords[-1])
        coords = coords.copy()
        coords = coords[:-1]
    print('>>> is_loop', is_loop)


    iteration = 0
    while cumulative_height < hmax and iteration < max_layers:
        # Extrude a new layer using front_extrude
        old_layer = layers[-1]
        if is_loop:
            old_layer = old_layer[:-1]  # Remove the last point for processing
        if method=='base':
            new_layer = front_extrude_base( old_layer, dy, kbSmooth=kbSmooth, volSmooth=volSmooth, explicit=explicit, debug=debug, is_loop=is_loop, max_layers=max_layers)
        elif method=='TMHW':
            new_layer = front_extrude_base( old_layer, dy, kbSmooth=kbSmooth, volSmooth=volSmooth, explicit=explicit, debug=debug, is_loop=is_loop, max_layers=max_layers)
        elif method=='manu':
            new_layer = front_extrude_manu( old_layer, dy, kbSmooth=kbSmooth, volSmooth=volSmooth, debug=debug, is_loop=is_loop, max_layers=max_layers)
        elif method=='gem':
            new_layer = front_extrude_gem ( old_layer, dy, kbSmooth=kbSmooth, volSmooth=volSmooth, is_loop=is_loop, max_layers=max_layers)

        if is_loop:
            new_layer = np.vstack([new_layer, new_layer[0]])
        dys.append(dy)

        # Check for crossing lines between the previous and new layer
        #if check_for_crossing_lines_between_layers(layers[-1], new_layer, iteration):
        #    print(f"[WARN]: Crossing lines detected at iteration {iteration}.")
        # Check if the new layer is too close to the previous one
        if np.allclose(new_layer, layers[-1], atol=1e-8):
            print('Layers are too close, stopping iteration.')
            raise Exception('Layers are too close, stopping iteration.')
        # Add the new layer to the list
        layers.append(new_layer)
        # Update cumulative height and layer spacing
        cumulative_height += dy
        dy *= gr
        iteration += 1

    # Convert layers to a NumPy array
    layers = np.array(layers)
    dys = np.array(dys)

    return layers, dys
# ---------------------------------------------------------------------------
# --- Meshing library
# ---------------------------------------------------------------------------
def compute_chord_lengths(front, is_loop=False):
    """
    Compute chord lengths for a closed loop using centered finite differences.
    
    Args:
        front (np.ndarray): Array of points representing the front.
    Returns:
        np.ndarray: Array of chord lengths.
    """
    if is_loop:
        front_prev = np.roll(front, 1, axis=0)  # Shifted back by 1
        front_next = np.roll(front, -1, axis=0)  # Shifted forward by 1
        chord_lengths = 0.5* (np.linalg.norm(front_next - front, axis=1) + (np.linalg.norm(front - front_prev, axis=1)))
    else:
        raise NotImplementedError("Only closed loops are supported for chord length calculation.")
    return chord_lengths

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
    return smoothed

def adjust_point_locally(new_layer, old_layer, normals, bad_indices, dy):
    """
    Adjust the position of points in the new_layer at bad_indices
    to improve quad quality.
    This is a simplified placeholder. More sophisticated methods might
    involve localized smoothing, reducing the extrusion distance, or
    projecting the point to a better location.
    """
    # Your implementation here
    # Simple approach: average with neighbors or reduce the step size for these points
    print(f"Adjusting {len(bad_indices)} points due to poor quad quality.")
    for idx in bad_indices:
        # Example: reduce the step size for the problematic point
        # This is a very basic fix; more advanced methods are needed for robustness
        new_layer[idx] = old_layer[idx] + normals[idx] * dy * 0.5 # Reduce extrusion by half
    return new_layer


def front_extrude_manu(front, dy, kbSmooth=3, volSmooth=0.3, max_iterations=10, tol=1e-6, debug=False, fig=None, is_loop=False, max_layers=10):
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
    normals, normal_mid = compute_normals(front, is_loop=is_loop)

    # Remove the last point if the front is a closed loop
    if is_loop and np.allclose(front[0], front[-1]):
        raise ValueError("The front is a closed loop, please remove the last point.")
    print('>>> is_loop', is_loop)

    normals_ori = normals.copy()  # Store original normals for later use

    # Compute initial chord lengths and total length
    c_cur = compute_chord_lengths(front, is_loop=is_loop)
    C_cur = np.sum(c_cur)
    c_bar_cur = C_cur / len(front)

    # Initial guess for the new front
    new_front = front + normals_ori * dy

    for iteration in range(max_iterations):
        if fig is not None:
            new_front2 = np.vstack([new_front, new_front[0]])  # Close the loop for plotting
            fig.gca().plot(new_front2[:, 0], new_front2[:, 1], '-o', label=f'{iteration}')

        # Compute the quads formed by the old and new fronts
        p0 = front
        p1 = np.roll(front, -1, axis=0)
        p2 = np.roll(new_front, -1, axis=0)
        p3 = new_front

        # Compute actual areas and chord lengths
        actual_area = penalized_area(p0, p1, p2, p3)
        c_new = compute_chord_lengths(new_front, is_loop=is_loop)
        C_new = np.sum(c_new)
        c_bar_new = C_new / len(new_front)

        # Compute desired chord lengths and areas
        l_desired = (C_new / C_cur) * ((1 - volSmooth) * c_cur + volSmooth * c_bar_cur)
        desired_area = dy * l_desired

        # Adjust normals and dy based on area and chord length differences
        for i in range(len(new_front)):
            # Compute area and chord differences
            area_diff = desired_area[i] - actual_area[i]
            chord_diff = l_desired[i] - c_new[i]

            # Adjust dy locally based on area difference
            dy_adjustment = np.clip(area_diff / l_desired[i], -0.1 * dy, 0.1 * dy)
            dy_local = dy + dy_adjustment

            # Adjust normals based on neighbor areas
            left_area = actual_area[i - 1] if i > 0 else actual_area[-1]
            right_area = actual_area[i + 1] if i < len(new_front) - 1 else actual_area[0]
            neighbor_influence = (right_area - left_area) / (right_area + left_area + 1e-6)
            normal_adjustment = np.clip(neighbor_influence, -np.radians(20), np.radians(20))
            rotation_matrix = np.array([
                [np.cos(normal_adjustment), -np.sin(normal_adjustment)],
                [np.sin(normal_adjustment), np.cos(normal_adjustment)]
            ])
            normals[i] = rotation_matrix @ normals[i]

            # Update the new front point
            new_front[i] = front[i] + normals[i] * dy_local

        # Check for crossings using penalized_area
        if np.any(penalized_area(p0, p1, np.roll(new_front, -1, axis=0), new_front) == 0):
            if debug:
                print(f"Crossing detected at iteration {iteration}. Rolling back changes.")
            new_front = front + normals_ori * dy
            break

        # Check for convergence
        chord_converged = np.allclose(c_new, l_desired, atol=tol)
        area_converged = np.allclose(actual_area, desired_area, atol=tol)

        if chord_converged and area_converged:
            if debug:
                print(f"Converged in {iteration + 1} iterations.")
            break
    else:
        if debug:
            print("Warning: Maximum iterations reached without convergence.")

    return new_front

def front_extrude_gem(front, dy, kbSmooth=3, volSmooth=0.3, implicit_or_explicit='implicit', debug=False, is_loop=False, max_layers=2):
    """
    Extrude a 2D line (front) to generate a new front for mesh generation,
    incorporating hyperbolic PDE concepts. This version uses vectorized
    normal calculations for improved efficiency and handles open and closed fronts.
    
    Args:
        front (np.ndarray): Current front (layer) of points.
        dy (float): Layer spacing for the extrusion.
        kbSmooth (int): Number of Kinsey-Barth smoothing iterations (default: 3).
        volSmooth (float): Volume smoothing factor (0 to 1, default: 0.3).
        implicit_or_explicit (str): Method for node placement ('implicit' or 'explicit').
        debug (bool): If True, print debug information.
    Returns:
        np.ndarray: New extruded layer.
    """
    # Determine if the front is a closed loop
    if is_loop and np.allclose(front[0], front[-1]):
        raise ValueError("The front is a closed loop, please remove the last point.")

    # Compute normals for the current front
    normals, _ = compute_normals(front, is_loop=is_loop)

    # --- Compute chord of current layer
    p0 = front
    p1 = np.roll(front, -1, axis=0)
    c_cur = compute_chord_lengths(front, is_loop=is_loop)
    C_cur = np.sum(c_cur)
    c_bar_cur = C_cur / len(front)

    if debug:
        print(f"Is loop: {is_loop}, kbSmooth: {kbSmooth}")

    # Step 1: Apply Kinsey-Barth smoothing to prevent grid line crossing
    effective_dy = np.full(front.shape[0], dy)  # Initialize with dy
    if kbSmooth >= 3:
        if is_loop:
            # Calculate curvature (approximation) - vectorized
            front_prev = np.roll(front, 1, axis=0)  # Shifted back by 1
            front_next = np.roll(front, -1, axis=0)  # Shifted forward by 1
            prev_vectors = front_prev - front
            next_vectors = front_next - front
            prev_angles = np.arctan2(prev_vectors[:, 1], prev_vectors[:, 0])
            next_angles = np.arctan2(next_vectors[:, 1], next_vectors[:, 0])
            # Handle angle wraparound (important for loops)
            d_angles = next_angles - prev_angles
            d_angles = (d_angles + np.pi) % (2 * np.pi) - np.pi
            curvature = d_angles
        else:
            raise NotImplementedError("Only closed loops are supported for curvature calculation.")
            #curvature = np.zeros(num_points)
            #prev_vectors = front_prev - front
            #next_vectors = front_next - front
            #for i in range(1, num_points - 1):
            #    prev_angle = np.arctan2(prev_vectors[i, 1], prev_vectors[i, 0])
            #    next_angle = np.arctan2(next_vectors[i, 1], next_vectors[i, 0])
            #    curvature[i] = (next_angle - prev_angle)

        kb_factor = 1.0 / (1 + kbSmooth * np.abs(curvature) * dy)
        effective_dy[:] = dy * kb_factor
        if any(effective_dy < 0):
            print("[WARN] Number of negative effective_dy values:", np.sum(effective_dy < 0))
            effective_dy = np.abs(effective_dy)  # Ensure non-negative

    if 0 < volSmooth < 1:
        max_iteration=50
        for it in range(max_iteration):
            normals= smooth_normals(normals, iterations=2, alpha=0.5, boundary_weight=0.8, is_loop=is_loop)

            growth_vec= normal_growth(normals, effective_dy)
            new_layer = consistent_growth(front, growth_vec, normals)

            # --- Compute metrics for new layer
            p2 = np.roll(new_layer, -1, axis=0)
            p3 = new_layer
            actual_area = penalized_area(p0, p1, p2, p3)
            c_new = compute_chord_lengths(new_layer, is_loop=is_loop)
            C_new = np.sum(c_new)
            c_bar_new = C_new / len(front)

            l_desired = (C_new / C_cur) * ((1 - volSmooth) * c_cur + volSmooth * c_bar_cur)
            area_desired = effective_dy * l_desired
            l_mean_error = np.mean(np.abs(l_desired-c_new))
            l_mean_rel_error= l_mean_error / c_bar_cur
            a_mean_error = np.mean(np.abs(area_desired-actual_area))
            a_mean_rel_error= a_mean_error / np.mean(area_desired)
            #if np.mod(it,10)==0:
            #    print("it, {:3d} {:6.3f}%".format(it, l_mean_rel_error*100), 'Uniform:', c_bar_new/c_bar_cur )
            if l_mean_rel_error*100 < 1:
                print('Stopping after  ', it, 'iterations {:6.3f}%'.format(l_mean_rel_error*100), 'avg', c_bar_new/c_bar_new )
                break






    #if 0 < volSmooth < 1:
    #    #import pdb; pdb.set_trace()
    #    m2 = np.roll(normals, -1, axis=0)  # Shifted forward by 2
    #    p2 = np.roll(normals,  1, axis=0)  # Shifted forward by 2
    #    normals = volSmooth * normals + (1 - volSmooth) * 0.5 * (p2+m2)
    else:
        # --- new_layer = front + normals * effective_dy[:, np.newaxis]
        growth_vec= normal_growth(normals, effective_dy)
        new_layer = consistent_growth(front, growth_vec, normals)

    # Step 3: Apply Volume Smoothing
    #if 0 < volSmooth < 1:
    #    smoothed_front = np.copy(new_layer)  # Copy to avoid overwriting
    #    #import pdb; pdb.set_trace()
    #    m2 = np.roll(new_layer, -2, axis=0)  # Shifted forward by 2
    #    p2 = np.roll(new_layer,  2, axis=0)  # Shifted forward by 2
    #    smoothed_front = volSmooth * new_layer + (1 - volSmooth) * 0.5 * (p2+m2)
    #    #smoothed_front[1:-1, :] = volSmooth * new_layer[1:-1] + (1 - volSmooth) * 0.5 * (new_layer[:-2] + new_layer[2:])
    #    #smoothed_front[1:-2, :] = volSmooth * new_layer[1:-2] + (1 - volSmooth) * 0.5 * (new_layer[:-3] + new_layer[2:-1])
    #    new_layer = smoothed_front

    #    #smoothed_layer = (1 - volSmooth) * new_layer + volSmooth * front
    #    #growth_vec = smoothed_front - front
    #    #new_layer = consistent_growth(front, growth_vec, normals, new_layer)

    # Step 4: Quad Quality Check and Correction
    # Form quads from old_layer and current_new_layer
    # Ensure old_layer and current_new_layer have the same number of points for checking

    p0_quad = front[:-1] if is_loop else front # old layer points i
    p1_quad = np.roll(front, -1, axis=0)[:-1] if is_loop else np.roll(front, -1, axis=0) # old layer points i+1
    p3_quad = new_layer[:-1] if is_loop else new_layer # new layer points i
    p2_quad = np.roll(new_layer, -1, axis=0)[:-1] if is_loop else np.roll(new_layer, -1, axis=0) # new layer points i+1
    bad_quad_indices = np.where(check_quad_quality(p0_quad, p1_quad, p2_quad, p3_quad))[0]

    if len(bad_quad_indices) > 0:
        pass
        #
        #print(f"[WARN]: Detected {len(bad_quad_indices)} / {len(p0)} poor quality quads.")
        # Implement a correction strategy. This could be localized adjustment or backtracking.
        # A more robust solution might involve reducing the local dy for the next extrusion step

        # Let's try to adjust the points in the new layer that are part of bad quads
        # The points in the new layer corresponding to bad_quad_indices are bad_quad_indices
        #new_layer = adjust_point_locally(new_layer, front, normals, bad_quad_indices, dy)

        ## Re-check quality after adjustment (optional, but good practice)
        #p0_quad = front[:-1] if is_loop else front # old layer points i
        #p1_quad = np.roll(front, -1, axis=0)[:-1] if is_loop else np.roll(front, -1, axis=0) # old layer points i+1
        #p3_quad = new_layer[:-1] if is_loop else new_layer # new layer points i
        #p2_quad = np.roll(new_layer, -1, axis=0)[:-1] if is_loop else np.roll(new_layer, -1, axis=0) # new layer points i+1
        #still_bad_indices = np.where(check_quad_quality(p0_quad, p1_quad, p2_quad, p3_quad))[0]
        #if len(still_bad_indices) > 0:
        #     if debug:
        #         print(f"[ERROR]: {len(still_bad_indices)} quads are still poor quality after adjustment.")
        #     # Depending on your needs, you might raise an error, log, or handle this externally
        #     # For now, we'll proceed with the adjusted layer
        #     pass # Or add more robust error handling


    if debug:
        print(f"New layer after GEM extrusion (first 3 points): {new_layer[:3]}")

    return new_layer

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
    is_loop=np.allclose(front[0], front[-1])
    front=front[:-1]  # Remove the last point for processing


    #fig,ax = plt.subplots(1, 1, sharey=False, figsize=(6.4,4.8))
    #fig.subplots_adjust(left=0.12, right=0.95, top=0.95, bottom=0.11, hspace=0.20, wspace=0.20)
    #ax.plot(front[:,0], front[:,1], '+-', label='old')

    ## Extrude the front
    #new_front = front_extrude_manu(front, dy=0.1, volSmooth=0.3, debug=True, max_iterations=max_iteration, fig=fig, is_loop=is_loop)
    #ax.plot(new_front[:,0], new_front[:,1], 'd-', label='new')

    ## Print the new front
    #print("New Front:")
    #print(new_front)

    #ax.set_xlabel('')
    #ax.set_ylabel('')
    #ax.legend()
    #plt.show()



    # User inputs
    nBody = 200
    h0 = 0.00000278  # first layer height
    h0 = 0.0000278  # first layer height
    gr_near = 1.04   # growth rate near the body
    gr_away = 1.09   # growth rate away from the body
    hmax_near = 0.4  # max height near the body
    hmax_away = 10   # max height away from the body
    kbSmooth_near = 3     # Kinsey-Barth smoothing near the body
    volSmooth_near = 0.3  # volume smoothing near the body
    kbSmooth_away  = 0     # Kinsey-Barth smoothing away from the body
    volSmooth_away = 0.5  # volume smoothing away from the body
    max_layers = 300

    h0 = 0.001  # first layer height
    gr_near = 1.02   # growth rate near the body
    #hmax_near = 0.1  # max height near the body
    #hmax_away = 1   # max height away from the body
    #kbSmooth_near = 3     # Kinsey-Barth smoothing near the body
    #volSmooth_near = 0.0  # volume smoothing near the body
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
    #coords = front
    layers_near, dys_near = hyp_extrude(coords, h0, gr_near, hmax_near, kbSmooth=kbSmooth_near, volSmooth=volSmooth_near, method='manu', is_loop=is_loop, max_layers=max_layers)
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
