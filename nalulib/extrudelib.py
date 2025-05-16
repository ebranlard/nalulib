import numpy as np
from airfoillib import compute_normals, smooth_normals




def normal_growth(normals, dy):
    if not hasattr(dy, '__len__'):
        dy = np.full(normals.shape[0], dy)
    growth_vec  = normals * dy[:, np.newaxis]  # Extrude along normals
    return growth_vec

def consistent_growth(current_layer, growth_vec, normals, backup_points=None, debug=False):
    """
    Ensure consistent growth by flipping normals if the growth direction is incorrect.
    Args:
        current_layer (np.ndarray): Current layer of points.
        growth_vec
    Returns:
        np.ndarray: Updated layer with consistent growth.
        np.ndarray: Updated normals with corrected directions.
    """
    new_layer = np.zeros_like(current_layer)  # Pre-allocate for efficiency
    for i in range(len(new_layer)):
        dy = np.dot(growth_vec[i], normals[i])
        if dy < 0:  # If growth is in the wrong direction
            #new_layer[i] = current_layer[i] + np.abs(dy) * normals[i]
            if backup_points is not None:
                new_layer[i] = backup_points[i]
            else:
                new_layer[i] = current_layer[i] - growth_vec[i]
                #import pdb; pdb.set_trace()
            print(f"Growth issue detected at point {i}. ")
        else:
            new_layer[i] = current_layer[i] + growth_vec[i]
    return new_layer


def front_extrude_base(front, dy, kbSmooth=3, volSmooth=0.3, explicit=0.5, implicit=None, debug=False):
    """
    Extrude a new layer from the current front using hyperbolic smoothing and volume control.
    
    Args:
        front (np.ndarray): Current front (layer) of points.
        dy (float): Layer spacing for the extrusion.
        kbSmooth (int): Number of Kinsey-Barth smoothing iterations (default: 3).
        volSmooth (float): Volume smoothing factor (0 to 1, default: 0.3).
        explicit (float): Explicit smoothing parameter (0 to 10, default: 0.5).
        implicit (float): Implicit smoothing parameter (default: 2 * explicit).
        debug (bool): If True, print debug information.
    Returns:
        np.ndarray: New extruded layer.
    """
    if implicit is None:
        implicit = 2 * explicit  # Default implicit smoothing is double the explicit smoothing

    # Compute normals for the current front
    normals, _ = compute_normals(front)

    # Compute the new layer by extruding along the normals
    growth_vec= normal_growth(normals, dy)
    new_layer = consistent_growth(front, growth_vec, normals)

    if debug:
        print(f"Extruding new layer with dy={dy}")
        print(f"New layer (first 3 points): {new_layer[:3]}")

    # Apply Kinsey-Barth smoothing
    if kbSmooth > 0:
        for _ in range(kbSmooth):
            new_layer = 0.5 * (new_layer + 0.5 * (front + new_layer))

    # Apply explicit smoothing
    if explicit > 0:
        explicit_smoothing = explicit * (front - new_layer)
        new_layer += explicit_smoothing

    # Apply implicit smoothing
    if implicit > 0:
        implicit_smoothing = implicit * compute_normals(new_layer)[0]
        normals_new = (1 - volSmooth) * normals + volSmooth * implicit_smoothing
        normals_new /= np.linalg.norm(normals, axis=1)[:, np.newaxis]  # Normalize
        growth_vec= normal_growth(normals_new, dy)
        new_layer = consistent_growth(front, growth_vec, normals)
        new_layer = front + normals * dy

    return new_layer

def front_extrude_TMHW(front, dy, kbSmooth=3, volSmooth=0.3, explicit=0.5, implicit=None, debug=False):
    """
    Extrude a new layer from the current front using a combination of Thomas-Middlecoff
    and Hilgenstock-White algorithms for hyperbolic smoothing and volume control.
    
    Args:
        front (np.ndarray): Current front (layer) of points.
        dy (float): Layer spacing for the extrusion.
        kbSmooth (int): Number of Kinsey-Barth smoothing iterations (default: 3).
        volSmooth (float): Volume smoothing factor (0 to 1, default: 0.3).
        explicit (float): Explicit smoothing parameter (0 to 10, default: 0.5).
        implicit (float): Implicit smoothing parameter (default: 2 * explicit).
        debug (bool): If True, print debug information.
    Returns:
        np.ndarray: New extruded layer.
    """
    if implicit is None:
        implicit = 2 * explicit  # Default implicit smoothing is double the explicit smoothing

    # Compute normals for the current front
    normals, _ = compute_normals(front)

    # Step 1: Initial extrusion along normals

    growth_vec= normal_growth(normals, dy)
    new_layer = consistent_growth(front, growth_vec, normals)

    if debug:
        print(f"Initial extrusion with dy={dy}")
        print(f"New layer (first 3 points): {new_layer[:3]}")

    # Step 2: Apply Kinsey-Barth smoothing to prevent crossing in concave regions
    if kbSmooth > 0:
        for _ in range(kbSmooth):
            new_layer = 0.5 * (new_layer + 0.5 * (front + new_layer))

    # Step 3: Apply explicit smoothing (Thomas-Middlecoff approach)
    if explicit > 0:
        explicit_smoothing = explicit * (front - new_layer)
        new_layer += explicit_smoothing

    # Step 4: Apply implicit smoothing (Hilgenstock-White approach)
    if implicit > 0:
        implicit_smoothing = implicit * compute_normals(new_layer)[0]
        normals_new = (1 - volSmooth) * normals + volSmooth * implicit_smoothing
        normals_new /= np.linalg.norm(normals, axis=1)[:, np.newaxis]  # Normalize
        growth_vec= normal_growth(normals_new, dy)
        new_layer = consistent_growth(front, growth_vec, normals)

    # Step 5: Volume smoothing (Hilgenstock-White approach)
    if volSmooth > 0:
        smoothed_layer = (1 - volSmooth) * new_layer + volSmooth * front
        #smoothed_front[1:-1] = (volSmooth * new_front[1:-1] + (1 - volSmooth) * 0.5 * (new_front[:-2] + new_front[2:]))
        new_layer = smoothed_layer

    if debug:
        print(f"Final new layer after TMHW smoothing (first 3 points): {new_layer[:3]}")

    return new_layer



def front_extrude_gem(front, dy, kbSmooth=3, volSmooth=0.3, implicit_or_explicit='implicit', debug=False):
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
    num_points = front.shape[0]
    new_front = np.zeros_like(front)  # Pre-allocate for efficiency

    # Compute normals for the current front
    normals, _ = compute_normals(front)

    # Determine if the front is a closed loop
    is_loop = np.allclose(front[0], front[-1])

    if debug:
        print(f"Is loop: {is_loop}, kbSmooth: {kbSmooth}")

    # Step 1: Apply Kinsey-Barth smoothing to prevent grid line crossing
    effective_dy = np.full(num_points, dy)  # Initialize with dy
    if kbSmooth >= 3:
        # Calculate curvature (approximation) - vectorized
        front_prev = np.roll(front, 1, axis=0)  # Shifted back by 1
        front_next = np.roll(front, -1, axis=0)  # Shifted forward by 1

        if is_loop:
            prev_vectors = front_prev - front
            next_vectors = front_next - front
            prev_angles = np.arctan2(prev_vectors[:, 1], prev_vectors[:, 0])
            next_angles = np.arctan2(next_vectors[:, 1], next_vectors[:, 0])
            # Handle angle wraparound (important for loops)
            d_angles = next_angles - prev_angles
            d_angles = (d_angles + np.pi) % (2 * np.pi) - np.pi
            curvature = d_angles
        else:
            curvature = np.zeros(num_points)
            prev_vectors = front_prev - front
            next_vectors = front_next - front
            for i in range(1, num_points - 1):
                prev_angle = np.arctan2(prev_vectors[i, 1], prev_vectors[i, 0])
                next_angle = np.arctan2(next_vectors[i, 1], next_vectors[i, 0])
                curvature[i] = (next_angle - prev_angle)

        kb_factor = 1.0 / (1 + kbSmooth * np.abs(curvature) * dy)
        effective_dy = dy * kb_factor
        if any(effective_dy < 0):
            print("[WARN] Number of negative effective_dy values:", np.sum(effective_dy < 0))
            effective_dy = np.abs(effective_dy)  # Ensure non-negative



    if 0 < volSmooth < 1:
        normals= smooth_normals(normals, iterations=50, alpha=volSmooth, boundary_weight=0.8)

    # Step 2: Calculate new point positions (Explicit or Implicit)
    if implicit_or_explicit.lower() == 'explicit':
        growth_vec= normal_growth(normals, effective_dy)
        new_layer = consistent_growth(front, growth_vec, normals)
        #new_layer = front + normals * effective_dy[:, np.newaxis]
    elif implicit_or_explicit.lower() == 'implicit':
        # Simplified implicit method
        growth_vec= normal_growth(normals, effective_dy)
        new_layer = consistent_growth(front, growth_vec, normals)
        #new_layer = front + normals * effective_dy[:, np.newaxis]
    else:
        raise ValueError("implicit_or_explicit must be 'explicit' or 'implicit'")

    # Step 3: Apply Volume Smoothing
    #if 0 < volSmooth < 1:
    #    smoothed_front = np.copy(new_layer)  # Copy to avoid overwriting
    #    #import pdb; pdb.set_trace()
    #    smoothed_front[1:-1, :] = volSmooth * new_layer[1:-1] + (1 - volSmooth) * 0.5 * (new_layer[:-2] + new_layer[2:])
    #    #smoothed_front[1:-2, :] = volSmooth * new_layer[1:-2] + (1 - volSmooth) * 0.5 * (new_layer[:-3] + new_layer[2:-1])
    #    new_layer = smoothed_front

        #smoothed_layer = (1 - volSmooth) * new_layer + volSmooth * front
        #growth_vec = smoothed_front - front
        #new_layer = consistent_growth(front, growth_vec, normals, new_layer)

    if debug:
        print(f"New layer after GEM extrusion (first 3 points): {new_layer[:3]}")

    return new_layer

def plot_layers(layers_near, layers_away, near=None, away='all'):
    """
    Plot the first, second, and last layers of the near and away regions.
    
    Args:
        layers_near (np.ndarray): Layers generated near the body.
        layers_away (np.ndarray): Layers generated away from the body.
    """
    fig, ax = plt.subplots(figsize=(10, 6))


    # Plot near layers
    if near is None:
        ax.plot(layers_near[0][:, 0], layers_near[0][:, 1], 'b-', label="Near - First Layer")
        if len(layers_near) > 1:
            ax.plot(layers_near[1][:, 0], layers_near[1][:, 1], 'g-', label="Near - Second Layer")
        if len(layers_near) > 2:
            ax.plot(layers_near[-2][:, 0], layers_near[-2][:, 1], 'r-', label="Near - Before Last Layer")
        ax.plot(layers_near[-1][:, 0], layers_near[-1][:, 1], 'r-', label="Near - Last Layer")
    if near is 'all':
        for i in range(len(layers_near)):
            ax.plot(layers_near[i][:, 0], layers_near[i][:, 1], 'b-', label=f"Near - Layer {i+1}" if i ==0 else '')

    # Plot away layers
    if away is None:
        ax.plot(layers_away[0][:, 0], layers_away[0][:, 1], 'c--', label="Away - First Layer")
        if len(layers_away) > 1:
            ax.plot(layers_away[1][:, 0], layers_away[1][:, 1], 'm--', label="Away - Second Layer")
        if len(layers_away) > 2:
            ax.plot(layers_away[2][:, 0], layers_away[2][:, 1], 'm--', label="Away - Third Layer")
        ax.plot(layers_away[-1][:, 0], layers_away[-1][:, 1], 'y--', label="Away - Last Layer")
    if away is 'all':     
        for i in range(len(layers_away)):
            ax.plot(layers_away[i][:, 0], layers_away[i][:, 1], 'c--', label=f"Away - Layer {i+1}" if i ==0 else '')

    ax.set_title("First, Second, and Last Layers of Near and Away Regions")
    ax.set_xlabel("X")
    ax.set_ylabel("Y")
    ax.legend()
    ax.axis('equal')


def check_for_crossing_lines_between_layers(layer1, layer2, iter=0):
    """
    Check if any mesh lines or layer lines cross each other between two layers.
    
    Args:
        layer1 (np.ndarray): First layer of points.
        layer2 (np.ndarray): Second layer of points.
    Returns:
        bool: True if crossing lines are detected, False otherwise.
    """
    for j in range(len(layer1) - 1):
        # Compute vectors for adjacent cells
        v1 = layer2[j] - layer1[j]
        v2 = layer2[j + 1] - layer1[j + 1]

        # Check if the vectors cross
        cross_product = np.cross(v1, v2)
        if cross_product < 0:  # If the cross product is negative, lines are crossing
            print(f"Crossing detected between points {j} and {j + 1} in the layer {iter}.")
            return True
    return False