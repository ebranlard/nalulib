import numpy as np
import matplotlib.pyplot as plt

def quad_area(a, b, c, d):
    """
    Compute the area of a quad using the shoelace formula.

    Args:
        a, b, c, d (np.ndarray): (n, 2) arrays representing the quad vertices.

    Returns:
        np.ndarray: Array of areas for each quad.
    """
    return 0.5 * (
        a[:, 0] * (b[:, 1] - d[:, 1]) +
        b[:, 0] * (c[:, 1] - a[:, 1]) +
        c[:, 0] * (d[:, 1] - b[:, 1]) +
        d[:, 0] * (a[:, 1] - c[:, 1])
    )

def smooth_heaviside(eta, scale):
    """
    Smooth Heaviside function that transitions from 0 to 1.

    Args:
        eta (np.ndarray): Array of eta values.
        scale (float): Scaling factor for the transition.

    Returns:
        np.ndarray: Smooth Heaviside values.
    """
    return 1 / (1 + np.exp(-scale * eta))


def calculate_quad_base_params(p0, p1):
    # Compute the base length (length scale)
    base_length = np.linalg.norm(p1 - p0, axis=1, keepdims=True)  # Length of the base
    # Transform points into the local coordinate system of the quad
    base_vector = p1 - p0  # Vector along the base (p0-p1)
    base_unit = base_vector / base_length  # Unit vector along the base
    # Compute the perpendicular (eta) direction
    eta_unit = np.stack([-base_unit[:, 1], base_unit[:, 0]], axis=1)  # Rotate base_unit by 90 degrees
    base_length = base_length.flatten() 
    return base_length, base_unit, eta_unit

def smoothed_area(p0, p1, p2, p3, base_params=None):
    """
    Smoothed area of a quad (p0, p1, p2, p3), with a penalty based on the eta coordinate in the quad's local system.

    Inputs:
      p0, p1, p2, p3: (n,2) arrays

    Returns:
      (n,) smoothed areas
    """
    # Compute the area using the shoelace formula
    area = quad_area(p0, p1, p2, p3)

    if base_params is None:
        base_length, base_unit, eta_unit = calculate_quad_base_params(p0, p1)
    else:
        # Use provided base parameters
        base_length = base_params['base_length']
        eta_unit = base_params['eta_unit']

    # Compute eta coordinates for p2 and p3
    p2_relative = p2 - p0
    p3_relative = p3 - p0
    xi_p2 = np.einsum('ij,ij->i', p2_relative, base_unit) / base_length  # Normalize by base length
    xi_p3 = np.einsum('ij,ij->i', p3_relative, base_unit) / base_length  # Normalize by base length
    eta_p2 = np.einsum('ij,ij->i', p2_relative, eta_unit) / base_length  # Normalize by base length
    eta_p3 = np.einsum('ij,ij->i', p3_relative, eta_unit) / base_length  # Normalize by base length

    # Apply smooth Heaviside functions
    scale = 30.0  # Scaling factor for the smooth transition
    heaviside_p2 = smooth_heaviside(eta_p2, scale)
    heaviside_p3 = smooth_heaviside(eta_p3, scale)
    heaviside_xi = smooth_heaviside(xi_p2-xi_p3, scale)
    print(eta_p2, eta_p3)  # Debugging line to check the values of eta_p2 and eta_p3
    # Combine penalties and scale the area
    penalty = heaviside_p2 * heaviside_p3 * heaviside_xi
    area = area * penalty
    #import pdb; pdb.set_trace()  # Debugging line to check the values of area and penalty

    return area

def penalized_area(p0, p1, p2, p3, base_params=None):
    """
    Penalized area of a quad (p0, p1, p2, p3), with a penalty based on the eta and xi coordinates
    in the quad's local system.

    Inputs:
      p0, p1, p2, p3: (n,2) arrays

    Returns:
      (n,) penalized areas
    """
    # Compute the area using the shoelace formula
    area = quad_area(p0, p1, p2, p3)

    if base_params is None:
        base_length, base_unit, eta_unit = calculate_quad_base_params(p0, p1)
    else:
        # Use provided base parameters
        base_length = base_params['base_length']
        base_unit = base_params['base_unit']
        eta_unit = base_params['eta_unit']

    # Compute relative positions of p2 and p3
    p2_relative = p2 - p0
    p3_relative = p3 - p0

    # Compute xi and eta coordinates for p2 and p3
    xi_p2 = np.einsum('ij,ij->i', p2_relative, base_unit) / base_length  # Normalize by base length
    xi_p3 = np.einsum('ij,ij->i', p3_relative, base_unit) / base_length  # Normalize by base length
    eta_p2 = np.einsum('ij,ij->i', p2_relative, eta_unit) / base_length  # Normalize by base length
    eta_p3 = np.einsum('ij,ij->i', p3_relative, eta_unit) / base_length  # Normalize by base length

    # Penalize if eta < 0 or xi_p2 > xi_p3 (crossing condition)
    invalid_condition = (eta_p2 < 0) | (eta_p3 < 0) | (xi_p2 > xi_p3)  # Logical OR for vectorized condition
    area = np.where(invalid_condition, 0.0, area)  # Set area to 0 where the condition is True
    #scale = 30.0  # Scaling factor for the smooth transition
    #heaviside_p2 = smooth_heaviside(eta_p2, scale)
    #heaviside_p3 = smooth_heaviside(eta_p3, scale)
    #heaviside_xi = smooth_heaviside(xi_p2-xi_p3, scale)
    #penalty = heaviside_p2 * heaviside_p3 * heaviside_xi
    #area = area * penalty

    return area


if __name__ == "__main__":
    #p0 = np.array([[0.0, 0.0]])  # bottom left
    #p1 = np.array([[1.0, 0.0]])  # bottom right
    #p2 = np.array([[1.0, 1.0]])  # top right
    #p3_base = np.array([[0.0, 1.0]])  # top left -- we will vary its y
    p0 = np.array([[1.0, 0.0]])  # bottom left
    p1 = np.array([[1.0, 1.0]])  # bottom right
    p2 = np.array([[0.0, 1.0]])  # top right
    p3_base = np.array([[0.0, 0.0]])  # top left -- we will vary its y

    area = smoothed_area(p0, p1, p2, p3_base)  # should be 1.0
    print('Area of square:', area[0])

    # Vary the y position of p3
    y_vals = np.linspace(2.0, -2.0, 100)  # from high above to below

    areas = []
    areas2 = []

    for y in y_vals:
        #p3 = np.array([[0.0, y]])  # move only y-coordinate
        #p2 = np.array([[1.0, 2-y]])  # move only y-coordinate
        p3 = np.array([[y, 0]])  # move only y-coordinate
        p2 = np.array([[0.0, 1]])  # move only y-coordinate
        #area = smoothed_area(p0, p1, p2, p3)
        area = penalized_area(p0, p1, p2, p3)
        areas.append(area[0])
        areas2.append(0.5 * (p0[:, 0] * (p1[:, 1] - p3[:, 1]) + p1[:, 0] * (p2[:, 1] - p0[:, 1]) + p2[:, 0] * (p3[:, 1] - p1[:, 1]) + p3[:, 0] * (p0[:, 1] - p2[:, 1]))[0])

    areas = np.array(areas)
    areas2 = np.array(areas2)

    # Plot
    fig, ax1 = plt.subplots(figsize=(8,5))
    ax1.plot(y_vals, areas2,      label='True area')
    ax1.plot(y_vals, areas ,'--', label='Penalized Area')
    ax1.set_xlabel('Top Left y-coordinate (p3_y)')
    ax1.set_ylabel('Smoothed Area', color='blue')
    ax1.tick_params(axis='y', labelcolor='blue')
    ax1.grid(True)
    ax1.legend()


    plt.title("Smoothed Area vs p3_y (Top-Left Corner Height)")
    fig.tight_layout()
    plt.show()
