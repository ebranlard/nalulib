import os
import matplotlib.pyplot as plt
import numpy as np

from nalulib.curves import contour_is_clockwise as is_clockwise 

# ---------------------------------------------------------------------------
# --- Airfoil library
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
    if is_clockwise(coords):
        print("Reversing airfoil coordinates to ensure counterclockwise order.")
        coords = coords[::-1]
    return coords

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


def detect_airfoil_te_type(coords):
    """
    Detects the type of trailing edge and returns the trailing edge indices and type.
    Args:
        coords (np.ndarray): Airfoil coordinates.
    Returns:
        trailing_edge_type (str): Type of trailing edge ("flatback", "sharp", or "cusp").
        trailing_edge_indices (np.ndarray): Indices of trailing edge points.
    """
    # Identify trailing edge points (x == 1.0)
    trailing_edge_indices = np.where(np.isclose(coords[:, 0], 1.0))[0]
    if len(trailing_edge_indices) > 1:
        trailing_edge_type = "flatback"
    elif len(trailing_edge_indices) == 1:
        trailing_edge_type = "sharp"
    else:
        trailing_edge_type = "cusp"

    return trailing_edge_type, trailing_edge_indices


def reindex_starting_from_te(coords, TE_indices):
    """
    Reindexes the airfoil so that the first point is at the middle of the flat trailing edge.
    Args:
        coords (np.ndarray): Airfoil coordinates.
        trailing_edge_indices (np.ndarray): Indices of trailing edge points.
    Returns:
        reindexed_coords (np.ndarray): Reindexed airfoil coordinates.
    """
    TE_indices = list(TE_indices)

    # Remove the last point (to avoid duplication)
    nCoords = len(coords)
    coords = coords[:-1]
    if nCoords-1 in TE_indices:  
        TE_indices.remove(nCoords-1)

    # If the trailing edge is flatback, reindex
    if len(TE_indices) > 1:
        if np.mod(len(TE_indices), 2) == 0:
            # If even number of trailing edge points, add a point in the middle
            # to avoid ambiguity in the middle point

            raise NotImplementedError("Even number of trailing edge points not supported yet.")
        else:
            mid_te_index = TE_indices[len(TE_indices)// 2-1]
        reindexed_coords = np.roll(coords, -mid_te_index, axis=0)
    else:
        # If not flatback, no reindexing is needed
        reindexed_coords = coords

    # Add the first point back to close the loop
    reindexed_coords = np.vstack([reindexed_coords, reindexed_coords[0]])

    return reindexed_coords


def detect_airfoil_features(coords):
    """
    Detects trailing edge, leading edge, upper surface, and lower surface points.
    Returns:
        trailing_edge_type (str): Type of trailing edge ("flatback", "sharp", or "cusp").
        indices (dict): Dictionary containing indices for TE, LE, upper, and lower surfaces.
    """
    # Identify trailing edge points (x == 1.0)
    trailing_edge_indices = np.where(np.isclose(coords[:, 0], 1.0))[0]
    if len(trailing_edge_indices) > 1:
        trailing_edge_type = "flatback"
    elif len(trailing_edge_indices) == 1:
        trailing_edge_type = "sharp"
    else:
        trailing_edge_type = "cusp"

    # Identify leading edge point (minimum x-coordinate)
    leading_edge_index = np.argmin(coords[:, 0])

    # Split into upper and lower surfaces
    upper_indices = np.arange(0, leading_edge_index + 1)
    lower_indices = np.arange(leading_edge_index, len(coords))

    # Include the first and last trailing edge points in upper and lower surfaces
    upper_indices = np.append(upper_indices, trailing_edge_indices[0])
    lower_indices = np.insert(lower_indices, 0, trailing_edge_indices[-1])

    indices = {
        "trailing_edge": trailing_edge_indices,
        "leading_edge": leading_edge_index,
        "upper_surface": upper_indices,
        "lower_surface": lower_indices,
    }

    return trailing_edge_type, indices
