""" Set of tools to manipulate angles"""
import numpy as np

def compute_angles(coords, center):
    """
    Compute angles of nodes relative to the center.

    Parameters:
        coords (np.ndarray): Node coordinates (N x 3).
        center (tuple): Center of rotation (x, y).

    Returns:
        np.ndarray: Angles in radians.
    """
    x_center, y_center = center
    relative_coords = coords[:, :2] - np.array([x_center, y_center])
    angles = np.arctan2(relative_coords[:, 1], relative_coords[:, 0])
    return np.mod(angles, 2 * np.pi)  # Ensure angles are in [0, 2π]

def is_angle_in_segment(angles, start_angle, span, strict_upper=True):
    """
    Check if angles are within a given angular segment.

    Parameters:
        angles (np.ndarray): Angles to check (in radians, normalized to [0, 2π]).
        start_angle (float): Start angle of the segment (in radians, normalized to [0, 2π]).
        span (float): Angular span of the segment (in radians).
        strict_upper (bool): Whether to use a strict upper bound for the segment.

    Returns:
        np.ndarray: Boolean array indicating whether each angle is within the segment.
    """
    angles = np.mod(angles, 2 * np.pi)
    start_angle = np.mod(start_angle, 2 * np.pi)
    end_angle = np.mod(start_angle + span, 2 * np.pi)

    if strict_upper:
        if end_angle > start_angle:
            return (angles >= start_angle) & (angles < end_angle)
        else:  # Segment wraps around 0
            return (angles >= start_angle) | (angles < end_angle)
    else:
        if end_angle > start_angle:
            return (angles >= start_angle) & (angles <= end_angle)
        else:  # Segment wraps around 0
            return (angles >= start_angle) | (angles <= end_angle)

def angle_segment(angles):
    """
    Find the angular segment that contains all the given angles in the anti-clockwise direction.

    Parameters:
        angles (np.ndarray): Angles in radians (normalized to [0, 2π]).

    Returns:
        tuple: Start angle and span of the segment (in radians).
    """
    # Normalize angles to [0, 2π]
    angles = np.mod(angles, 2 * np.pi)
    angles = np.sort(angles)

    # Compute differences between consecutive angles, including wrap-around
    diffs = np.diff(np.concatenate((angles, [angles[0] + 2 * np.pi])))

    # Find the largest gap (clockwise direction)
    max_gap_idx = np.argmax(diffs)
    start_angle = angles[(max_gap_idx + 1) % len(angles)]  # Start angle is after the largest gap
    span = 2 * np.pi - diffs[max_gap_idx]  # Span is the complement of the largest gap

    return start_angle, span