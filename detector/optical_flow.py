"""Optical flow analysis for temporal consistency checking."""

import numpy as np
import cv2


def compute_flow(frame_a: np.ndarray, frame_b: np.ndarray) -> np.ndarray:
    """Compute dense optical flow between two frames.
    
    Args:
        frame_a: First frame (H, W) or (H, W, 3) uint8
        frame_b: Second frame (H, W) or (H, W, 3) uint8
    
    Returns:
        Optical flow field (H, W, 2) with (dx, dy) displacement vectors
    """
    # Convert to grayscale if needed
    if len(frame_a.shape) == 3:
        gray_a = cv2.cvtColor(frame_a, cv2.COLOR_BGR2GRAY)
    else:
        gray_a = frame_a
        
    if len(frame_b.shape) == 3:
        gray_b = cv2.cvtColor(frame_b, cv2.COLOR_BGR2GRAY)
    else:
        gray_b = frame_b
    
    # Compute flow using Farneback method
    flow = cv2.calcOpticalFlowFarneback(
        gray_a,
        gray_b,
        None,
        pyr_scale=0.5,
        levels=3,
        winsize=15,
        iterations=3,
        poly_n=5,
        poly_sigma=1.2,
        flags=0,
    )
    
    return flow


def warping_error(flow_a: np.ndarray, flow_b: np.ndarray) -> float:
    """Compute warping error between forward and backward flow.
    
    This measures how well the forward and backward flows are consistent.
    Low error indicates smooth, consistent motion.
    
    Args:
        flow_a: Forward optical flow (H, W, 2)
        flow_b: Backward optical flow (H, W, 2)
    
    Returns:
        Mean warping error (lower is better)
    """
    h, w = flow_a.shape[:2]
    
    # Create coordinate grids
    y_coords, x_coords = np.mgrid[0:h, 0:w].astype(np.float32)
    
    # Warp forward flow using backward flow
    # Sample flow_a at locations displaced by flow_b
    map_x = x_coords + flow_b[:, :, 0]
    map_y = y_coords + flow_b[:, :, 1]
    
    # Warp flow_a
    warped_flow_a = cv2.remap(
        flow_a[:, :, 0],
        map_x,
        map_y,
        cv2.INTER_LINEAR,
        borderMode=cv2.BORDER_REPLICATE,
    )
    
    # Compare with negative backward flow
    error = np.sqrt(np.sum((warped_flow_a + flow_b[:, :, 0]) ** 2))
    
    # Normalize by image size
    return error / (h * w)


def flow_consistency(flow_sequence: list[np.ndarray]) -> float:
    """Compute temporal consistency across a sequence of optical flows.
    
    Measures how smoothly the flow fields evolve over time.
    
    Args:
        flow_sequence: List of optical flow fields, each (H, W, 2)
    
    Returns:
        Consistency score (0-1, 1 = perfectly consistent)
    """
    if len(flow_sequence) < 2:
        return 1.0
    
    consistency_scores = []
    
    for i in range(len(flow_sequence) - 1):
        flow_curr = flow_sequence[i]
        flow_next = flow_sequence[i + 1]
        
        # Compute flow magnitude and direction
        mag_curr = np.sqrt(np.sum(flow_curr ** 2, axis=2))
        mag_next = np.sqrt(np.sum(flow_next ** 2, axis=2))
        
        # Compute angle
        angle_curr = np.arctan2(flow_curr[:, :, 1], flow_curr[:, :, 0])
        angle_next = np.arctan2(flow_next[:, :, 1], flow_next[:, :, 0])
        
        # Magnitude consistency (normalized)
        mag_diff = np.mean(np.abs(mag_curr - mag_next)) / (np.mean(mag_curr) + 1e-6)
        
        # Direction consistency (angle difference)
        angle_diff = np.abs(angle_curr - angle_next)
        angle_diff = np.minimum(angle_diff, 2 * np.pi - angle_diff)
        angle_consistency = 1.0 - np.mean(angle_diff) / np.pi
        
        # Combined consistency for this pair
        pair_consistency = 0.5 * (1.0 - min(mag_diff, 1.0)) + 0.5 * angle_consistency
        consistency_scores.append(pair_consistency)
    
    return float(np.mean(consistency_scores))


def flow_magnitude_stats(flow: np.ndarray) -> dict:
    """Compute statistics of optical flow magnitude.
    
    Args:
        flow: Optical flow field (H, W, 2)
    
    Returns:
        Dictionary with mean, median, std, max of flow magnitude
    """
    magnitude = np.sqrt(np.sum(flow ** 2, axis=2))
    
    return {
        "mean": float(np.mean(magnitude)),
        "median": float(np.median(magnitude)),
        "std": float(np.std(magnitude)),
        "max": float(np.max(magnitude)),
    }


def detect_flow_anomalies(flow: np.ndarray, threshold: float = 2.0) -> np.ndarray:
    """Detect regions with anomalous optical flow.
    
    Args:
        flow: Optical flow field (H, W, 2)
        threshold: Standard deviations from mean to consider anomalous
    
    Returns:
        Boolean mask (H, W) where True indicates anomalous regions
    """
    magnitude = np.sqrt(np.sum(flow ** 2, axis=2))
    
    mean_mag = np.mean(magnitude)
    std_mag = np.std(magnitude)
    
    # Anomalies are regions with magnitude > threshold * std from mean
    anomalies = magnitude > (mean_mag + threshold * std_mag)
    
    return anomalies