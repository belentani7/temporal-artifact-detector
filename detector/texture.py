"""Texture feature analysis for instability detection."""

import numpy as np
import cv2


def compute_texture_features(frame: np.ndarray) -> np.ndarray:
    """Extract texture features from a frame.
    
    Computes a combination of:
    - Color histogram (global appearance)
    - Edge histogram (structural content)
    - Gabor-like features (texture patterns)
    
    Args:
        frame: Input frame (H, W, 3) uint8
    
    Returns:
        Feature vector representing texture characteristics
    """
    # Resize for consistent feature dimensions
    resized = cv2.resize(frame, (256, 256))
    
    # Color histogram
    if len(resized.shape) == 3:
        hist_b = cv2.calcHist([resized], [0], None, [32], [0, 256]).flatten()
        hist_g = cv2.calcHist([resized], [1], None, [32], [0, 256]).flatten()
        hist_r = cv2.calcHist([resized], [2], None, [32], [0, 256]).flatten()
        color_hist = np.concatenate([hist_b, hist_g, hist_r])
    else:
        color_hist = cv2.calcHist([resized], [0], None, [64], [0, 256]).flatten()
    
    color_hist = color_hist / (np.sum(color_hist) + 1e-6)
    
    # Convert to grayscale for edge/texture analysis
    gray = cv2.cvtColor(resized, cv2.COLOR_BGR2GRAY) if len(resized.shape) == 3 else resized
    
    # Edge histogram
    edges = cv2.Canny(gray, 50, 150)
    edge_hist = cv2.calcHist([edges], [0], None, [16], [0, 256]).flatten()
    edge_hist = edge_hist / (np.sum(edge_hist) + 1e-6)
    
    # Gabor features (multiple orientations and frequencies)
    gabor_features = []
    for theta in np.arange(0, np.pi, np.pi / 4):  # 4 orientations
        for sigma in [1, 3]:  # 2 scales
            kernel_size = int(6 * sigma + 1)
            kernel = cv2.getGaborKernel(
                (kernel_size, kernel_size),
                sigma=sigma,
                theta=theta,
                lambd=10.0,
                gamma=0.5,
                psi=0,
            )
            filtered = cv2.filter2D(gray, cv2.CV_32F, kernel)
            gabor_features.extend([
                np.mean(filtered),
                np.std(filtered),
            ])
    
    gabor_features = np.array(gabor_features)
    gabor_features = gabor_features / (np.max(np.abs(gabor_features)) + 1e-6)
    
    # Combine all features
    features = np.concatenate([color_hist, edge_hist, gabor_features])
    
    return features


def texture_delta(frame_a: np.ndarray, frame_b: np.ndarray) -> float:
    """Compute texture difference between two frames.
    
    Uses a combination of color histogram difference and edge structure change.
    
    Args:
        frame_a: First frame (H, W, 3) uint8
        frame_b: Second frame (H, W, 3) uint8
    
    Returns:
        Texture delta (0 = identical textures, 1 = completely different)
    """
    # Ensure same size
    if frame_a.shape != frame_b.shape:
        frame_b = cv2.resize(frame_b, (frame_a.shape[1], frame_a.shape[0]))
    
    # Color histogram difference
    if len(frame_a.shape) == 3:
        hist_a = cv2.calcHist([frame_a], [0, 1, 2], None, [8, 8, 8], [0, 256, 0, 256, 0, 256])
        hist_b = cv2.calcHist([frame_b], [0, 1, 2], None, [8, 8, 8], [0, 256, 0, 256, 0, 256])
    else:
        hist_a = cv2.calcHist([frame_a], [0], None, [64], [0, 256])
        hist_b = cv2.calcHist([frame_b], [0], None, [64], [0, 256])
    
    # Normalize
    hist_a = hist_a / (np.sum(hist_a) + 1e-6)
    hist_b = hist_b / (np.sum(hist_b) + 1e-6)
    
    # Chi-squared distance
    color_diff = cv2.compareHist(
        hist_a.reshape(-1, 1).astype(np.float32),
        hist_b.reshape(-1, 1).astype(np.float32),
        cv2.HISTCMP_CHISQR,
    )
    
    # Edge structure difference
    gray_a = cv2.cvtColor(frame_a, cv2.COLOR_BGR2GRAY) if len(frame_a.shape) == 3 else frame_a
    gray_b = cv2.cvtColor(frame_b, cv2.COLOR_BGR2GRAY) if len(frame_b.shape) == 3 else frame_b
    
    edges_a = cv2.Canny(gray_a, 50, 150).astype(np.float32) / 255.0
    edges_b = cv2.Canny(gray_b, 50, 150).astype(np.float32) / 255.0
    
    # Edge overlap (Jaccard-like)
    intersection = np.sum(edges_a * edges_b)
    union = np.sum(edges_a) + np.sum(edges_b) - intersection
    edge_diff = 1.0 - (intersection / (union + 1e-6))
    
    # Combine (weighted average)
    delta = 0.5 * min(color_diff, 1.0) + 0.5 * edge_diff
    
    return min(delta, 1.0)


def detect_texture_flicker(frames: list[np.ndarray], threshold: float = 0.15) -> list[int]:
    """Detect frames with texture flickering.
    
    Args:
        frames: List of frames (H, W, 3) uint8
        threshold: Delta threshold to flag as flickering
    
    Returns:
        List of frame indices where texture flickering occurs
    """
    if len(frames) < 2:
        return []
    
    flicker_frames = []
    
    for i in range(len(frames) - 1):
        delta = texture_delta(frames[i], frames[i + 1])
        if delta > threshold:
            flicker_frames.append(i + 1)
    
    return flicker_frames


def compute_texture_energy(frame: np.ndarray) -> float:
    """Compute texture energy (complexity measure).
    
    Higher energy indicates more complex/detailed texture.
    
    Args:
        frame: Input frame (H, W, 3) uint8
    
    Returns:
        Texture energy value
    """
    gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY) if len(frame.shape) == 3 else frame
    
    # Laplacian for edge detection
    laplacian = cv2.Laplacian(gray, cv2.CV_64F)
    
    # Energy is sum of squared Laplacian values
    energy = np.sum(laplacian ** 2) / (gray.shape[0] * gray.shape[1])
    
    return float(energy)