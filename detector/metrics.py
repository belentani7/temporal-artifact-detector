"""Metric computation for video frame comparison."""

import numpy as np
from skimage.metrics import structural_similarity as skimage_ssim
from skimage.metrics import peak_signal_noise_ratio as skimage_psnr

from detector.report import AnalysisReport


def _mock_lpips(frame_a: np.ndarray, frame_b: np.ndarray) -> float:
    """Mock LPIPS implementation using pixel-level differences.
    
    Real implementation would use a pre-trained VGG network.
    This provides a reasonable approximation based on color histogram differences.
    """
    if frame_a.shape != frame_b.shape:
        raise ValueError("Frames must have the same shape")
    
    # Normalize to [0, 1]
    a = frame_a.astype(np.float32) / 255.0
    b = frame_b.astype(np.float32) / 255.0
    
    # Compute per-channel mean absolute difference
    diff = np.mean(np.abs(a - b), axis=(0, 1))
    
    # Weight channels (RGB) - approximate VGG perceptual weighting
    weights = np.array([0.35, 0.45, 0.20])
    weighted_diff = np.sum(diff * weights)
    
    # Scale to approximate LPIPS range [0, 1]
    return min(weighted_diff * 3.0, 1.0)


def _get_lpips_function():
    """Get LPIPS function - real or mock based on availability."""
    try:
        import lpips
        loss_fn = lpips.LPIPS(net='vgg')
        
        def real_lpips(frame_a: np.ndarray, frame_b: np.ndarray) -> float:
            import torch
            # Convert BGR to RGB if needed
            if len(frame_a.shape) == 3 and frame_a.shape[2] == 3:
                a = frame_a[:, :, ::-1]
                b = frame_b[:, :, ::-1]
            else:
                a = frame_a
                b = frame_b
            
            # Convert to tensor [1, C, H, W]
            a_tensor = torch.from_numpy(a).permute(2, 0, 1).unsqueeze(0).float() / 127.5 - 1.0
            b_tensor = torch.from_numpy(b).permute(2, 0, 1).unsqueeze(0).float() / 127.5 - 1.0
            
            with torch.no_grad():
                distance = loss_fn(a_tensor, b_tensor)
            
            return distance.item()
        
        return real_lpips
    except ImportError:
        return _mock_lpips


_lpips_func = None


def lpips_score(frame_a: np.ndarray, frame_b: np.ndarray) -> float:
    """Compute LPIPS perceptual distance between two frames.
    
    Args:
        frame_a: First frame (H, W, 3) uint8
        frame_b: Second frame (H, W, 3) uint8
    
    Returns:
        LPIPS distance (0 = identical, 1 = maximally different)
    """
    global _lpips_func
    if _lpips_func is None:
        _lpips_func = _get_lpips_function()
    return _lpips_func(frame_a, frame_b)


def psnr(frame_a: np.ndarray, frame_b: np.ndarray) -> float:
    """Compute Peak Signal-to-Noise Ratio between two frames.
    
    Args:
        frame_a: First frame (H, W, 3) uint8
        frame_b: Second frame (H, W, 3) uint8
    
    Returns:
        PSNR value in dB (higher is better, inf for identical frames)
    """
    return skimage_psnr(frame_a.astype(np.float64), frame_b.astype(np.float64))


def ssim(frame_a: np.ndarray, frame_b: np.ndarray) -> float:
    """Compute Structural Similarity Index between two frames.
    
    Args:
        frame_a: First frame (H, W, 3) uint8
        frame_b: Second frame (H, W, 3) uint8
    
    Returns:
        SSIM value (-1 to 1, 1 = identical)
    """
    return skimage_ssim(frame_a, frame_b, channel_axis=2 if len(frame_a.shape) == 3 else None)


def artifact_score(analysis: AnalysisReport) -> float:
    """Compute overall artifact score from analysis report.
    
    Score is 0-1 where lower is better (0 = no artifacts, 1 = severe).
    
    Args:
        analysis: AnalysisReport from analysis pipeline
    
    Returns:
        Artifact score between 0 and 1
    """
    if analysis.frame_count == 0:
        return 0.0
    
    scores = []
    
    # Flicker contribution (0-0.4 weight)
    if analysis.flicker_events:
        high_severity = sum(1 for e in analysis.flicker_events if e.severity == "high")
        med_severity = sum(1 for e in analysis.flicker_events if e.severity == "medium")
        flicker_score = min((high_severity * 0.15 + med_severity * 0.08) / analysis.frame_count, 1.0)
        scores.append(("flicker", flicker_score, 0.4))
    
    # Drift contribution (0-0.3 weight)
    if analysis.drift_report:
        drift_score = 1.0 - analysis.drift_report.max_drift
        scores.append(("drift", drift_score, 0.3))
    
    # Texture contribution (0-0.3 weight)
    if analysis.texture_issues:
        texture_score = min(len(analysis.texture_issues) * 0.05 / analysis.frame_count, 1.0)
        scores.append(("texture", texture_score, 0.3))
    
    if not scores:
        return 0.0
    
    # Weighted average
    total_weight = sum(w for _, _, w in scores)
    weighted_sum = sum(s * w for _, s, w in scores)
    
    return weighted_sum / total_weight if total_weight > 0 else 0.0