"""Main analysis pipeline for temporal artifact detection."""

import cv2
import numpy as np
from typing import Optional

from detector.report import AnalysisReport, FlickerEvent, DriftReport, TextureIssue
from detector.optical_flow import compute_flow, flow_consistency
from detector.identity import track_identity
from detector.texture import detect_texture_flicker, texture_delta
from detector.metrics import artifact_score


def _compute_brightness(frame: np.ndarray) -> float:
    """Compute average brightness of a frame."""
    if len(frame.shape) == 3:
        gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
    else:
        gray = frame
    return float(np.mean(gray))


def detect_flickering(
    frames: list[np.ndarray],
    brightness_threshold: float = 0.1,
) -> list[FlickerEvent]:
    """Detect flickering events between consecutive frames.
    
    Flickering is detected when brightness changes significantly between frames.
    
    Args:
        frames: List of video frames (H, W, 3) uint8
        brightness_threshold: Relative brightness change to flag (0-1)
    
    Returns:
        List of FlickerEvent objects
    """
    if len(frames) < 2:
        return []
    
    events = []
    
    # Compute brightness for each frame
    brightness_values = [_compute_brightness(f) for f in frames]
    
    for i in range(1, len(frames)):
        prev_bright = brightness_values[i - 1]
        curr_bright = brightness_values[i]
        
        if prev_bright > 0:
            change = abs(curr_bright - prev_bright) / prev_bright
        else:
            change = abs(curr_bright - prev_bright)
        
        if change > brightness_threshold:
            # Determine severity
            if change > brightness_threshold * 3:
                severity = "high"
            elif change > brightness_threshold * 1.5:
                severity = "medium"
            else:
                severity = "low"
            
            events.append(FlickerEvent(
                frame_index=i,
                brightness_change=change,
                severity=severity,
            ))
    
    return events


def detect_identity_drift(
    frames: list[np.ndarray],
    reference_frame: Optional[int] = None,
    drift_threshold: float = 0.8,
) -> DriftReport:
    """Detect identity drift across frames.
    
    Args:
        frames: List of video frames (H, W, 3) uint8
        reference_frame: Index of reference frame (default: first frame)
        drift_threshold: Similarity threshold below which drift is flagged
    
    Returns:
        DriftReport with similarity scores and drift analysis
    """
    if not frames:
        return DriftReport(
            reference_frame=0,
            similarities=[],
            max_drift=0.0,
            drift_frames=[],
        )
    
    ref_idx = reference_frame if reference_frame is not None else 0
    
    # Track identity
    similarities = track_identity(frames)
    
    # Find drift frames (below threshold)
    drift_frames = []
    max_drift = 0.0
    
    for i, sim in enumerate(similarities):
        drift = 1.0 - sim
        if drift > max_drift:
            max_drift = drift
        if sim < drift_threshold:
            drift_frames.append(i)
    
    return DriftReport(
        reference_frame=ref_idx,
        similarities=similarities,
        max_drift=max_drift,
        drift_frames=drift_frames,
    )


def analyze_frames(
    frames: list[np.ndarray],
    flicker_threshold: float = 0.1,
    drift_threshold: float = 0.8,
    texture_threshold: float = 0.15,
    reference_frame: Optional[int] = None,
) -> AnalysisReport:
    """Analyze a sequence of frames for temporal artifacts.
    
    Args:
        frames: List of video frames (H, W, 3) uint8
        flicker_threshold: Brightness change threshold for flickering
        drift_threshold: Similarity threshold for identity drift
        texture_threshold: Delta threshold for texture instability
        reference_frame: Reference frame index for drift analysis
    
    Returns:
        AnalysisReport with all detected artifacts
    """
    if not frames:
        return AnalysisReport(frame_count=0, artifact_score=0.0)
    
    # Detect flickering
    flicker_events = detect_flickering(frames, brightness_threshold=flicker_threshold)
    
    # Detect identity drift
    drift_report = detect_identity_drift(
        frames,
        reference_frame=reference_frame,
        drift_threshold=drift_threshold,
    )
    
    # Detect texture flicker
    flicker_frame_indices = detect_texture_flicker(frames, threshold=texture_threshold)
    
    # Convert to TextureIssue objects
    texture_issues = []
    for idx in flicker_frame_indices:
        delta = texture_delta(frames[idx - 1], frames[idx])
        texture_issues.append(TextureIssue(
            frame_index=idx,
            delta=delta,
            feature_type="texture",
        ))
    
    # Create report
    report = AnalysisReport(
        flicker_events=flicker_events,
        drift_report=drift_report,
        texture_issues=texture_issues,
        frame_count=len(frames),
    )
    
    # Compute artifact score
    report.artifact_score = artifact_score(report)
    
    return report


def analyze_video(
    video_path: str,
    max_frames: Optional[int] = None,
    sample_rate: int = 1,
    flicker_threshold: float = 0.1,
    drift_threshold: float = 0.8,
    texture_threshold: float = 0.15,
    reference_frame: Optional[int] = None,
) -> AnalysisReport:
    """Analyze a video file for temporal artifacts.
    
    Args:
        video_path: Path to video file
        max_frames: Maximum number of frames to analyze (None = all)
        sample_rate: Analyze every Nth frame (1 = every frame)
        flicker_threshold: Brightness change threshold for flickering
        drift_threshold: Similarity threshold for identity drift
        texture_threshold: Delta threshold for texture instability
        reference_frame: Reference frame index for drift analysis
    
    Returns:
        AnalysisReport with all detected artifacts
    """
    # Open video
    cap = cv2.VideoCapture(video_path)
    if not cap.isOpened():
        raise ValueError(f"Could not open video: {video_path}")
    
    try:
        frames = []
        frame_idx = 0
        
        while True:
            ret, frame = cap.read()
            if not ret:
                break
            
            # Sample frames
            if frame_idx % sample_rate == 0:
                frames.append(frame)
            
            frame_idx += 1
            
            # Check max frames
            if max_frames and len(frames) >= max_frames:
                break
    finally:
        cap.release()
    
    return analyze_frames(
        frames,
        flicker_threshold=flicker_threshold,
        drift_threshold=drift_threshold,
        texture_threshold=texture_threshold,
        reference_frame=reference_frame,
    )