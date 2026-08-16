"""Tests for the main analyzer module."""

import pytest
import numpy as np

from detector.analyzer import (
    analyze_video,
    analyze_frames,
    detect_flickering,
    detect_identity_drift,
    _compute_brightness,
)
from detector.report import AnalysisReport, FlickerEvent, DriftReport


class TestComputeBrightness:
    """Tests for brightness computation."""
    
    def test_bgr_frame(self, sample_frame):
        brightness = _compute_brightness(sample_frame)
        assert 0 <= brightness <= 255
    
    def test_grayscale_frame(self):
        frame = np.full((100, 100), 128, dtype=np.uint8)
        brightness = _compute_brightness(frame)
        assert brightness == 128.0
    
    def test_white_frame(self):
        frame = np.full((100, 100, 3), 255, dtype=np.uint8)
        brightness = _compute_brightness(frame)
        assert brightness == 255.0
    
    def test_black_frame(self):
        frame = np.zeros((100, 100, 3), dtype=np.uint8)
        brightness = _compute_brightness(frame)
        assert brightness == 0.0


class TestDetectFlickering:
    """Tests for flickering detection."""
    
    def test_no_flicker(self, stable_frames):
        events = detect_flickering(stable_frames, brightness_threshold=0.1)
        assert len(events) == 0
    
    def test_detects_flicker(self, flickering_frames):
        events = detect_flickering(flickering_frames, brightness_threshold=0.1)
        assert len(events) > 0
    
    def test_empty_frames(self):
        events = detect_flickering([])
        assert events == []
    
    def test_single_frame(self, sample_frame):
        events = detect_flickering([sample_frame])
        assert events == []
    
    def test_severity_levels(self, flickering_frames):
        events = detect_flickering(flickering_frames, brightness_threshold=0.01)
        severities = [e.severity for e in events]
        assert any(s in ["low", "medium", "high"] for s in severities)
    
    def test_event_structure(self, flickering_frames):
        events = detect_flickering(flickering_frames, brightness_threshold=0.01)
        for event in events:
            assert isinstance(event, FlickerEvent)
            assert isinstance(event.frame_index, int)
            assert isinstance(event.brightness_change, float)
            assert event.brightness_change > 0


class TestDetectIdentityDrift:
    """Tests for identity drift detection."""
    
    def test_no_drift(self, stable_frames):
        report = detect_identity_drift(stable_frames)
        assert isinstance(report, DriftReport)
        assert report.max_drift >= 0
        assert report.max_drift <= 1
    
    def test_with_drift(self, drift_frames):
        report = detect_identity_drift(drift_frames, drift_threshold=0.9)
        assert isinstance(report, DriftReport)
        assert len(report.similarities) == len(drift_frames)
    
    def test_empty_frames(self):
        report = detect_identity_drift([])
        assert report.similarities == []
        assert report.max_drift == 0.0
    
    def test_reference_frame(self, sample_frames):
        report = detect_identity_drift(sample_frames, reference_frame=5)
        assert report.reference_frame == 5
    
    def test_similarities_range(self, sample_frames):
        report = detect_identity_drift(sample_frames)
        for sim in report.similarities:
            assert -1 <= sim <= 1


class TestAnalyzeFrames:
    """Tests for frame sequence analysis."""
    
    def test_returns_report(self, sample_frames):
        report = analyze_frames(sample_frames)
        assert isinstance(report, AnalysisReport)
        assert report.frame_count == len(sample_frames)
    
    def test_empty_frames(self):
        report = analyze_frames([])
        assert report.frame_count == 0
        assert report.artifact_score == 0.0
    
    def test_artifact_score_range(self, sample_frames):
        report = analyze_frames(sample_frames)
        assert 0 <= report.artifact_score <= 1
    
    def test_flicker_in_report(self, flickering_frames):
        report = analyze_frames(flickering_frames, flicker_threshold=0.01)
        assert len(report.flicker_events) > 0
    
    def test_custom_thresholds(self, sample_frames):
        report = analyze_frames(
            sample_frames,
            flicker_threshold=0.5,
            drift_threshold=0.5,
            texture_threshold=0.5,
        )
        assert isinstance(report, AnalysisReport)


class TestAnalyzeVideo:
    """Tests for video file analysis."""
    
    def test_invalid_video(self):
        with pytest.raises(ValueError):
            analyze_video("nonexistent_video.mp4")