"""Tests for optical flow module."""

import pytest
import numpy as np

from detector.optical_flow import (
    compute_flow,
    warping_error,
    flow_consistency,
    flow_magnitude_stats,
    detect_flow_anomalies,
)


class TestComputeFlow:
    """Tests for optical flow computation."""
    
    def test_same_frame(self, sample_frame):
        flow = compute_flow(sample_frame, sample_frame)
        assert flow.shape == (sample_frame.shape[0], sample_frame.shape[1], 2)
        # Flow should be near zero for identical frames
        assert np.mean(np.abs(flow)) < 1.0
    
    def test_different_frames(self, sample_frames):
        flow = compute_flow(sample_frames[0], sample_frames[1])
        assert flow.shape[2] == 2
        assert flow.shape[0] == sample_frames[0].shape[0]
        assert flow.shape[1] == sample_frames[0].shape[1]
    
    def test_grayscale_frames(self):
        frame_a = np.random.randint(0, 255, (100, 100), dtype=np.uint8)
        frame_b = np.random.randint(0, 255, (100, 100), dtype=np.uint8)
        flow = compute_flow(frame_a, frame_b)
        assert flow.shape == (100, 100, 2)
    
    def test_flow_dtype(self, sample_frame):
        flow = compute_flow(sample_frame, sample_frame)
        assert flow.dtype == np.float32


class TestWarpingError:
    """Tests for warping error computation."""
    
    def test_identical_flows(self, sample_frame):
        flow = compute_flow(sample_frame, sample_frame)
        error = warping_error(flow, flow)
        assert error >= 0
    
    def test_error_range(self, sample_frames):
        flow_a = compute_flow(sample_frames[0], sample_frames[1])
        flow_b = compute_flow(sample_frames[1], sample_frames[2])
        error = warping_error(flow_a, flow_b)
        assert error >= 0


class TestFlowConsistency:
    """Tests for flow sequence consistency."""
    
    def test_single_flow(self):
        flow = np.zeros((100, 100, 2), dtype=np.float32)
        consistency = flow_consistency([flow])
        assert consistency == 1.0
    
    def test_empty_sequence(self):
        consistency = flow_consistency([])
        assert consistency == 1.0
    
    def test_consistent_flows(self):
        # Create identical flows - should be perfectly consistent
        flow = np.ones((100, 100, 2), dtype=np.float32)
        consistency = flow_consistency([flow, flow, flow])
        assert consistency == 1.0
    
    def test_inconsistent_flows(self):
        # Create very different flows
        flow_a = np.ones((100, 100, 2), dtype=np.float32)
        flow_b = -np.ones((100, 100, 2), dtype=np.float32)
        consistency = flow_consistency([flow_a, flow_b])
        assert consistency < 1.0
    
    def test_consistency_range(self, sample_frames):
        flows = [compute_flow(sample_frames[i], sample_frames[i+1]) 
                for i in range(len(sample_frames)-1)]
        consistency = flow_consistency(flows)
        assert 0 <= consistency <= 1


class TestFlowMagnitudeStats:
    """Tests for flow magnitude statistics."""
    
    def test_zero_flow(self):
        flow = np.zeros((100, 100, 2), dtype=np.float32)
        stats = flow_magnitude_stats(flow)
        assert stats["mean"] == 0.0
        assert stats["median"] == 0.0
        assert stats["std"] == 0.0
        assert stats["max"] == 0.0
    
    def test_stats_structure(self, sample_frame):
        flow = compute_flow(sample_frame, sample_frame)
        stats = flow_magnitude_stats(flow)
        assert "mean" in stats
        assert "median" in stats
        assert "std" in stats
        assert "max" in stats
        assert stats["mean"] >= 0
        assert stats["max"] >= stats["mean"]


class TestDetectFlowAnomalies:
    """Tests for anomaly detection in optical flow."""
    
    def test_normal_flow(self):
        flow = np.random.randn(100, 100, 2).astype(np.float32) * 0.1
        anomalies = detect_flow_anomalies(flow, threshold=2.0)
        assert anomalies.shape == (100, 100)
        assert anomalies.dtype == bool
    
    def test_anomalous_flow(self):
        flow = np.zeros((100, 100, 2), dtype=np.float32)
        # Add large anomaly in center
        flow[40:60, 40:60, :] = 100.0
        anomalies = detect_flow_anomalies(flow, threshold=2.0)
        assert anomalies[50, 50] == True
    
    def test_threshold_sensitivity(self):
        flow = np.random.randn(100, 100, 2).astype(np.float32)
        anomalies_low = detect_flow_anomalies(flow, threshold=0.5)
        anomalies_high = detect_flow_anomalies(flow, threshold=3.0)
        # Lower threshold should detect more anomalies
        assert np.sum(anomalies_low) >= np.sum(anomalies_high)