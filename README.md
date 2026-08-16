# Temporal Artifact Detector

A Python tool for detecting flickering and temporal artifacts in AI-generated videos.

## Overview

Temporal Artifact Detector analyzes video frames to identify common issues in AI-generated content:

- **Flickering**: Rapid brightness/color changes between frames
- **Identity Drift**: Facial features changing across frames
- **Texture Instability**: Surface patterns shifting or warping
- **Temporal Inconsistencies**: Optical flow violations and warping errors

## Installation

```bash
# Basic installation
pip install temporal-artifact-detector

# With ML dependencies (PyTorch, ArcFace, LPIPS)
pip install temporal-artifact-detector[ml]

# Development installation
pip install -e ".[dev]"
```

## Quick Start

```python
from detector import analyze_video

# Analyze a video file
report = analyze_video("generated_video.mp4")

# Print summary
print(report.to_markdown())

# Get JSON output
print(report.to_json())
```

## CLI Usage

```bash
# Analyze a video
tad analyze video.mp4

# Output as JSON
tad analyze video.mp4 --output json

# Compare two videos
tad compare video_a.mp4 video_b.mp4
```

## API Reference

### Core Analysis

```python
from detector import analyze_video, analyze_frames

# From video file
report = analyze_video("video.mp4")

# From frame list (numpy arrays)
report = analyze_frames(frame_list)
```

### Individual Detectors

```python
from detector.optical_flow import compute_flow, flow_consistency
from detector.identity import track_identity
from detector.texture import detect_texture_flicker
from detector.metrics import artifact_score

# Optical flow analysis
flow = compute_flow(frame_a, frame_b)
consistency = flow_consistency(flow_sequence)

# Identity tracking
embeddings = track_identity(frames)

# Texture flicker detection
flicker_frames = detect_texture_flicker(frames, threshold=0.15)

# Overall artifact score (0-1, lower is better)
score = artifact_score(analysis_report)
```

## Configuration

### Mock vs Real ML Dependencies

By default, the tool uses mock implementations for heavy ML dependencies (ArcFace, LPIPS) that provide reasonable approximations. To use real implementations:

```bash
pip install temporal-artifact-detector[ml]
```

The tool will automatically detect and use the real implementations when available.

### Custom Thresholds

```python
from detector import analyze_frames

report = analyze_frames(
    frames,
    flicker_threshold=0.1,      # Brightness change threshold
    drift_threshold=0.8,        # Cosine similarity threshold
    texture_threshold=0.15,     # Texture delta threshold
)
```

## Architecture

```
detector/
├── __init__.py        # Public API exports
├── analyzer.py        # Main analysis pipeline
├── optical_flow.py    # Optical flow computation
├── identity.py        # Identity embedding & tracking
├── texture.py         # Texture feature analysis
├── metrics.py         # LPIPS, PSNR, SSIM metrics
├── report.py          # Report dataclasses
└── cli.py             # Command-line interface
```

## License

Apache License 2.0 - see [LICENSE](LICENSE) for details.