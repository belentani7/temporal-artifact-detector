"""Report dataclasses for temporal artifact analysis."""

from dataclasses import dataclass, field
from typing import List, Optional
import json


@dataclass
class FlickerEvent:
    """Represents a detected flickering event between frames."""
    
    frame_index: int
    brightness_change: float
    severity: str  # "low", "medium", "high"
    
    def to_dict(self) -> dict:
        return {
            "frame_index": self.frame_index,
            "brightness_change": self.brightness_change,
            "severity": self.severity,
        }


@dataclass
class DriftReport:
    """Report on identity drift across frames."""
    
    reference_frame: int
    similarities: List[float]
    max_drift: float
    drift_frames: List[int] = field(default_factory=list)
    
    def to_dict(self) -> dict:
        return {
            "reference_frame": self.reference_frame,
            "similarities": self.similarities,
            "max_drift": self.max_drift,
            "drift_frames": self.drift_frames,
        }


@dataclass
class TextureIssue:
    """Represents a detected texture instability."""
    
    frame_index: int
    delta: float
    feature_type: str  # "color", "edge", "frequency"
    
    def to_dict(self) -> dict:
        return {
            "frame_index": self.frame_index,
            "delta": self.delta,
            "feature_type": self.feature_type,
        }


@dataclass
class AnalysisReport:
    """Complete analysis report for a video or frame sequence."""
    
    flicker_events: List[FlickerEvent] = field(default_factory=list)
    drift_report: Optional[DriftReport] = None
    texture_issues: List[TextureIssue] = field(default_factory=list)
    frame_count: int = 0
    artifact_score: float = 0.0
    
    def to_dict(self) -> dict:
        return {
            "flicker_events": [e.to_dict() for e in self.flicker_events],
            "drift_report": self.drift_report.to_dict() if self.drift_report else None,
            "texture_issues": [t.to_dict() for t in self.texture_issues],
            "frame_count": self.frame_count,
            "artifact_score": self.artifact_score,
        }
    
    def to_json(self) -> str:
        """Export report as JSON string."""
        return json.dumps(self.to_dict(), indent=2)
    
    def to_markdown(self) -> str:
        """Export report as Markdown."""
        lines = ["# Temporal Artifact Analysis Report\n"]
        
        lines.append(f"**Frames Analyzed**: {self.frame_count}")
        lines.append(f"**Artifact Score**: {self.artifact_score:.3f} (0=perfect, 1=severe)\n")
        
        # Flicker events
        lines.append("## Flickering Events\n")
        if self.flicker_events:
            lines.append("| Frame | Brightness Change | Severity |")
            lines.append("|-------|-------------------|----------|")
            for event in self.flicker_events:
                lines.append(f"| {event.frame_index} | {event.brightness_change:.3f} | {event.severity} |")
        else:
            lines.append("No flickering events detected.\n")
        
        # Drift report
        lines.append("\n## Identity Drift\n")
        if self.drift_report:
            lines.append(f"- Reference frame: {self.drift_report.reference_frame}")
            lines.append(f"- Max drift: {self.drift_report.max_drift:.3f}")
            if self.drift_report.drift_frames:
                lines.append(f"- Drift frames: {', '.join(map(str, self.drift_report.drift_frames))}")
        else:
            lines.append("No drift analysis performed.\n")
        
        # Texture issues
        lines.append("\n## Texture Instability\n")
        if self.texture_issues:
            lines.append("| Frame | Delta | Feature Type |")
            lines.append("|-------|-------|--------------|")
            for issue in self.texture_issues:
                lines.append(f"| {issue.frame_index} | {issue.delta:.3f} | {issue.feature_type} |")
        else:
            lines.append("No texture issues detected.\n")
        
        return "\n".join(lines)