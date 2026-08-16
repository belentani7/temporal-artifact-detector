"""Command-line interface for temporal artifact detection."""

import argparse
import sys
from pathlib import Path

from detector.analyzer import analyze_video


def cmd_analyze(args):
    """Analyze a video file for temporal artifacts."""
    try:
        report = analyze_video(
            args.video,
            max_frames=args.max_frames,
            sample_rate=args.sample_rate,
        )
        
        if args.output == "json":
            print(report.to_json())
        else:
            print(report.to_markdown())
    
    except Exception as e:
        print(f"Error: {e}", file=sys.stderr)
        sys.exit(1)


def cmd_compare(args):
    """Compare two videos for temporal artifacts."""
    try:
        report_a = analyze_video(args.video_a)
        report_b = analyze_video(args.video_b)
        
        print("# Video Comparison\n")
        print("## Video A")
        print(f"- Frames: {report_a.frame_count}")
        print(f"- Artifact Score: {report_a.artifact_score:.3f}")
        print(f"- Flicker Events: {len(report_a.flicker_events)}")
        print(f"- Texture Issues: {len(report_a.texture_issues)}")
        
        print("\n## Video B")
        print(f"- Frames: {report_b.frame_count}")
        print(f"- Artifact Score: {report_b.artifact_score:.3f}")
        print(f"- Flicker Events: {len(report_b.flicker_events)}")
        print(f"- Texture Issues: {len(report_b.texture_issues)}")
        
        print("\n## Comparison")
        if report_a.artifact_score < report_b.artifact_score:
            print("Video A has fewer temporal artifacts.")
        elif report_b.artifact_score < report_a.artifact_score:
            print("Video B has fewer temporal artifacts.")
        else:
            print("Both videos have similar artifact levels.")
    
    except Exception as e:
        print(f"Error: {e}", file=sys.stderr)
        sys.exit(1)


def main():
    """Main entry point for CLI."""
    parser = argparse.ArgumentParser(
        prog="tad",
        description="Temporal Artifact Detector - Detect flickering and artifacts in AI-generated videos",
    )
    
    subparsers = parser.add_subparsers(dest="command", help="Available commands")
    
    # Analyze command
    analyze_parser = subparsers.add_parser("analyze", help="Analyze a video for temporal artifacts")
    analyze_parser.add_argument("video", help="Path to video file")
    analyze_parser.add_argument(
        "--output", "-o",
        choices=["json", "markdown"],
        default="markdown",
        help="Output format (default: markdown)",
    )
    analyze_parser.add_argument(
        "--max-frames", "-m",
        type=int,
        default=None,
        help="Maximum number of frames to analyze",
    )
    analyze_parser.add_argument(
        "--sample-rate", "-s",
        type=int,
        default=1,
        help="Analyze every Nth frame (default: 1)",
    )
    analyze_parser.set_defaults(func=cmd_analyze)
    
    # Compare command
    compare_parser = subparsers.add_parser("compare", help="Compare two videos")
    compare_parser.add_argument("video_a", help="Path to first video file")
    compare_parser.add_argument("video_b", help="Path to second video file")
    compare_parser.set_defaults(func=cmd_compare)
    
    args = parser.parse_args()
    
    if not args.command:
        parser.print_help()
        sys.exit(1)
    
    args.func(args)


if __name__ == "__main__":
    main()