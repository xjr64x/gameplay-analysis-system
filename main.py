#!/usr/bin/env python3
"""
Gameplay Analysis System - Unified CLI

Single entrypoint for all operations:
    - analyze: Full pipeline (interpreter → reasoner → optional chat)
    - interpret: Run interpreter only (video → narration)
    - chat: Start chat session with existing analysis
    - profile: Manage player profiles
    - check: Verify system components are working

Examples:
    # Full analysis with interactive chat
    python main.py analyze video.mp4 --map Nuketown --mode Domination

    # Batch mode (no chat, for CI/containers)
    python main.py analyze video.mp4 --map Nuketown --mode Domination --batch

    # Interpreter only
    python main.py interpret video.mp4 --map Nuketown --mode Domination

    # Check system health
    python main.py check --all
    python main.py check --ollama
    python main.py check --ffmpeg

    # Profile management
    python main.py profile show
    python main.py profile create --player-id myname
"""

from __future__ import annotations

import argparse
import json
import os
import shutil
import subprocess
import sys
from dataclasses import dataclass
from enum import Enum
from pathlib import Path
from typing import Optional, Protocol, NoReturn


# =============================================================================
# CONFIGURATION
# =============================================================================

@dataclass(frozen=True)
class AppConfig:
    """Immutable application configuration."""
    
    # Ollama connection
    ollama_host: str
    interpreter_model: str
    reasoner_model: str
    
    # Processing
    quality_mode: str
    target_fps: int
    overlap_frames: int
    
    # Paths
    output_dir: Path
    profile_path: Path
    
    # Runtime
    verbose: bool
    
    @classmethod
    def from_args(cls, args: argparse.Namespace) -> "AppConfig":
        """Build config from parsed CLI arguments with env var fallbacks."""
        return cls(
            ollama_host=args.ollama_host or os.getenv(
                "OLLAMA_HOST", _default_ollama_host()
            ),
            interpreter_model=args.interpreter_model or os.getenv(
                "INTERPRETER_MODEL", "qwen3-vl:8b-instruct-q4_K_M"
            ),
            reasoner_model=args.reasoner_model or os.getenv(
                "REASONER_MODEL", "qwen3:14b-q4_K_M"
            ),
            quality_mode=args.quality if hasattr(args, "quality") and args.quality else os.getenv(
                "QUALITY_MODE", "high"
            ),
            target_fps=args.fps if hasattr(args, "fps") and args.fps else 1,
            overlap_frames=args.overlap if hasattr(args, "overlap") and args.overlap else 5,
            output_dir=Path(args.output_dir if hasattr(args, "output_dir") and args.output_dir else os.getenv(
                "OUTPUT_DIR", "."
            )),
            profile_path=Path(args.profile if hasattr(args, "profile") and args.profile else os.getenv(
                "PROFILE_PATH", "player_profile.json"
            )),
            verbose=not (hasattr(args, "quiet") and args.quiet),
        )


def _default_ollama_host() -> str:
    """Determine default Ollama host based on environment."""
    in_container = (
        os.path.exists("/.dockerenv") or
        os.path.exists("/run/.containerenv") or
        os.getenv("DOCKER_CONTAINER") == "true"
    )
    return "http://ollama:11434" if in_container else "http://127.0.0.1:11434"


# =============================================================================
# OUTPUT HANDLING
# =============================================================================

class OutputLevel(Enum):
    """Output verbosity levels."""
    SILENT = 0
    ERROR = 1
    WARN = 2
    INFO = 3
    DEBUG = 4


class Console:
    """Centralized console output with levels and formatting."""
    
    def __init__(self, verbose: bool = True):
        self.level = OutputLevel.DEBUG if verbose else OutputLevel.INFO
    
    def _print(self, level: OutputLevel, prefix: str, msg: str, **kwargs) -> None:
        if level.value <= self.level.value:
            print(f"{prefix} {msg}", **kwargs)
    
    def info(self, msg: str) -> None:
        self._print(OutputLevel.INFO, "•", msg)
    
    def success(self, msg: str) -> None:
        self._print(OutputLevel.INFO, "✓", msg)
    
    def warn(self, msg: str) -> None:
        self._print(OutputLevel.WARN, "⚠", msg)
    
    def error(self, msg: str) -> None:
        self._print(OutputLevel.ERROR, "✗", msg, file=sys.stderr)
    
    def debug(self, msg: str) -> None:
        self._print(OutputLevel.DEBUG, "  ", msg)
    
    def header(self, msg: str) -> None:
        if self.level.value >= OutputLevel.INFO.value:
            print(f"\n{'=' * 60}")
            print(msg)
            print(f"{'=' * 60}")
    
    def subheader(self, msg: str) -> None:
        if self.level.value >= OutputLevel.INFO.value:
            print(f"\n{'-' * 60}")
            print(msg)
            print(f"{'-' * 60}")


# =============================================================================
# SYSTEM CHECKS
# =============================================================================

@dataclass
class CheckResult:
    """Result of a system check."""
    name: str
    passed: bool
    message: str
    details: Optional[str] = None


class SystemChecker:
    """Verify system components are functional."""
    
    def __init__(self, config: AppConfig, console: Console):
        self.config = config
        self.console = console
    
    def check_ffmpeg(self) -> CheckResult:
        """Check if ffmpeg/ffprobe are available."""
        try:
            result = subprocess.run(
                ["ffmpeg", "-version"],
                capture_output=True,
                text=True,
                timeout=5,
            )
            if result.returncode == 0:
                version_line = result.stdout.split("\n")[0]
                return CheckResult("ffmpeg", True, "Available", version_line)
            return CheckResult("ffmpeg", False, "ffmpeg returned error")
        except FileNotFoundError:
            return CheckResult("ffmpeg", False, "ffmpeg not found in PATH")
        except subprocess.TimeoutExpired:
            return CheckResult("ffmpeg", False, "ffmpeg timed out")
    
    def check_ffprobe(self) -> CheckResult:
        """Check if ffprobe is available."""
        try:
            result = subprocess.run(
                ["ffprobe", "-version"],
                capture_output=True,
                text=True,
                timeout=5,
            )
            if result.returncode == 0:
                version_line = result.stdout.split("\n")[0]
                return CheckResult("ffprobe", True, "Available", version_line)
            return CheckResult("ffprobe", False, "ffprobe returned error")
        except FileNotFoundError:
            return CheckResult("ffprobe", False, "ffprobe not found in PATH")
        except subprocess.TimeoutExpired:
            return CheckResult("ffprobe", False, "ffprobe timed out")
    
    def check_ollama_connection(self) -> CheckResult:
        """Check if Ollama server is reachable."""
        import urllib.request
        import urllib.error
        
        url = f"{self.config.ollama_host}/api/version"
        try:
            with urllib.request.urlopen(url, timeout=5) as response:
                data = json.loads(response.read().decode())
                version = data.get("version", "unknown")
                return CheckResult(
                    "ollama",
                    True,
                    f"Connected to {self.config.ollama_host}",
                    f"Version: {version}",
                )
        except urllib.error.URLError as e:
            return CheckResult(
                "ollama",
                False,
                f"Cannot connect to {self.config.ollama_host}",
                str(e.reason),
            )
        except Exception as e:
            return CheckResult("ollama", False, "Connection failed", str(e))
    
    def check_ollama_model(self, model_name: str) -> CheckResult:
        """Check if a specific model is available in Ollama."""
        import urllib.request
        import urllib.error
        
        url = f"{self.config.ollama_host}/api/tags"
        try:
            with urllib.request.urlopen(url, timeout=10) as response:
                data = json.loads(response.read().decode())
                models = [m.get("name", "") for m in data.get("models", [])]
                
                # Check for exact match or prefix match (model:tag format)
                found = any(
                    m == model_name or m.startswith(f"{model_name}:")
                    for m in models
                )
                
                if found:
                    return CheckResult(
                        f"model:{model_name}",
                        True,
                        "Model available",
                    )
                return CheckResult(
                    f"model:{model_name}",
                    False,
                    "Model not found",
                    f"Available: {', '.join(models[:5])}{'...' if len(models) > 5 else ''}",
                )
        except Exception as e:
            return CheckResult(f"model:{model_name}", False, "Check failed", str(e))
    
    def check_python_deps(self) -> CheckResult:
        """Check if required Python packages are available."""
        missing = []
        for pkg in ["cv2", "PIL", "numpy", "ollama"]:
            try:
                __import__(pkg)
            except ImportError:
                # PIL is imported as PIL but package is pillow
                actual = "pillow" if pkg == "PIL" else pkg
                actual = "opencv-python" if pkg == "cv2" else actual
                missing.append(actual)
        
        if missing:
            return CheckResult(
                "python-deps",
                False,
                f"Missing packages: {', '.join(missing)}",
            )
        return CheckResult("python-deps", True, "All packages available")
    
    def check_profile_path(self) -> CheckResult:
        """Check if profile path is writable."""
        profile_dir = self.config.profile_path.parent
        try:
            profile_dir.mkdir(parents=True, exist_ok=True)
            test_file = profile_dir / ".write_test"
            test_file.write_text("test")
            test_file.unlink()
            return CheckResult(
                "profile-path",
                True,
                f"Writable: {profile_dir}",
            )
        except Exception as e:
            return CheckResult(
                "profile-path",
                False,
                f"Cannot write to {profile_dir}",
                str(e),
            )
    
    def check_output_dir(self) -> CheckResult:
        """Check if output directory is writable."""
        try:
            self.config.output_dir.mkdir(parents=True, exist_ok=True)
            test_file = self.config.output_dir / ".write_test"
            test_file.write_text("test")
            test_file.unlink()
            return CheckResult(
                "output-dir",
                True,
                f"Writable: {self.config.output_dir}",
            )
        except Exception as e:
            return CheckResult(
                "output-dir",
                False,
                f"Cannot write to {self.config.output_dir}",
                str(e),
            )
    
    def run_all(self) -> list[CheckResult]:
        """Run all system checks."""
        results = [
            self.check_ffmpeg(),
            self.check_ffprobe(),
            self.check_python_deps(),
            self.check_ollama_connection(),
            self.check_ollama_model(self.config.interpreter_model),
            self.check_ollama_model(self.config.reasoner_model),
            self.check_profile_path(),
            self.check_output_dir(),
        ]
        return results
    
    def print_results(self, results: list[CheckResult]) -> bool:
        """Print check results and return True if all passed."""
        all_passed = True
        for r in results:
            if r.passed:
                self.console.success(f"{r.name}: {r.message}")
            else:
                self.console.error(f"{r.name}: {r.message}")
                all_passed = False
            if r.details:
                self.console.debug(f"  {r.details}")
        return all_passed


# =============================================================================
# COMMAND HANDLERS
# =============================================================================

def cmd_check(args: argparse.Namespace) -> int:
    """Handle 'check' command - verify system components."""
    config = AppConfig.from_args(args)
    console = Console(verbose=True)
    checker = SystemChecker(config, console)
    
    console.header("System Health Check")
    
    results = []
    
    if args.all or args.ffmpeg:
        results.append(checker.check_ffmpeg())
        results.append(checker.check_ffprobe())
    
    if args.all or args.deps:
        results.append(checker.check_python_deps())
    
    if args.all or args.ollama:
        results.append(checker.check_ollama_connection())
        if results[-1].passed:  # Only check models if connected
            results.append(checker.check_ollama_model(config.interpreter_model))
            results.append(checker.check_ollama_model(config.reasoner_model))
    
    if args.all or args.paths:
        results.append(checker.check_profile_path())
        results.append(checker.check_output_dir())
    
    if not results:
        # No specific checks requested, run all
        results = checker.run_all()
    
    print()
    all_passed = checker.print_results(results)
    
    print()
    if all_passed:
        console.success("All checks passed!")
        return 0
    else:
        console.error("Some checks failed")
        return 1


def cmd_analyze(args: argparse.Namespace) -> int:
    """Handle 'analyze' command - full pipeline."""
    # Import here to avoid loading heavy deps for simple commands
    from interpreter import GameplayInterpreter, MatchMetadata
    from reasoner import GameplayReasoner
    
    config = AppConfig.from_args(args)
    console = Console(verbose=config.verbose)
    
    # Validate inputs
    video_path = Path(args.video)
    if not video_path.exists():
        console.error(f"Video not found: {video_path}")
        return 1
    
    if not args.map:
        console.error("--map is required")
        return 1
    
    if not args.mode:
        console.error("--mode is required")
        return 1
    
    console.header("Gameplay Analysis System")
    console.info(f"Video: {video_path}")
    console.info(f"Map: {args.map} | Mode: {args.mode}")
    console.info(f"Quality: {config.quality_mode}")
    console.info(f"Profile: {config.profile_path}")
    if args.batch:
        console.info("Mode: Batch (non-interactive)")
    
    # Set OLLAMA_HOST for the modules that read it
    os.environ["OLLAMA_HOST"] = config.ollama_host
    
    # Step 1: Run interpreter
    console.subheader("Step 1/3: Video Analysis")
    
    try:
        interpreter = GameplayInterpreter(
            quality=config.quality_mode,
            model=config.interpreter_model,
            target_fps=config.target_fps,
            overlap_frames=config.overlap_frames,
            verbose=config.verbose,
        )
        metadata = MatchMetadata(map_name=args.map, mode=args.mode)
        interpreter_result = interpreter.analyze(str(video_path), metadata)
    except Exception as e:
        console.error(f"Interpreter failed: {e}")
        return 1
    
    console.success(f"Video analyzed: {interpreter_result.duration_seconds:.1f}s")
    console.info(f"Narration: {len(interpreter_result.merged_narration)} chars")
    
    # Save interpreter output
    analysis_path = config.output_dir / f"{video_path.stem}_analysis.json"
    config.output_dir.mkdir(parents=True, exist_ok=True)
    interpreter_result.save(str(analysis_path))
    console.success(f"Analysis saved: {analysis_path}")
    
    # Step 2: Run reasoner
    console.subheader("Step 2/3: Performance Analysis")
    
    try:
        reasoner = GameplayReasoner(
            profile_path=str(config.profile_path),
            model=config.reasoner_model,
            verbose=config.verbose,
        )
        
        # Create profile if needed
        if not reasoner.has_profile():
            player_id = args.player_id or os.getenv("PLAYER_ID", "player")
            console.info(f"Creating new profile: {player_id}")
            reasoner.create_profile(player_id)
        
        session = reasoner.start_session(interpreter_result)
    except Exception as e:
        console.error(f"Reasoner failed: {e}")
        return 1
    
    # Print initial analysis
    console.subheader("Analysis")
    print(session.opening_message)
    
    if session.key_observations:
        print("\n📌 Key Observations:")
        for obs in session.key_observations:
            print(f"   • {obs}")
    
    if session.comparisons_to_past:
        print("\n📊 Compared to Past:")
        for comp in session.comparisons_to_past:
            print(f"   • {comp}")
    
    # Step 3: Chat loop (unless batch mode)
    if not args.batch:
        console.subheader("Step 3/3: Discussion")
        print("Ask questions about your gameplay. Type 'quit' to end.\n")
        
        while True:
            try:
                user_input = input("You: ").strip()
            except (EOFError, KeyboardInterrupt):
                print()
                break
            
            if not user_input:
                continue
            if user_input.lower() in ("quit", "exit", "q"):
                break
            
            try:
                response = reasoner.chat(user_input)
                print(f"\nCoach: {response.message}\n")
                
                if response.new_recommendations:
                    print("💡 New Recommendations:")
                    for rec in response.new_recommendations:
                        print(f"   • {rec.recommendation}")
                    print()
            except Exception as e:
                console.error(f"Chat error: {e}")
    
    # End session
    console.subheader("Session Complete")
    
    try:
        summary = reasoner.end_session()
        console.success(f"Match recorded: {summary.match_id}")
        console.info(f"Tendencies identified: {len(summary.tendencies_identified)}")
        console.info(f"Recommendations given: {len(summary.recommendations_given)}")
        console.success(f"Profile saved: {config.profile_path}")
    except Exception as e:
        console.error(f"Failed to save session: {e}")
        return 1
    
    return 0


def cmd_interpret(args: argparse.Namespace) -> int:
    """Handle 'interpret' command - interpreter only, no reasoner."""
    from interpreter import GameplayInterpreter, MatchMetadata
    
    config = AppConfig.from_args(args)
    console = Console(verbose=config.verbose)
    
    video_path = Path(args.video)
    if not video_path.exists():
        console.error(f"Video not found: {video_path}")
        return 1
    
    if not args.map or not args.mode:
        console.error("--map and --mode are required")
        return 1
    
    console.header("Gameplay Interpreter")
    console.info(f"Video: {video_path}")
    console.info(f"Map: {args.map} | Mode: {args.mode}")
    
    os.environ["OLLAMA_HOST"] = config.ollama_host
    
    try:
        interpreter = GameplayInterpreter(
            quality=config.quality_mode,
            model=config.interpreter_model,
            target_fps=config.target_fps,
            overlap_frames=config.overlap_frames,
            verbose=config.verbose,
        )
        metadata = MatchMetadata(map_name=args.map, mode=args.mode)
        result = interpreter.analyze(str(video_path), metadata)
    except Exception as e:
        console.error(f"Interpreter failed: {e}")
        return 1
    
    # Save output
    output_path = config.output_dir / f"{video_path.stem}_analysis.json"
    config.output_dir.mkdir(parents=True, exist_ok=True)
    result.save(str(output_path))
    
    console.success(f"Analysis saved: {output_path}")
    
    # Print narration if requested
    if args.print_narration:
        console.subheader("Narration")
        print(result.merged_narration or result.narration)
    
    return 0 if not result.failed_batch_indices else 1


def cmd_profile(args: argparse.Namespace) -> int:
    """Handle 'profile' command - profile management."""
    from reasoner import GameplayReasoner
    
    config = AppConfig.from_args(args)
    console = Console(verbose=True)
    
    if args.profile_command == "show":
        if not config.profile_path.exists():
            console.error(f"Profile not found: {config.profile_path}")
            return 1
        
        reasoner = GameplayReasoner(
            profile_path=str(config.profile_path),
            verbose=False,
        )
        summary = reasoner.get_profile_summary()
        
        console.header(f"Player Profile: {summary['player_id']}")
        console.info(f"Total matches: {summary['total_matches']}")
        console.info(f"Strengths identified: {summary['strengths']}")
        console.info(f"Weaknesses identified: {summary['weaknesses']}")
        console.info(f"Pending recommendations: {summary['pending_recommendations']}")
        
        return 0
    
    elif args.profile_command == "create":
        if config.profile_path.exists() and not args.force:
            console.error(f"Profile already exists: {config.profile_path}")
            console.info("Use --force to overwrite")
            return 1
        
        player_id = args.player_id or "player"
        reasoner = GameplayReasoner(
            profile_path=str(config.profile_path),
            verbose=False,
        )
        reasoner.create_profile(player_id)
        console.success(f"Created profile: {config.profile_path}")
        return 0
    
    elif args.profile_command == "export":
        if not config.profile_path.exists():
            console.error(f"Profile not found: {config.profile_path}")
            return 1
        
        with open(config.profile_path, "r") as f:
            data = json.load(f)
        
        if args.output:
            output_path = Path(args.output)
            with open(output_path, "w") as f:
                json.dump(data, f, indent=2)
            console.success(f"Exported to: {output_path}")
        else:
            print(json.dumps(data, indent=2))
        
        return 0
    
    else:
        console.error(f"Unknown profile command: {args.profile_command}")
        return 1


# =============================================================================
# CLI SETUP
# =============================================================================

def build_parser() -> argparse.ArgumentParser:
    """Build the argument parser with all commands and options."""
    
    parser = argparse.ArgumentParser(
        prog="gameplay",
        description="Gameplay Analysis System - AI-powered video analysis and coaching",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
  %(prog)s analyze video.mp4 --map Nuketown --mode Domination
  %(prog)s analyze video.mp4 --map Raid --mode Hardpoint --batch --quality fast
  %(prog)s interpret video.mp4 --map Nuketown --mode TDM --print-narration
  %(prog)s check --all
  %(prog)s profile show
        """,
    )
    
    # Global options
    parser.add_argument(
        "--ollama-host",
        help="Ollama server URL (default: auto-detect or OLLAMA_HOST env)",
    )
    parser.add_argument(
        "--interpreter-model",
        help="Vision model for interpreter (default: INTERPRETER_MODEL env)",
    )
    parser.add_argument(
        "--reasoner-model",
        help="Reasoning model for analysis (default: REASONER_MODEL env)",
    )
    
    subparsers = parser.add_subparsers(dest="command", help="Available commands")
    
    # ---- analyze command ----
    analyze_parser = subparsers.add_parser(
        "analyze",
        help="Full analysis pipeline: interpret video → analyze → chat",
    )
    analyze_parser.add_argument("video", help="Path to gameplay video file")
    analyze_parser.add_argument("--map", "-m", required=True, help="Map name (e.g., Nuketown)")
    analyze_parser.add_argument("--mode", "-M", required=True, help="Game mode (e.g., Domination)")
    analyze_parser.add_argument(
        "--batch", "-b",
        action="store_true",
        help="Batch mode: no interactive chat, exit after analysis",
    )
    analyze_parser.add_argument(
        "--quality", "-q",
        choices=["high", "fast"],
        default="high",
        help="Processing quality (default: high)",
    )
    analyze_parser.add_argument("--fps", type=int, help="Frames per second to extract (default: 1)")
    analyze_parser.add_argument("--overlap", type=int, help="Overlap frames between batches (default: 5)")
    analyze_parser.add_argument("--profile", "-p", help="Path to player profile JSON")
    analyze_parser.add_argument("--player-id", help="Player ID for new profiles")
    analyze_parser.add_argument("--output-dir", "-o", help="Output directory for analysis files")
    analyze_parser.add_argument("--quiet", action="store_true", help="Reduce output verbosity")
    
    # ---- interpret command ----
    interpret_parser = subparsers.add_parser(
        "interpret",
        help="Run interpreter only (video → narration JSON)",
    )
    interpret_parser.add_argument("video", help="Path to gameplay video file")
    interpret_parser.add_argument("--map", "-m", required=True, help="Map name")
    interpret_parser.add_argument("--mode", "-M", required=True, help="Game mode")
    interpret_parser.add_argument(
        "--quality", "-q",
        choices=["high", "fast"],
        default="high",
        help="Processing quality (default: high)",
    )
    interpret_parser.add_argument("--fps", type=int, help="Frames per second to extract")
    interpret_parser.add_argument("--overlap", type=int, help="Overlap frames between batches")
    interpret_parser.add_argument("--output-dir", "-o", help="Output directory")
    interpret_parser.add_argument(
        "--print-narration",
        action="store_true",
        help="Print narration to stdout after saving",
    )
    interpret_parser.add_argument("--quiet", action="store_true", help="Reduce output verbosity")
    
    # ---- check command ----
    check_parser = subparsers.add_parser(
        "check",
        help="Verify system components are working",
    )
    check_parser.add_argument("--all", "-a", action="store_true", help="Run all checks")
    check_parser.add_argument("--ffmpeg", action="store_true", help="Check ffmpeg/ffprobe")
    check_parser.add_argument("--ollama", action="store_true", help="Check Ollama connection and models")
    check_parser.add_argument("--deps", action="store_true", help="Check Python dependencies")
    check_parser.add_argument("--paths", action="store_true", help="Check output/profile paths")
    
    # ---- profile command ----
    profile_parser = subparsers.add_parser(
        "profile",
        help="Manage player profiles",
    )
    profile_subparsers = profile_parser.add_subparsers(dest="profile_command")
    
    profile_show = profile_subparsers.add_parser("show", help="Show profile summary")
    profile_show.add_argument("--profile", "-p", help="Path to profile JSON")
    
    profile_create = profile_subparsers.add_parser("create", help="Create new profile")
    profile_create.add_argument("--profile", "-p", help="Path to profile JSON")
    profile_create.add_argument("--player-id", help="Player identifier")
    profile_create.add_argument("--force", "-f", action="store_true", help="Overwrite existing")
    
    profile_export = profile_subparsers.add_parser("export", help="Export profile as JSON")
    profile_export.add_argument("--profile", "-p", help="Path to profile JSON")
    profile_export.add_argument("--output", "-o", help="Output file (default: stdout)")
    
    return parser


def main() -> int:
    """Main entry point."""
    parser = build_parser()
    args = parser.parse_args()
    
    if args.command is None:
        parser.print_help()
        return 0
    
    # Dispatch to command handler
    handlers = {
        "analyze": cmd_analyze,
        "interpret": cmd_interpret,
        "check": cmd_check,
        "profile": cmd_profile,
    }
    
    handler = handlers.get(args.command)
    if handler:
        return handler(args)
    else:
        parser.print_help()
        return 1


if __name__ == "__main__":
    sys.exit(main())