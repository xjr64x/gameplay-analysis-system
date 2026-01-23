"""
Batch (non-interactive) mode for containerized execution.

Runs the full pipeline without requiring stdin input.
Configuration is read from environment variables.
"""

import sys
from pathlib import Path

from config import config
from interpreter import GameplayInterpreter, MatchMetadata
from reasoner import GameplayReasoner


def main() -> int:
    """Run batch analysis without interactive prompts."""

    # Validate required environment variables
    if not config.video_path:
        print("ERROR: VIDEO_PATH environment variable is required for batch mode")
        return 1
    if not config.map_name:
        print("ERROR: MAP_NAME environment variable is required for batch mode")
        return 1
    if not config.game_mode:
        print("ERROR: GAME_MODE environment variable is required for batch mode")
        return 1

    video_path = Path(config.video_path)
    if not video_path.exists():
        print(f"ERROR: Video file not found: {video_path}")
        return 1

    print("=" * 60)
    print("BATCH MODE - Gameplay Analysis System")
    print("=" * 60)
    print(f"Video: {config.video_path}")
    print(f"Map: {config.map_name}")
    print(f"Mode: {config.game_mode}")
    print(f"Quality: {config.quality_mode}")
    print(f"Ollama Host: {config.ollama_host}")
    print(f"Player ID: {config.player_id}")
    print("=" * 60)

    # Step 1: Run interpreter
    print("\n[1/3] Running interpreter...")
    interpreter = GameplayInterpreter(
        quality=config.quality_mode,
        model=config.interpreter_model,
        verbose=True,
    )
    metadata = MatchMetadata(map_name=config.map_name, mode=config.game_mode)

    try:
        interpreter_result = interpreter.analyze(str(video_path), metadata)
    except Exception as e:
        print(f"ERROR: Interpreter failed: {e}")
        return 1

    print(f"\nVideo analyzed: {interpreter_result.duration_seconds:.1f}s")
    print(f"Narration length: {len(interpreter_result.merged_narration)} chars")

    # Save interpreter output
    output_path = config.output_dir / f"{video_path.stem}_analysis.json"
    output_path.parent.mkdir(parents=True, exist_ok=True)
    interpreter_result.save(str(output_path))
    print(f"Interpreter output saved to: {output_path}")

    # Step 2: Run reasoner (initial analysis only, no chat)
    print("\n[2/3] Running reasoner...")
    reasoner = GameplayReasoner(
        profile_path=str(config.profile_path),
        model=config.reasoner_model,
        verbose=True,
    )

    # Create profile if it doesn't exist
    if not reasoner.has_profile():
        print(f"Creating new player profile for '{config.player_id}'...")
        reasoner.create_profile(config.player_id)

    try:
        session = reasoner.start_session(interpreter_result)
    except Exception as e:
        print(f"ERROR: Reasoner failed: {e}")
        return 1

    # Print analysis
    print("\n" + "-" * 60)
    print("ANALYSIS:")
    print("-" * 60)
    print(session.opening_message)

    if session.key_observations:
        print("\nKey Observations:")
        for obs in session.key_observations:
            print(f"  - {obs}")

    if session.comparisons_to_past:
        print("\nCompared to Past:")
        for comp in session.comparisons_to_past:
            print(f"  - {comp}")

    # Step 3: End session and save
    print("\n[3/3] Saving session...")
    summary = reasoner.end_session()

    print("\n" + "=" * 60)
    print("BATCH ANALYSIS COMPLETE")
    print("=" * 60)
    print(f"Match ID: {summary.match_id}")
    print(f"Tendencies identified: {len(summary.tendencies_identified)}")
    print(f"Recommendations given: {len(summary.recommendations_given)}")
    print(f"Profile saved to: {config.profile_path}")
    print(f"Analysis saved to: {output_path}")

    return 0


if __name__ == "__main__":
    sys.exit(main())
