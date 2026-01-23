"""
Test script for the Gameplay Analysis System

This demonstrates the full flow:
1. Analyze a video with the interpreter
2. Start a reasoner session with the analysis
3. Chat with the coach
4. End session and save data
"""

from config import config
from interpreter import GameplayInterpreter, MatchMetadata, analyze_video
from reasoner import GameplayReasoner, analyze_and_discuss


def main():
    # ==========================================================================
    # CONFIGURATION (uses environment variables with fallbacks)
    # ==========================================================================

    VIDEO_PATH = config.video_path or "videos/sample_gameplay.MP4"
    MAP_NAME = config.map_name or "Nuketown"
    MODE = config.game_mode or "Domination"
    PROFILE_PATH = str(config.profile_path)
    PLAYER_ID = config.player_id
    
    # ==========================================================================
    # OPTION 1: Full Control (recommended for integration)
    # ==========================================================================
    
    print("=" * 60)
    print("STEP 1: Analyzing video with interpreter...")
    print("=" * 60)
    
    # Run the interpreter
    interpreter = GameplayInterpreter(quality=config.quality_mode, verbose=True)
    metadata = MatchMetadata(map_name=MAP_NAME, mode=MODE)
    interpreter_result = interpreter.analyze(VIDEO_PATH, metadata)
    
    print(f"\n✓ Video analyzed: {interpreter_result.duration_seconds:.1f}s")
    print(f"✓ Narration: {len(interpreter_result.merged_narration)} chars")
    print("\n" + "=" * 60)
    print("STEP 2: Starting reasoner session...")
    print("=" * 60)
    
    # Initialize reasoner
    reasoner = GameplayReasoner(
        profile_path=PROFILE_PATH,
        verbose=True
    )
    
    # Create profile if it doesn't exist
    if not reasoner.has_profile():
        print(f"Creating new player profile for '{PLAYER_ID}'...")
        reasoner.create_profile(PLAYER_ID)
    
    # Start session with interpreter output
    session = reasoner.start_session(interpreter_result)
    
    print("\n" + "-" * 60)
    print("System's INITIAL ANALYSIS:")
    print("-" * 60)
    print(session.opening_message)
    
    if session.key_observations:
        print("\n📌 Key Observations:")
        for obs in session.key_observations:
            print(f"   • {obs}")
    
    if session.comparisons_to_past:
        print("\n📊 Compared to Past:")
        for comp in session.comparisons_to_past:
            print(f"   • {comp}")
    
    print("\n" + "=" * 60)
    print("STEP 3: Chat with the coach")
    print("=" * 60)
    print("Type your questions. Type 'quit' to end.\n")
    
    # Chat loop
    while True:
        try:
            user_input = input("You: ").strip()
        except (EOFError, KeyboardInterrupt):
            print("\n")
            break
        
        if not user_input:
            continue
        if user_input.lower() in ("quit", "exit", "q"):
            break
        
        response = reasoner.chat(user_input)
        print(f"\nCoach: {response.message}\n")
        
        if response.new_recommendations:
            print("💡 New Recommendations:")
            for rec in response.new_recommendations:
                print(f"   • {rec.recommendation}")
            print()
    
    print("=" * 60)
    print("STEP 4: Ending session...")
    print("=" * 60)
    
    # End session and save
    summary = reasoner.end_session()
    
    print(f"✓ Match recorded: {summary.match_id}")
    print(f"✓ Tendencies identified: {len(summary.tendencies_identified)}")
    print(f"✓ Recommendations given: {len(summary.recommendations_given)}")
    print(f"✓ Data saved to: {PROFILE_PATH}")
    
    print("\nDone! Run again with another video to build your profile.")


def quick_test():
    """
    OPTION 2: Quick one-liner using convenience functions
    
    This is simpler but gives you less control.
    """
    
    VIDEO_PATH = "sample_gameplay.MP4"
    MAP_NAME = "Nuketown"
    MODE = "Domination"
    
    # One-liner for interpreter
    result = analyze_video(
        VIDEO_PATH,
        map_name=MAP_NAME,
        mode=MODE,
        quality="high",
        verbose=False  # Quieter output
    )
    
    # One-liner for interactive session
    analyze_and_discuss(
        interpreter_output=result,
        profile_path="player_profile.json",
        player_id="player"
    )


if __name__ == "__main__":
    # Use main() for full control, or quick_test() for simplicity
    main()
    # quick_test()