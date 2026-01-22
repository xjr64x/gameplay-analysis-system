# Gameplay Analysis System (Call of Duty)

An end-to-end AI system that analyzes Call of Duty gameplay videos to extract player behavior, identify tendencies, and generate evidence-grounded performance insights. The system uses vision-language models to interpret gameplay. Reasoning models reason over patterns, and track player development across matches.

**This project is designed as an AI systems pipeline, not a single model demo.**

## Key Features

- **Multi-stage AI Pipeline**: Separates perception (vision-language model) from reasoning (text model) for better explainability
- **Evidence-Grounded Analysis**: Every claim backed by video timestamps and HUD text
- **Temporal Context Preservation**: Batched processing with overlap maintains narrative continuity across long matches
- **Longitudinal Player Profiling**: Tracks tendencies, strengths, and weaknesses across multiple matches
- **Interactive Coaching Agent**: Conversational interface for asking follow-up questions about specific moments
- **Adaptive Processing**: Quality modes balance detail vs. coverage based on hardware constraints

## Overview

The system processes raw gameplay footage and produces structured, explainable analysis focused on player-controlled behavior, not team coordination. It is built to answer questions such as:

- What patterns define my playstyle on this map?
- What did I do differently compared to previous matches?
- Which tendencies help or hurt my performance?
- What concrete experiment should I try next match?
- Did I actually apply the previous advice?

The project emphasizes:

- Video understanding under real-world constraints
- Evidence-grounded reasoning
- Reproducibility
- Clear separation of perception and analysis

## High-Level Architecture
```
Gameplay Video
      ↓
Frame Extraction & Preprocessing
      ↓
Interpreter (Vision Analysis)
      ↓
Segmented, Timestamped Narration
      ↓
Reasoner (Pattern & Tendency Analysis)
      ↓
Player Profile + Recommendations
```

The system is intentionally modular:

- **Interpreter** answers: *What happened?*
- **Reasoner** answers: *What does it mean?*

## Interpreter

The interpreter is responsible for **perception**.

### What it does

- Extracts frames from gameplay video at a fixed rate
- Processes frames in overlapping batches to preserve temporal context
- Uses a vision-language model to interpret HUD elements, on-screen text, and visible events
- Produces event-driven, timestamped narration describing what occurs during the match

### Design choices

- Event-driven narration was chosen over strict schemas because smaller/quantized models tend to hallucinate missing fields.
- The interpreter only claims player actions when supported by HUD evidence.
- Objective medals are handled carefully:
  - "capturing" is only stated when the HUD explicitly shows capturing progress
  - "captured" is treated as team or objective credit unless the player is clearly on point

### Output

- Segmented narration blocks with explicit time ranges
- High recall for HUD text (kills, medals, objectives)
- Conservative language when evidence is ambiguous

## Reasoner

The reasoner is responsible for **analysis and judgment**.

### What it does

- Consumes the interpreter's full narration (or relevant time segments)
- Identifies recurring behaviors and tendencies
- Compares current performance to historical matches on the same map
- Generates actionable, player-focused recommendations
- Stores tendencies and advice in a persistent player profile

### Constraints

The reasoner is explicitly instructed to:

- Avoid motivational or "supportive companion" tone
- Avoid team coordination advice (callouts, teammate actions)
- Avoid inventing game mechanics, vehicles, or scorestreaks
- Ground all claims in narration timestamps
- Focus only on player-controlled decisions

### Output

- Pattern-based analysis
- Strengths and weaknesses with evidence
- One concrete experiment to try next match
- Comparison to past matches when available

## Workflow

1. **Run analysis on a gameplay video**  
   The interpreter analyzes the video and produces narration.

2. **Initial reasoning pass**  
   The reasoner generates a structured analysis and recommendations.

3. **Interactive follow-up (optional)**  
   Ask questions about the match ("Should I rotate more?").  
   The reasoner answers using the full narration context.

4. **Profile update**  
   Tendencies and recommendations are saved.

5. **Repeat**  
   Running additional matches builds a longitudinal player profile.

## Usage

### Requirements

- Python 3.12.10 (recommended version)
- Ollama (for vision-language and reasoning model serving)
- FFmpeg
- Models: `qwen3:14b-q4_K_M` (reasoning model), `qwen3-vl:8b-instruct-q4_K_M` (vision model)
- System: RTX 5080 with 16GB+ VRAM, or equivalent

### Setup

```bash
# Install dependencies
pip install -r requirements.txt

# Pull required Ollama models
ollama pull qwen3-vl:8b-instruct-q4_K_M
ollama pull qwen3:14b-q4_K_M

# Place your gameplay videos in the videos/ directory
```

### Running the System

```bash
# Full pipeline with interactive coaching session
python system_test.py

# Or use components directly in Python:
```

```python
from interpreter import analyze_video
from reasoner import analyze_and_discuss

# Analyze a video
result = analyze_video(
    "videos/your_match.mp4",
    map_name="Nuketown",
    mode="Domination",
    quality="high"  # or "fast"
)

# Interactive coaching session
analyze_and_discuss(result, profile_path="player_profile.json")
```

## Limitations & Assumptions

- Haven't tested on 10+ minute matches
- Assumes clear HUD visibility
- Does not attempt full semantic understanding or 3D reconstruction
- Analysis quality depends on video clarity and HUD legibility
- Vision-language models may still misinterpret ambiguous visual cues; conservative phrasing is preferred
- Assumes hardware systems have gpu's with 16GB or more of vram. 

**These limitations are intentional** to keep the system explainable and reproducible.

## Why This Project Exists

This project emerged from a practical need and a technical challenge: how do you build a system that surfaces unconscious behavioral patterns from unstructured video data?

As a player, I noticed gaps between intent and execution. Patterns I couldn't articulate. Habits I didn't know I had. Traditional post-game stats answer "what happened" but rarely explain "why" or offer actionable insight into decision-making tendencies.

This project tackles that problem through systems engineering. Rather than relying on a single model, it implements a **multi-stage pipeline** that separates perception from reasoning, maintains evidence chains, and builds longitudinal profiles. The technical challenges were significant:

- Processing video under real-world constraints (consumer hardware, quantized models)
- Maintaining temporal context across long sequences
- Grounding analysis in verifiable evidence (no hallucinated insights)
- Building a conversational agent that stays focused on individual performance
- Managing model context budgets across multi-turn interactions

The result is a system that doesn't just analyze—it teaches pattern recognition. It forces specificity. It builds a persistent understanding of playstyle over time.

Beyond the practical application, this project demonstrates proficiency in AI systems architecture: orchestrating vision-language models, reasoning models, prompt engineering, context management, and stateful agents. It represents the kind of end-to-end thinking required when AI components need to work together toward a coherent goal.

This is portfolio work that showcases both technical depth and product thinking.

## Future Plans For This Project
- Containerize it using docker
- Make customization easier with configuration options
- Potentially make the system available for broader applications and not only Call of Duty gameplay