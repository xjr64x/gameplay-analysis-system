# 🎮 Gameplay Analysis System (Call of Duty)

&#x20; &#x20;

> **An end-to-end AI system for extracting player behavior, tendencies, and evidence-grounded insights from raw Call of Duty gameplay footage.**

**Audience:** AI/ML engineers, systems engineers, and technically inclined players interested in explainable, evidence-based performance analysis.

This project is built as an **AI systems pipeline**, not a single-model demo. It separates *seeing* from *reasoning* to produce explainable, reproducible, and actionable insights.

---

## ⚡ TL;DR (One-Screen Overview)

- 📹 Input: Raw Call of Duty gameplay video (no APIs, no game stats)
- 👁️ Interpreter: Vision-language model produces timestamped narration
- 🧠 Reasoner: Analysis model extracts patterns and tendencies
- 📊 Output: Evidence-grounded strengths, weaknesses, and next-match experiments
- 🧩 Design goal: Explainable AI system under real-world hardware constraints

---

## ✨ What This System Does

- Analyzes **raw gameplay video** (no game APIs or stats exports)
- Interprets **HUD elements, events, and player actions** using vision-language models
- Reasons over **patterns and tendencies across time** using a dedicated analysis model
- Builds a **persistent player profile** that evolves across matches
- Provides **evidence-backed coaching**, not vague advice

This system answers questions like:

- *What patterns define my playstyle on this map?*
- *What changed compared to my last match?*
- *Which habits help me win gunfights faster?*
- *What specific experiment should I try next match?*
- *Did I actually apply the previous advice?*

---

## 🧠 Core Philosophy

- **Explainability first** — every claim is grounded in timestamps and HUD evidence
- **Separation of concerns** — perception ≠ reasoning
- **Player-centric** — focuses only on player-controlled decisions
- **Longitudinal** — insights improve as more matches are analyzed
- **Real-world constraints** — works on consumer hardware with quantized models

---

## 🏗️ High-Level Architecture

```
┌──────────────────┐
│ Gameplay Video   │
└────────┬─────────┘
         ↓
┌──────────────────┐
│ Frame Extraction │
│ & Preprocessing │
└────────┬─────────┘
         ↓
┌──────────────────────────┐
│ Interpreter (Perception) │
│ Vision-Language Model    │
└────────┬─────────────────┘
         ↓
┌──────────────────────────┐
│ Segmented, Timestamped   │
│ Narration (What Happened)│
└────────┬─────────────────┘
         ↓
┌──────────────────────────┐
│ Reasoner (Analysis)      │
│ Pattern & Tendency Model │
└────────┬─────────────────┘
         ↓
┌──────────────────────────┐
│ Player Profile +         │
│ Recommendations          │
└──────────────────────────┘
```

> **Interpreter** answers *“What happened?”*\
> **Reasoner** answers *“What does it mean?”*

---

## 👁️ Interpreter (Perception Layer)

The interpreter is responsible for **seeing and describing the match**.

### Responsibilities

- Extract frames from gameplay video at a fixed sampling rate
- Process frames in **overlapping batches** to preserve temporal continuity
- Interpret:
  - HUD text (kills, medals, objectives)
  - Visible player actions
  - Environmental interactions
- Produce **event-driven, timestamped narration**

### Key Design Choices

- Event-based narration instead of rigid schemas
  - Smaller / quantized models hallucinate less when allowed natural language
- Conservative claims policy
  - Player actions are only stated when supported by HUD or clear visuals
- Objective medal precision
  - **“capturing”** → only when HUD explicitly shows capture progress
  - **“captured”** → treated as team/objective credit unless player presence is explicit

### Output

- Segmented narration blocks with explicit time ranges
- High recall for HUD elements
- Explicit uncertainty when evidence is ambiguous

---

## 🧩 Reasoner (Analysis Layer)

The reasoner is responsible for **judgment, comparison, and insight generation**.

### Responsibilities

- Consume full narration or selected time ranges
- Identify:
  - Recurring behaviors
  - Positional habits
  - Engagement tendencies
- Compare current performance against historical matches
- Generate **actionable, player-focused recommendations**
- Update a persistent **player profile**

### Hard Constraints

The reasoner is explicitly instructed to:

- ❌ Avoid motivational or “supportive companion” tone
- ❌ Avoid team coordination or callout advice
- ❌ Avoid inventing mechanics, vehicles, or scorestreaks
- ❌ Avoid claims without timestamped evidence
- ✅ Focus only on player-controlled decisions

### Output

- Pattern-based analysis
- Strengths and weaknesses with evidence
- One concrete experiment for the next match
- Historical comparison when available

---

## 🔄 End-to-End Workflow

1. **Analyze a gameplay video**\
   Interpreter generates timestamped narration.

2. **Initial reasoning pass**\
   Reasoner produces structured analysis and recommendations.

3. **Interactive follow-up (optional)**\
   Ask questions like *“Should I rotate earlier?”* or *“Was my positioning risky here?”*

4. **Profile update**\
   Tendencies and advice are stored.

5. **Repeat**\
   Additional matches refine the longitudinal player profile.

---

## ⚙️ Usage

### Requirements

- Python **3.12.10** (recommended)
- Ollama (model serving)
- FFmpeg
- Models:
  - `qwen3-vl:8b-instruct-q4_K_M` (vision-language)
  - `qwen3:14b-q4_K_M` (reasoning)
- GPU with **16GB+ VRAM** (RTX 5080 or equivalent)

### Setup

```bash
pip install -r requirements.txt

ollama pull qwen3-vl:8b-instruct-q4_K_M
ollama pull qwen3:14b-q4_K_M

# Place gameplay videos in videos/
```

### Running the System

```bash
# Full pipeline with interactive coaching
python system_test.py
```

```python
from interpreter import analyze_video
from reasoner import analyze_and_discuss

result = analyze_video(
    "videos/your_match.mp4",
    map_name="Nuketown",
    mode="Domination",
    quality="high"  # or "fast"
)

analyze_and_discuss(result, profile_path="player_profile.json")
```

---

## ⚠️ Limitations & Assumptions (Intentional)

- Not tested extensively on 10+ minute matches
- Requires clear HUD visibility
- No full semantic world modeling or 3D reconstruction
- Vision-language models may misinterpret ambiguous visuals
- Designed for systems with 16GB+ VRAM
- **Assumes Ollama is installed and available locally** for model serving

These constraints exist to preserve **explainability and reproducibility**.

---

## 🎯 Why This Project Exists

Traditional post-game stats answer *“what happened.”*\
This system is built to answer *“why it keeps happening.”*

As a player, I noticed recurring gaps between intent and execution—habits I couldn’t articulate, positioning tendencies I didn’t consciously choose, and patterns that only became obvious when viewed across multiple matches.

This project tackles that problem through **AI systems engineering**:

- Video understanding under real-world hardware constraints
- Explicit separation of perception and reasoning
- Evidence-grounded insights (no hallucinated coaching)
- Stateful, longitudinal player modeling
- Conversational analysis constrained by hard rules

Beyond gameplay, this project demonstrates end-to-end thinking across:

- Vision-language models
- Reasoning models
- Prompt engineering
- Context management
- Modular system design

It is built as **portfolio-grade AI systems work**, not a toy demo.

---

## 🚀 Future Plans

- Docker-based containerization
- Config-driven customization
- Support for additional games and analysis domains
- Improved visualization of tendencies over time

---

## 🤝 Contributing

Contributions are welcome, especially in the following areas:

- Improving vision-language interpretation accuracy
- Adding support for new game modes or HUD layouts
- Enhancing reasoning prompts or evaluation logic
- Performance optimizations for longer matches

Suggested workflow:

1. Fork the repository
2. Create a feature branch
3. Make focused, well-documented changes
4. Submit a pull request with a clear description

This project prioritizes **clarity, evidence-grounding, and reproducibility** over feature sprawl.

---

## ❓ FAQ

**Q: Why not use in-game stats or APIs?**\
A: The goal is to analyze *behavior*, not post-game aggregates. Video captures intent, positioning, and decision-making that stats miss.

**Q: Why separate the interpreter and reasoner?**\
A: Separation improves explainability, reduces hallucination, and makes debugging and iteration significantly easier.

**Q: Is this meant to replace coaching or VOD review?**\
A: No. It augments manual review by surfacing patterns that are hard to notice across matches.

**Q: Can this work on lower-end hardware?**\
A: Partially. Quality modes trade coverage for detail, but 16GB+ VRAM is recommended for full fidelity.

**Q: Is this Call of Duty–specific?**\
A: The current implementation is, but the architecture is intentionally generalizable.

---

## 📄 License

License is **to be determined**.

The project is currently intended for portfolio, research, and educational use. Licensing will be added once scope and distribution goals are finalized.

