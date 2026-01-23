"""
Gameplay Reasoner - Conversational Analysis Agent

A conversational agent that analyzes gameplay using interpreter output,
maintains player history, identifies patterns, and provides coaching.

Works with Interpreter_v4 output to provide contextual feedback.

Model: qwen3:14b-q4_K_M via Ollama (reasoning model, text-only)
"""

from __future__ import annotations

import json
import os
import re
from dataclasses import dataclass, field, asdict
from datetime import datetime
from pathlib import Path
from typing import Optional, Literal
from uuid import uuid4

from ollama import chat

from config import config


# =============================================================================
# CONFIGURATION
# =============================================================================

DEFAULT_MODEL = config.reasoner_model
MAX_CONTEXT_MATCHES = 5
MAX_CONTEXT_RECOMMENDATIONS = 5
MAX_CHAT_HISTORY_TURNS = 10  # Keep last N conversation turns (user + assistant pairs)


# =============================================================================
# DATA MODELS - Public API
# =============================================================================

@dataclass
class Tendency:
    """A observed player tendency (strength or weakness)."""
    id: str
    description: str
    tendency_type: Literal["strength", "weakness"]
    first_observed: str  # ISO date
    last_observed: str   # ISO date
    observation_count: int
    map_specific: Optional[str] = None  # If only applies to certain map
    mode_specific: Optional[str] = None  # If only applies to certain mode
    notes: Optional[str] = None

    @classmethod
    def create(
        cls,
        description: str,
        tendency_type: Literal["strength", "weakness"],
        map_specific: Optional[str] = None,
        mode_specific: Optional[str] = None,
        notes: Optional[str] = None,
    ) -> Tendency:
        now = datetime.now().strftime("%Y-%m-%d")
        return cls(
            id=f"tend_{uuid4().hex[:8]}",
            description=description,
            tendency_type=tendency_type,
            first_observed=now,
            last_observed=now,
            observation_count=1,
            map_specific=map_specific,
            mode_specific=mode_specific,
            notes=notes,
        )


@dataclass
class Recommendation:
    """A coaching recommendation given to the player."""
    id: str
    created_at: str  # ISO datetime
    recommendation: str
    targets_tendency: Optional[str] = None  # ID of related tendency
    context: Optional[str] = None  # What prompted this recommendation
    status: Literal["pending", "in_progress", "applied", "dismissed"] = "pending"
    outcome: Optional[Literal["helped", "no_effect", "hurt"]] = None
    outcome_notes: Optional[str] = None

    @classmethod
    def create(
        cls,
        recommendation: str,
        targets_tendency: Optional[str] = None,
        context: Optional[str] = None,
    ) -> Recommendation:
        return cls(
            id=f"rec_{uuid4().hex[:8]}",
            created_at=datetime.now().isoformat(),
            recommendation=recommendation,
            targets_tendency=targets_tendency,
            context=context,
        )


@dataclass
class MatchRecord:
    """Summary of an analyzed match."""
    match_id: str
    analyzed_at: str  # ISO datetime
    map_name: str
    mode: str
    video_path: str
    duration_seconds: float
    summary: str  # Brief AI-generated summary
    key_moments: list[str] = field(default_factory=list)
    tendencies_observed: list[str] = field(default_factory=list)  # Tendency IDs
    recommendations_given: list[str] = field(default_factory=list)  # Recommendation IDs

    @classmethod
    def create(
        cls,
        map_name: str,
        mode: str,
        video_path: str,
        duration_seconds: float,
        summary: str,
        key_moments: Optional[list[str]] = None,
    ) -> MatchRecord:
        return cls(
            match_id=f"match_{uuid4().hex[:8]}",
            analyzed_at=datetime.now().isoformat(),
            map_name=map_name,
            mode=mode,
            video_path=video_path,
            duration_seconds=duration_seconds,
            summary=summary,
            key_moments=key_moments or [],
        )


@dataclass
class FollowUp:
    """Records whether a recommendation was applied in a subsequent match."""
    id: str
    recommendation_id: str
    match_id: str
    recorded_at: str  # ISO datetime
    was_applied: bool
    applied_correctly: bool = False
    notes: Optional[str] = None

    @classmethod
    def create(
        cls,
        recommendation_id: str,
        match_id: str,
        was_applied: bool,
        applied_correctly: bool = False,
        notes: Optional[str] = None,
    ) -> FollowUp:
        return cls(
            id=f"follow_{uuid4().hex[:8]}",
            recommendation_id=recommendation_id,
            match_id=match_id,
            recorded_at=datetime.now().isoformat(),
            was_applied=was_applied,
            applied_correctly=applied_correctly,
            notes=notes,
        )


@dataclass
class SessionStart:
    """Returned when starting a new analysis session."""
    session_id: str
    match_id: str
    opening_message: str
    key_observations: list[str]
    comparisons_to_past: list[str]
    relevant_recommendations: list[Recommendation]


@dataclass
class ChatResponse:
    """Returned from each chat turn."""
    message: str
    new_recommendations: list[Recommendation] = field(default_factory=list)
    tendencies_updated: list[str] = field(default_factory=list)


@dataclass
class SessionSummary:
    """Returned when ending a session."""
    match_id: str
    tendencies_identified: list[Tendency]
    recommendations_given: list[Recommendation]
    follow_ups_recorded: list[FollowUp]


# =============================================================================
# DATABASE LAYER - Internal
# =============================================================================

@dataclass
class _PlayerData:
    """Internal representation of the full player profile."""
    player_id: str
    created_at: str
    updated_at: str
    tendencies: list[Tendency] = field(default_factory=list)
    matches: list[MatchRecord] = field(default_factory=list)
    recommendations: list[Recommendation] = field(default_factory=list)
    follow_ups: list[FollowUp] = field(default_factory=list)


class PlayerDatabase:
    """
    Handles persistence of player data to JSON file.
    
    Internal class - external code should use GameplayReasoner methods.
    """

    def __init__(self, path: str):
        self.path = Path(path)
        self._data: Optional[_PlayerData] = None

    def exists(self) -> bool:
        return self.path.exists()

    def load(self) -> _PlayerData:
        """Load player data from disk."""
        if not self.path.exists():
            raise FileNotFoundError(f"Player profile not found: {self.path}")

        with open(self.path, "r") as f:
            raw = json.load(f)

        self._data = _PlayerData(
            player_id=raw["player_id"],
            created_at=raw["created_at"],
            updated_at=raw["updated_at"],
            tendencies=[Tendency(**t) for t in raw.get("tendencies", [])],
            matches=[MatchRecord(**m) for m in raw.get("matches", [])],
            recommendations=[Recommendation(**r) for r in raw.get("recommendations", [])],
            follow_ups=[FollowUp(**f) for f in raw.get("follow_ups", [])],
        )
        return self._data

    def save(self) -> None:
        """Save player data to disk."""
        if self._data is None:
            raise ValueError("No data to save - load or create first")

        self._data.updated_at = datetime.now().isoformat()

        # Convert to dict, handling nested dataclasses
        data_dict = {
            "player_id": self._data.player_id,
            "created_at": self._data.created_at,
            "updated_at": self._data.updated_at,
            "tendencies": [asdict(t) for t in self._data.tendencies],
            "matches": [asdict(m) for m in self._data.matches],
            "recommendations": [asdict(r) for r in self._data.recommendations],
            "follow_ups": [asdict(f) for f in self._data.follow_ups],
        }

        # Ensure directory exists
        self.path.parent.mkdir(parents=True, exist_ok=True)

        with open(self.path, "w") as f:
            json.dump(data_dict, f, indent=2)

    def create(self, player_id: str) -> _PlayerData:
        """Create a new player profile."""
        now = datetime.now().isoformat()
        self._data = _PlayerData(
            player_id=player_id,
            created_at=now,
            updated_at=now,
        )
        self.save()
        return self._data

    @property
    def data(self) -> _PlayerData:
        """Get current data, loading if necessary."""
        if self._data is None:
            self.load()
        return self._data

    # ---- Tendency Operations ----

    def add_tendency(self, tendency: Tendency) -> None:
        self.data.tendencies.append(tendency)

    def get_tendency(self, tendency_id: str) -> Optional[Tendency]:
        for t in self.data.tendencies:
            if t.id == tendency_id:
                return t
        return None

    def update_tendency_observation(self, tendency_id: str, notes: Optional[str] = None) -> bool:
        """Increment observation count and update last_observed."""
        tendency = self.get_tendency(tendency_id)
        if tendency:
            tendency.observation_count += 1
            tendency.last_observed = datetime.now().strftime("%Y-%m-%d")
            if notes:
                tendency.notes = notes
            return True
        return False

    def get_tendencies_by_type(self, tendency_type: Literal["strength", "weakness"]) -> list[Tendency]:
        return [t for t in self.data.tendencies if t.tendency_type == tendency_type]

    # ---- Match Operations ----

    def add_match(self, match: MatchRecord) -> None:
        self.data.matches.append(match)

    def get_match(self, match_id: str) -> Optional[MatchRecord]:
        for m in self.data.matches:
            if m.match_id == match_id:
                return m
        return None

    def get_recent_matches(self, limit: int = 5) -> list[MatchRecord]:
        """Get most recent matches, newest first."""
        sorted_matches = sorted(
            self.data.matches,
            key=lambda m: m.analyzed_at,
            reverse=True,
        )
        return sorted_matches[:limit]

    def get_matches_by_map(self, map_name: str) -> list[MatchRecord]:
        return [m for m in self.data.matches if m.map_name.lower() == map_name.lower()]

    # ---- Recommendation Operations ----

    def add_recommendation(self, rec: Recommendation) -> None:
        self.data.recommendations.append(rec)

    def get_recommendation(self, rec_id: str) -> Optional[Recommendation]:
        for r in self.data.recommendations:
            if r.id == rec_id:
                return r
        return None

    def get_pending_recommendations(self) -> list[Recommendation]:
        return [r for r in self.data.recommendations if r.status == "pending"]

    def update_recommendation_status(
        self,
        rec_id: str,
        status: Literal["pending", "in_progress", "applied", "dismissed"],
        outcome: Optional[Literal["helped", "no_effect", "hurt"]] = None,
        outcome_notes: Optional[str] = None,
    ) -> bool:
        rec = self.get_recommendation(rec_id)
        if rec:
            rec.status = status
            if outcome:
                rec.outcome = outcome
            if outcome_notes:
                rec.outcome_notes = outcome_notes
            return True
        return False

    # ---- Follow-up Operations ----

    def add_follow_up(self, follow_up: FollowUp) -> None:
        self.data.follow_ups.append(follow_up)

    def get_follow_ups_for_recommendation(self, rec_id: str) -> list[FollowUp]:
        return [f for f in self.data.follow_ups if f.recommendation_id == rec_id]

    # ---- Context Building ----

    def get_context_summary(self, current_map: Optional[str] = None) -> str:
        """
        Build a context summary for injection into prompts.
        
        This is what the model sees about the player's history.
        """
        lines = []

        # Strengths
        strengths = self.get_tendencies_by_type("strength")
        if strengths:
            lines.append("PLAYER STRENGTHS:")
            for s in strengths[:5]:  # Limit to 5
                map_note = f" (on {s.map_specific})" if s.map_specific else ""
                lines.append(f"- {s.description}{map_note} [observed {s.observation_count}x]")
            lines.append("")

        # Weaknesses
        weaknesses = self.get_tendencies_by_type("weakness")
        if weaknesses:
            lines.append("PLAYER WEAKNESSES:")
            for w in weaknesses[:5]:
                map_note = f" (on {w.map_specific})" if w.map_specific else ""
                lines.append(f"- {w.description}{map_note} [observed {w.observation_count}x]")
            lines.append("")

        # Pending recommendations
        pending = self.get_pending_recommendations()
        if pending:
            lines.append("PENDING RECOMMENDATIONS TO FOLLOW UP ON:")
            for r in pending[:MAX_CONTEXT_RECOMMENDATIONS]:
                lines.append(f"- [{r.id}] {r.recommendation}")
            lines.append("")

        # Recent match history
        recent = self.get_recent_matches(MAX_CONTEXT_MATCHES)
        if recent:
            lines.append("RECENT MATCHES:")
            for m in recent:
                lines.append(f"- {m.map_name} ({m.mode}) on {m.analyzed_at[:10]}: {m.summary}")
            lines.append("")

        # Map-specific context if provided
        if current_map:
            map_matches = self.get_matches_by_map(current_map)
            if map_matches:
                lines.append(f"HISTORY ON {current_map.upper()} ({len(map_matches)} matches played):")
                for m in map_matches[-3:]:  # Last 3 on this map
                    lines.append(f"- {m.analyzed_at[:10]}: {m.summary}")
                lines.append("")

        return "\n".join(lines) if lines else "No prior history with this player."


# =============================================================================
# PROMPTS
# =============================================================================

SYSTEM_PROMPT = """You are an expert Call of Duty: Black Ops 6 performance analyst reviewing first-person gameplay to help the POV player improve.

DATA YOU HAVE:
- Text-only narration derived from gameplay frames (no audio).
- You can only analyze what is explicitly stated in that narration.

OUTPUT GOAL:
Write like an observer who identifies *tendencies* and *their impact*, e.g.:
"From what I can see... What helped you... A tendency that helped... One pattern to improve... My guess is..."

EVIDENCE REQUIREMENT (non-negotiable):
- Every claim about a tendency (aggression, aim quality, cover usage, map area, objective behavior, etc.) must cite at least 1 concrete anchor from the narration.
- Valid anchors: a timestamp like [12.3s], a segment label like [120.0s-150.0s], or exact HUD/medal text.
- If you cannot cite an anchor, say it's unclear and do not state it as fact.

STRICT LIMITS:
- Do NOT talk about allies/coordination/communication/callouts/voice chat.
- Do NOT speculate about what others did or should do.
- Avoid the word "team" entirely unless the match mode name provided literally contains it and you are repeating the mode title verbatim.

MODE AWARENESS:
- Use only the terminology appropriate to the provided mode name.
- Never mention other modes.

ANALYSIS STYLE:
- Observation → tendency → impact → actionable experiment.
- Prefer concrete details (cover objects, landmarks, lanes, angles) if they appear in the narration.
- Keep recommendations high-leverage (1-3), specific, and implementable solo.

FORMAT:
- Initial analysis: respond in the requested JSON schema.
- Follow-up chat: respond in the requested JSON schema."""


def _build_session_start_prompt(
    player_context: str,
    match_narration: str,
    map_name: str,
    mode: str,
    duration: float,
) -> str:
    """Build the prompt for starting a new analysis session."""
    return f"""A player just completed a {mode} match on {map_name} ({duration:.0f} seconds).

CRITICAL REMINDER: This is {mode.upper()} mode. Use only {mode}-appropriate terminology.

=== PLAYER HISTORY ===
{player_context if player_context else "This is a new player with no history yet."}

=== CURRENT MATCH NARRATION ===
(This is a text description of what was observed in the player's POV. There is NO audio data.)
{match_narration}

=== YOUR TASK ===
Analyze THIS PLAYER's individual performance. Focus ONLY on what the player did and could do differently.

Your job is to produce an evidence-grounded, tendency-based analysis in a human style.

Opening_message requirements (2-3 short paragraphs):
- Paragraph 1: overall style (tempo/aggression), where the player spends time on the map, and how they approach the objective — with evidence anchors.
- Paragraph 2: what HELPED the player (2-3 strengths) — each backed by evidence anchors and why it mattered.
- Paragraph 3: one pattern to improve (1-2 weaknesses) + ONE concrete experiment to try next match.

Key_observations requirements:
- 3-6 items.
- Each item starts with "Strength:" or "Weakness:".
- Each item includes at least one evidence anchor in parentheses.

Hard constraints:
- Do not mention allies, coordination, callouts, or anything outside the POV narration.
- Do not use the word "team" unless you are repeating the provided mode title verbatim.
- Use only {mode}-appropriate terminology.

Respond in this JSON format:
{{
    "opening_message": "Your conversational analysis (2-3 paragraphs, focused on individual performance)",
    "key_observations": ["observation about player habit 1", "observation 2", ...],
    "comparisons_to_past": ["comparison 1", ...],  // Empty list if new player
    "new_tendencies": [
        {{"type": "strength|weakness", "description": "...", "map_specific": null|"MapName"}}
    ],
    "new_recommendations": [
        {{"recommendation": "specific actionable advice for THIS player", "reason": "why this will help"}}
    ],
    "follow_up_notes": [
        {{"recommendation_id": "rec_xxx", "was_applied": true|false, "notes": "..."}}
    ]
}}"""


def _build_chat_prompt(user_message: str, match_narration: str, mode: str) -> str:
    """Build prompt for a follow-up chat message."""
    return f"""The player asks: "{user_message}"

REMINDERS:
- This was a {mode} match. Use correct terminology.
- You only have the narration (no audio).
- Focus on what THIS player did and can do.
- Ground claims in concrete anchors (timestamps/segment labels/HUD text). If you can't, say it's unclear.
- Do not mention allies, coordination, or communication.

Match narration for reference:
{match_narration}

Respond naturally to their question. Focus on individual performance analysis.
If your response includes new recommendations or tendency observations, include them in JSON.

Respond in this JSON format:
{{
    "message": "Your conversational response",
    "new_tendencies": [],
    "new_recommendations": []
}}"""


# =============================================================================
# RESPONSE PARSING
# =============================================================================

def _parse_json_response(response: str) -> dict:
    """Extract JSON from model response, handling markdown blocks."""
    text = response.strip()

    # Handle markdown code blocks
    if text.startswith("```"):
        lines = text.split("\n")
        json_lines = []
        in_block = False
        for line in lines:
            if line.startswith("```"):
                if in_block:
                    break
                in_block = True
                continue
            if in_block:
                json_lines.append(line)
        text = "\n".join(json_lines)

    # Try direct parse
    try:
        return json.loads(text)
    except json.JSONDecodeError:
        pass

    # Try to find JSON object in response
    start = response.find("{")
    end = response.rfind("}") + 1
    if 0 <= start < end:
        try:
            return json.loads(response[start:end])
        except json.JSONDecodeError:
            pass

    # Fallback - return response as message
    return {"message": response, "opening_message": response}


# =============================================================================
# PUBLIC API
# =============================================================================

class GameplayReasoner:
    """
    Conversational gameplay analysis agent.

    Example usage:
        reasoner = GameplayReasoner("player_profile.json")
        
        # Create profile if needed
        if not reasoner.has_profile():
            reasoner.create_profile("player1")
        
        # Start session with interpreter output
        session = reasoner.start_session(interpreter_result)
        print(session.opening_message)
        
        # Chat
        response = reasoner.chat("Why do I keep dying there?")
        print(response.message)
        
        # End session
        summary = reasoner.end_session()
    """

    def __init__(self, profile_path: str, model: str = DEFAULT_MODEL, verbose: bool = True):
        """
        Initialize the reasoner.

        Args:
            profile_path: Path to player profile JSON file
            model: Ollama model name for reasoning
            verbose: Whether to print debug info
        """
        self._db = PlayerDatabase(profile_path)
        self._model = model
        self._verbose = verbose

        # Session state
        self._session_id: Optional[str] = None
        self._current_match: Optional[MatchRecord] = None
        self._match_narration: Optional[str] = None
        self._match_mode: Optional[str] = None  # Store mode for chat context
        self._player_context: Optional[str] = None  # Player history at session start
        self._initial_analysis: Optional[str] = None  # The opening analysis (always kept in context)
        self._conversation_history: list[dict] = []  # Ongoing chat messages
        self._session_tendencies: list[Tendency] = []
        self._session_recommendations: list[Recommendation] = []
        self._session_follow_ups: list[FollowUp] = []

    def _log(self, msg: str) -> None:
        if self._verbose:
            print(f"[Reasoner] {msg}")

    # ---- Profile Management ----

    def has_profile(self) -> bool:
        """Check if a player profile exists."""
        return self._db.exists()

    def create_profile(self, player_id: str) -> None:
        """Create a new player profile."""
        self._db.create(player_id)
        self._log(f"Created profile for player: {player_id}")

    def get_profile_summary(self) -> dict:
        """Get a summary of the player profile for external use."""
        data = self._db.data
        return {
            "player_id": data.player_id,
            "total_matches": len(data.matches),
            "strengths": len(self._db.get_tendencies_by_type("strength")),
            "weaknesses": len(self._db.get_tendencies_by_type("weakness")),
            "pending_recommendations": len(self._db.get_pending_recommendations()),
        }

    # ---- Session Management ----

    def start_session(self, interpreter_output) -> SessionStart:
        """
        Start a new analysis session with interpreter output.

        Args:
            interpreter_output: PipelineResult from Interpreter_v4

        Returns:
            SessionStart with opening analysis
        """
        if self._session_id is not None:
            raise RuntimeError("Session already active. Call end_session() first.")

        self._session_id = f"session_{uuid4().hex[:8]}"
        self._log(f"Starting session: {self._session_id}")

        # Extract match info from interpreter output
        match_meta = interpreter_output.match_metadata
        map_name = match_meta.map_name
        mode = match_meta.mode
        duration = interpreter_output.duration_seconds
        video_path = interpreter_output.video_path
        narration = interpreter_output.merged_narration or interpreter_output.narration

        self._match_narration = narration
        self._match_mode = mode  # Store for chat context

        # Build player context
        player_context = self._db.get_context_summary(current_map=map_name)
        self._player_context = player_context  # Store for reference

        # Get pending recommendations for reference
        pending_recs = self._db.get_pending_recommendations()

        # Build and send prompt
        prompt = _build_session_start_prompt(
            player_context=player_context,
            match_narration=narration,
            map_name=map_name,
            mode=mode,
            duration=duration,
        )

        self._log("Generating initial analysis...")
        response = self._call_model(prompt)
        parsed = _parse_json_response(response)

        # Store the initial analysis (this stays in context for the whole session)
        self._initial_analysis = parsed.get("opening_message", response)

        # Create match record
        self._current_match = MatchRecord.create(
            map_name=map_name,
            mode=mode,
            video_path=video_path,
            duration_seconds=duration,
            summary=parsed.get("opening_message", "")[:200],  # Truncate for summary
            key_moments=parsed.get("key_observations", []),
        )

        # Process new tendencies from response
        for t in parsed.get("new_tendencies", []):
            tendency = Tendency.create(
                description=t["description"],
                tendency_type=t["type"],
                map_specific=t.get("map_specific"),
            )
            self._session_tendencies.append(tendency)
            self._current_match.tendencies_observed.append(tendency.id)

        # Process new recommendations from response
        for r in parsed.get("new_recommendations", []):
            rec = Recommendation.create(
                recommendation=r["recommendation"],
                context=r.get("reason"),
            )
            self._session_recommendations.append(rec)
            self._current_match.recommendations_given.append(rec.id)

        # Process follow-up notes
        for f in parsed.get("follow_up_notes", []):
            if f.get("recommendation_id"):
                follow_up = FollowUp.create(
                    recommendation_id=f["recommendation_id"],
                    match_id=self._current_match.match_id,
                    was_applied=f.get("was_applied", False),
                    applied_correctly=f.get("applied_correctly", False),
                    notes=f.get("notes"),
                )
                self._session_follow_ups.append(follow_up)

        # NOTE: We don't store the initial prompt/response in _conversation_history
        # because it's large. Instead, we build a condensed context for chat.

        return SessionStart(
            session_id=self._session_id,
            match_id=self._current_match.match_id,
            opening_message=parsed.get("opening_message", response),
            key_observations=parsed.get("key_observations", []),
            comparisons_to_past=parsed.get("comparisons_to_past", []),
            relevant_recommendations=pending_recs[:MAX_CONTEXT_RECOMMENDATIONS],
        )

    def chat(self, message: str) -> ChatResponse:
        """
        Send a follow-up message in the current session.

        Args:
            message: User's message

        Returns:
            ChatResponse with model's reply
        """
        if self._session_id is None:
            raise RuntimeError("No active session. Call start_session() first.")

        self._log(f"Processing: {message[:50]}...")
        
        # Build chat prompt with mode context
        prompt = _build_chat_prompt(message, self._match_narration, self._match_mode)
        
        # For chat, we send the structured prompt
        # The model has context via _build_chat_context() in _call_model
        response = self._call_model(prompt, include_history=True)
        
        # For chat responses, we accept either plain text or JSON
        # Try to parse as JSON, but fall back to plain text
        parsed = _parse_json_response(response)
        response_message = parsed.get("message", response)

        # Store in history - exactly what the model saw and responded
        self._conversation_history.append({"role": "user", "content": message})
        self._conversation_history.append({"role": "assistant", "content": response_message})

        # Trim history if it gets too long
        max_messages = MAX_CHAT_HISTORY_TURNS * 2
        if len(self._conversation_history) > max_messages:
            self._conversation_history = self._conversation_history[-max_messages:]

        # Process any new tendencies (if model included them in JSON)
        new_tendency_descriptions = []
        for t in parsed.get("new_tendencies", []):
            tendency = Tendency.create(
                description=t.get("description", ""),
                tendency_type=t.get("type", "weakness"),
                map_specific=t.get("map_specific"),
            )
            if tendency.description:
                self._session_tendencies.append(tendency)
                new_tendency_descriptions.append(tendency.description)
                if self._current_match:
                    self._current_match.tendencies_observed.append(tendency.id)

        # Process any new recommendations (if model included them in JSON)
        new_recs = []
        for r in parsed.get("new_recommendations", []):
            rec = Recommendation.create(
                recommendation=r.get("recommendation", ""),
                context=r.get("reason"),
            )
            if rec.recommendation:
                self._session_recommendations.append(rec)
                new_recs.append(rec)
                if self._current_match:
                    self._current_match.recommendations_given.append(rec.id)

        return ChatResponse(
            message=response_message,
            new_recommendations=new_recs,
            tendencies_updated=new_tendency_descriptions,
        )

    def end_session(self) -> SessionSummary:
        """
        End the current session and save all data.

        Returns:
            SessionSummary with what was recorded
        """
        if self._session_id is None:
            raise RuntimeError("No active session.")

        self._log("Ending session and saving data...")

        # Save match record
        if self._current_match:
            self._db.add_match(self._current_match)

        # Save tendencies
        for tendency in self._session_tendencies:
            # Check if similar tendency already exists
            existing = None
            for t in self._db.data.tendencies:
                if t.description.lower() == tendency.description.lower():
                    existing = t
                    break

            if existing:
                self._db.update_tendency_observation(existing.id)
            else:
                self._db.add_tendency(tendency)

        # Save recommendations
        for rec in self._session_recommendations:
            self._db.add_recommendation(rec)

        # Save follow-ups
        for follow_up in self._session_follow_ups:
            self._db.add_follow_up(follow_up)
            # Update recommendation status if applied
            if follow_up.was_applied:
                self._db.update_recommendation_status(
                    follow_up.recommendation_id,
                    "applied" if follow_up.applied_correctly else "in_progress",
                )

        # Persist to disk
        self._db.save()

        summary = SessionSummary(
            match_id=self._current_match.match_id if self._current_match else "",
            tendencies_identified=self._session_tendencies,
            recommendations_given=self._session_recommendations,
            follow_ups_recorded=self._session_follow_ups,
        )

        # Reset session state
        self._session_id = None
        self._current_match = None
        self._match_narration = None
        self._match_mode = None
        self._player_context = None
        self._initial_analysis = None
        self._conversation_history = []
        self._session_tendencies = []
        self._session_recommendations = []
        self._session_follow_ups = []

        self._log("Session ended and data saved.")
        return summary

    # ---- Direct Queries (for UI) ----

    def get_match_history(self, limit: int = 10) -> list[MatchRecord]:
        """Get recent match history."""
        return self._db.get_recent_matches(limit)

    def get_recommendations(self, status: Optional[str] = None) -> list[Recommendation]:
        """Get recommendations, optionally filtered by status."""
        if status == "pending":
            return self._db.get_pending_recommendations()
        return self._db.data.recommendations

    def get_tendencies(self, tendency_type: Optional[str] = None) -> list[Tendency]:
        """Get player tendencies, optionally filtered by type."""
        if tendency_type in ("strength", "weakness"):
            return self._db.get_tendencies_by_type(tendency_type)
        return self._db.data.tendencies

    # ---- Model Interaction ----

    def _build_chat_context(self) -> str:
        """Build a condensed context summary for chat turns."""
        parts = []
        
        # Match info with mode emphasis
        if self._current_match:
            parts.append(f"Current match: {self._current_match.map_name} ({self._current_match.mode})")
            parts.append(f"IMPORTANT: This is {self._current_match.mode} mode. Use correct terminology.")
        
        # Initial analysis summary (condensed)
        if self._initial_analysis:
            # Take first 500 chars of initial analysis
            summary = self._initial_analysis[:500]
            if len(self._initial_analysis) > 500:
                summary += "..."
            parts.append(f"Initial analysis: {summary}")
        
        # Full match narration summary (so model can reference specific moments)
        if self._match_narration:
            parts.append(f"=== MATCH NARRATION (text only, no audio) ===\n{self._match_narration}")
        
        # Any recommendations given this session
        if self._session_recommendations:
            recs = [r.recommendation for r in self._session_recommendations[:3]]
            parts.append(f"Recommendations given: {'; '.join(recs)}")
        
        return "\n".join(parts)

    def _call_model(self, prompt: str, include_history: bool = False) -> str:
        """
        Call the language model with proper context management.
        
        For chat (include_history=True):
        - System prompt
        - Condensed session context (match info, initial analysis summary)
        - Recent conversation history (last N turns)
        - Current user message
        
        For initial analysis (include_history=False):
        - System prompt
        - Full prompt with all context
        """
        messages = [{"role": "system", "content": SYSTEM_PROMPT}]

        if include_history:
            # Add condensed context as a system-level reminder
            context_reminder = self._build_chat_context()
            if context_reminder:
                messages.append({
                    "role": "system", 
                    "content": f"Session context:\n{context_reminder}"
                })
            
            # Add conversation history (last N turns, each turn = 2 messages)
            history_to_include = self._conversation_history[-(MAX_CHAT_HISTORY_TURNS * 2):]
            messages.extend(history_to_include)

        messages.append({"role": "user", "content": prompt})

        def has_forbidden_social_language(text: str) -> bool:
            """Guardrail to prevent drifting into ally/coordination commentary."""
            if not text:
                return False
            mode = (self._match_mode or "").strip().lower()
            cleaned = text
            # If the mode name itself contains the word, allow ONLY the exact title.
            if mode == "team deathmatch":
                cleaned = re.sub(r"\bTeam Deathmatch\b", "", cleaned, flags=re.I)
            # Catch "team", "teammate", "teamwork", etc.
            return bool(re.search(r"\bteam\w*\b", cleaned, flags=re.I))

        # Use a lower temperature to reduce generic/fluffy summaries and constraint violations.
        temperature = 0.45 if include_history else 0.3

        try:
            response = chat(
                model=self._model,
                messages=messages,
                options={
                    "num_predict": 2800,
                    "temperature": temperature,
                },
            )
            text = response.message.content

            # One targeted retry if the model violates the "no allies/coordination" constraint.
            if has_forbidden_social_language(text):
                messages.append({
                    "role": "system",
                    "content": (
                        "Rewrite your response to remove the word 'team' (and any derivatives such as 'teammate') "
                        "and remove any implication of ally coordination. Focus strictly on the POV player. "
                        "Keep the exact same JSON schema. Ground claims in evidence anchors."),
                })
                retry = chat(
                    model=self._model,
                    messages=messages,
                    options={
                        "num_predict": 2800,
                        "temperature": 0.2,
                    },
                )
                return retry.message.content

            return text
        except Exception as e:
            self._log(f"Model error: {e}")
            raise


# =============================================================================
# CONVENIENCE FUNCTION
# =============================================================================

def analyze_and_discuss(
    interpreter_output,
    profile_path: str = "player_profile.json",
    player_id: str = "player",
    model: str = DEFAULT_MODEL,
) -> None:
    """
    Convenience function for interactive analysis session.

    Args:
        interpreter_output: PipelineResult from interpreter_v4
        profile_path: Path to player profile JSON
        player_id: Player ID if creating new profile
        model: Ollama model name
    """
    reasoner = GameplayReasoner(profile_path, model=model)

    # Create profile if needed
    if not reasoner.has_profile():
        print(f"Creating new player profile: {player_id}")
        reasoner.create_profile(player_id)

    # Start session
    print("\n" + "=" * 60)
    print("GAMEPLAY ANALYSIS")
    print("=" * 60 + "\n")

    session = reasoner.start_session(interpreter_output)
    print(session.opening_message)

    if session.key_observations:
        print("\n📌 Key Observations:")
        for obs in session.key_observations:
            print(f"   • {obs}")

    if session.comparisons_to_past:
        print("\n📊 Compared to Past:")
        for comp in session.comparisons_to_past:
            print(f"   • {comp}")

    print("\n" + "-" * 60)
    print("Ask questions about your gameplay, or type 'quit' to end.")
    print("-" * 60 + "\n")

    # Chat loop
    while True:
        try:
            user_input = input("You: ").strip()
        except (EOFError, KeyboardInterrupt):
            break

        if not user_input:
            continue
        if user_input.lower() in ("quit", "exit", "q"):
            break

        response = reasoner.chat(user_input)
        print(f"\nAnalyst: {response.message}\n")

        if response.new_recommendations:
            print("💡 New Recommendations:")
            for rec in response.new_recommendations:
                print(f"   • {rec.recommendation}")
            print()

    # End session
    summary = reasoner.end_session()

    print("\n" + "=" * 60)
    print("SESSION SUMMARY")
    print("=" * 60)
    print(f"Match recorded: {summary.match_id}")
    print(f"Tendencies identified: {len(summary.tendencies_identified)}")
    print(f"Recommendations given: {len(summary.recommendations_given)}")
    print(f"Follow-ups recorded: {len(summary.follow_ups_recorded)}")
    print("\nData saved to profile. See you next match!")


# =============================================================================
# CLI
# =============================================================================

def main() -> int:
    """Command-line entry point for testing."""
    import sys

    print("Gameplay Reasoner")
    print("=" * 70)
    print("\nThis module is designed to be used with Interpreter_v4 output.")
    print("\nExample usage:")
    print("  from interpreter_v4 import analyze_video")
    print("  from reasoner import analyze_and_discuss")
    print("")
    print("  result = analyze_video('match.mp4', 'Nuketown', 'Domination')")
    print("  analyze_and_discuss(result)")
    print("")

    # Quick test of database
    if len(sys.argv) > 1 and sys.argv[1] == "--test":
        print("\nRunning database test...")
        db = PlayerDatabase("test_profile.json")
        db.create("test_player")

        # Add some test data
        t = Tendency.create("Good aim", "strength")
        db.add_tendency(t)

        r = Recommendation.create("Check corners more often")
        db.add_recommendation(r)

        db.save()
        print(f"Created test profile: test_profile.json")
        print(f"Context summary:\n{db.get_context_summary()}")

        # Cleanup
        os.remove("test_profile.json")
        print("Test passed!")
        return 0

    return 0


if __name__ == "__main__":
    exit(main())