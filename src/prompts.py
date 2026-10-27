"""Prompt construction for the 9 strategies defined in 03_策略设计详细方案.md.

Strategy id = <semantic part>[_<purpose>]:
  S0, S1_T1, S3_T1, S5_T1, S5_T2, S5_T3, S1_T2, S1_T3, S3_only

All few-shot examples and pitfalls below are self-written and must never
quote a line from the main / dev / adversarial sets (see revision note in 03).
"""
import re
from typing import Dict, List

HEADER = "Translate the following Chinese classical poem to English."

# ---- translation purposes (T1 / T2 / T3) -----------------------------------
PURPOSES = {
    "T1": (
        "Translation Purpose: Academic translation for English readers with no Chinese background.\n"
        "Audience: University students, general readers interested in Chinese poetry.\n"
        "Constraints:\n"
        "- Preserve all key semantic units\n"
        "- Keep metaphors and personification\n"
        "- Allow reordering for English grammar\n"
        "- Don't rhyme at the cost of meaning\n"
        "- Don't add characters, events, or facts not in the original\n"
        "- You may add at most one brief note in brackets [like this] if a cultural reference needs explanation"
    ),
    "T2": (
        "Translation Purpose: Poetic rendering for English poetry reading.\n"
        "Audience: Poetry lovers at literary events.\n"
        "Constraints:\n"
        "- Prioritize rhythm and natural flow when read aloud\n"
        "- You may adapt imagery for English poetic conventions\n"
        "- You may reorder or restructure lines for English verse\n"
        "- Preserve the core emotional tone and main imagery\n"
        "- You may sacrifice minor details for poetic effect\n"
        "- No notes (pure poetry)"
    ),
    "T3": (
        "Translation Purpose: Teaching material for Chinese literature courses.\n"
        "Audience: Students learning about Chinese poetry.\n"
        "Constraints:\n"
        "- Preserve meaning accurately\n"
        "- Explain cultural references clearly\n"
        "- Make implicit logic explicit if needed\n"
        "- Prioritize clarity over poetic beauty\n"
        "- Don't add characters, events, or facts not in the original\n"
        "- You may add 1-2 brief explanatory notes in brackets"
    ),
}

OUTPUT_FORMATS = {
    "T1": "Output Format: 2-4 lines of English, no title, no author name, optionally one bracketed note.",
    "T2": "Output Format: 2-4 lines of English verse, no title, no author name, no notes.",
    "T3": "Output Format: 2-4 lines of English, no title, no author name, followed by 1-2 bracketed notes if useful.",
}

# Used by S1 / S3 / S5 together with a purpose block.
KEY_REQUIREMENTS = (
    "Key Requirements:\n"
    "- Preserve core meaning, key imagery, and emotional tone\n"
    "- Classical Chinese often uses:\n"
    "  * Emotional paradoxes (seeming contradictions)\n"
    "  * Personification of objects\n"
    "  * Implicit negation\n"
    "  * Metaphorical references\n"
    "- Keep personification and metaphors in English"
)

# ---- self-written boundary examples (S3 / S5) ------------------------------
EXAMPLES = (
    "Examples of Common Mistakes to Avoid:\n\n"
    "Example 1: Emotional Paradox\n"
    '❌ Wrong: "情愈深时语愈淡" → "passionate but plain words" (literal, loses the escalation)\n'
    '✅ Right: "情愈深时语愈淡" → "the deeper the feeling, the plainer the words"\n\n'
    "Example 2: Emotional State\n"
    '❌ Wrong: "歌不成声泣无言" → "only feel cannot sing" (broken grammar)\n'
    '✅ Right: "歌不成声泣无言" → "my throat chokes, unable to sing; tears flow without words"\n\n'
    "Example 3: Implicit Negation\n"
    '❌ Wrong: "莫待花谢始惜春" → "wait until flowers fall to cherish spring" (negation reversed!)\n'
    '✅ Right: "莫待花谢始惜春" → "don\'t wait until flowers fall to cherish spring"\n\n'
    "Example 4: Personification\n"
    '❌ Wrong: "风为我叩门" → "I knock on the door when wind blows" (loses personification)\n'
    '✅ Right: "风为我叩门" → "the wind knocks at the door for me"\n\n'
    "Example 5: Action Subject\n"
    '❌ Wrong: "邻舟问客何处去" → "the traveler asks the boat" (wrong subject)\n'
    '✅ Right: "邻舟问客何处去" → "someone from the neighboring boat asks the traveler where he is headed"'
)

# ---- self-written pitfalls (S4 / S5) ---------------------------------------
PITFALLS = (
    "Common Pitfalls to Avoid:\n\n"
    "1. Negation Reversal\n"
    "   - Don't translate implicit negation as affirmation\n"
    '   - "莫待花谢始惜春" means "don\'t wait until flowers fall to cherish spring", not "wait until flowers fall"\n'
    "2. Personification Loss\n"
    "   - Keep personification of objects\n"
    '   - "山欲留客" = "the hills wish to keep the guest", not just "I stayed by the hills"\n'
    "3. Emotional Paradox Flattening\n"
    "   - Don't flatten emotional paradoxes into simple statements; preserve the tension\n"
    '   - "情愈深时语愈淡" has escalating tension between feeling and words\n'
    "4. Adding Information\n"
    "   - Don't add characters, events, or relationships not in the original\n"
    '   - "风为我叩门" = the wind knocks for me, not "wind brings a visitor"\n'
    "5. Who Does What to Whom\n"
    "   - Check action subjects and objects carefully\n"
    '   - "邻舟问客何处去" = the neighbor asks where the traveler goes (not the reverse)\n'
    "6. Temporal Relations\n"
    "   - Distinguish past, present, future, and hypothetical\n"
    '   - "待明年花开时" = when blossoms bloom next year (future), not past'
)

# strategy -> (uses_purpose, analysis, examples, pitfalls, purpose, requirements)
STRATEGIES: Dict[str, dict] = {
    "S0":      dict(purpose=None, req=False, analysis=False, examples=False, pitfalls=False),
    "S1_T1":   dict(purpose="T1", req=True, analysis=False, examples=False, pitfalls=False),
    "S3_T1":   dict(purpose="T1", req=True, analysis=False, examples=True, pitfalls=False),
    "S5_T1":   dict(purpose="T1", req=True, analysis=True, examples=True, pitfalls=True),
    "S5_T2":   dict(purpose="T2", req=True, analysis=True, examples=True, pitfalls=True),
    "S5_T3":   dict(purpose="T3", req=True, analysis=True, examples=True, pitfalls=True),
    "S1_T2":   dict(purpose="T2", req=True, analysis=False, examples=False, pitfalls=False),
    "S1_T3":   dict(purpose="T3", req=True, analysis=False, examples=False, pitfalls=False),
    "S3_only": dict(purpose=None, req=False, analysis=False, examples=True, pitfalls=False),
    # Added after the stage-1 review: isolate Chinese analysis (RQ1.2) and pitfall warnings (RQ1.3).
    # Not part of the 9 strategies screened on the dev set; appended to stage 2 as S1_T1 + one component.
    "S2_T1":   dict(purpose="T1", req=True, analysis=True, examples=False, pitfalls=False),
    "S4_T1":   dict(purpose="T1", req=True, analysis=False, examples=False, pitfalls=True),
}

DEFAULT_OUTPUT = "Output Format: English translation only, no title, no author name."


def format_poem(text: str) -> str:
    """One line per sentence (break after 。？！；), commas stay inside the line."""
    parts = [p for p in re.split(r"(?<=[。？！；])", text) if p.strip()]
    return "\n".join(parts)


def build_semantic_analysis(poem: dict) -> str:
    """Chinese interpretation built from the semantic-unit definitions.

    Only Chinese fields are used (chinese / semantic_content / must_preserve,
    or trap_point / must_detect for adversarial poems). English good/bad
    examples and pass conditions are never included.
    NOTE: this shares its source with the scoring rubric (see report limitations).
    """
    lines: List[str] = []
    for u in poem["semantic_units"]:
        if "semantic_content" in u:
            lines.append("“%s”：%s。要点：%s。" % (u["chinese"], u["semantic_content"], u["must_preserve"]))
        else:  # adversarial units
            lines.append("%s：%s。" % (u["trap_point"], u["must_detect"]))
    return "\n".join(lines)


def build_prompt(poem: dict, strategy: str) -> str:
    cfg = STRATEGIES[strategy]
    parts = [HEADER]
    if cfg["purpose"]:
        parts.append(PURPOSES[cfg["purpose"]])
    if cfg["req"]:
        parts.append(KEY_REQUIREMENTS)
    if cfg["analysis"]:
        parts.append("Semantic Analysis (in Chinese):\n" + build_semantic_analysis(poem))
    if cfg["examples"]:
        parts.append(EXAMPLES)
    if cfg["pitfalls"]:
        parts.append(PITFALLS)
    if cfg["purpose"]:
        parts.append(OUTPUT_FORMATS[cfg["purpose"]])
    elif strategy == "S3_only":
        parts.append(DEFAULT_OUTPUT)
    parts.append("Poem:\n" + format_poem(poem["text"]))
    return "\n\n".join(parts)


PATH_INSTRUCTION = (
    "Before translating, first state your understanding of the poem, then translate it.\n"
    "Respond with JSON only, in this format:\n"
    '{"understanding": [{"line": "<Chinese line>", "literal": "<literal meaning>", '
    '"deep_meaning": "<intended meaning>", "devices": ["paradox", "personification", "negation", "metaphor"], '
    '"negation": true or false}], "translation": ["<English line 1>", "<English line 2>"]}'
)


def build_path_prompt(poem: dict, strategy: str) -> str:
    """Stage-4 prompt: base strategy prompt + understanding-then-translation JSON."""
    return build_prompt(poem, strategy) + "\n\n" + PATH_INSTRUCTION
