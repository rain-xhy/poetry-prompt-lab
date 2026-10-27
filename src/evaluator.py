"""Scorer per 05 guide: blind unit-level binary check with evidence + five-dim quality.

The scorer only sees (original, unit, translation). Strategy / model / prompt are never passed in.
"""
import hashlib
import json
import re
from typing import Dict, List, Optional

from .config import ModelConfig
from .models import OllamaModel
from .prompts import build_semantic_analysis

SCORER = ModelConfig("Scorer", "qwen3:4b", think=False, temperature=0.0, top_p=1.0, max_tokens=400)

QUALITY_DIMS = ["adequacy", "imagery", "emotion", "fluency", "poetic_quality"]
QUALITY_WEIGHTS = {"adequacy": 0.30, "imagery": 0.25, "emotion": 0.20, "fluency": 0.15, "poetic_quality": 0.10}


def anon_code(translation_id: str) -> str:
    """Hash code that carries no strategy information."""
    return "T_" + hashlib.sha256(translation_id.encode("utf-8")).hexdigest()[:8]


def strip_notes(text: str) -> str:
    """Drop bracketed notes / 'Note:' lines so units are judged on the translation itself."""
    text = re.sub(r"\[[^\]]*\]", "", text)
    text = re.sub(r"(?im)^\s*note:.*$", "", text)
    return text.strip()


def parse_json(raw: str) -> Optional[dict]:
    if not raw:
        return None
    m = re.search(r"\{.*\}", raw, re.S)
    if not m:
        return None
    try:
        return json.loads(m.group(0))
    except json.JSONDecodeError:
        return None


def _norm(s: str) -> str:
    return re.sub(r"\s+", " ", s).strip().lower()


def _conds(unit: dict) -> List[str]:
    if "pass_conditions" in unit:
        return unit["pass_conditions"]
    return [unit.get("pass_condition", "")]


def unit_prompt(unit: dict, translation: str) -> str:
    focus = unit.get("chinese") or unit.get("trap_point", "")
    content = unit.get("semantic_content") or unit.get("must_detect", "")
    conds = "\n".join("- " + c for c in _conds(unit))
    return (
        "You are checking a translation of a Chinese classical poem.\n\n"
        "Original phrase: %s\nWhat must be preserved: %s\nPass conditions:\n%s\n\n"
        "Translation:\n%s\n\n"
        "The translation is in English, so the Chinese phrase will not appear in it. Judge by MEANING only: "
        "does the English express the required meaning, even if the wording differs from any example? "
        "A pass needs every condition met; quote the exact English words that show it.\n"
        'Reply with JSON only: {"pass": true or false, "evidence": "exact words quoted from the translation", "reasoning": "one sentence"}'
    ) % (focus, content, conds, translation)


def quality_prompt(original: str, translation: str, reference: str = "") -> str:
    """Layer 2 prompt (05 guide, 3.2): anchored 1-5 rubric plus reference notes on the original."""
    ref = ("Reference notes on what the original means (Chinese):\n%s\n\n" % reference) if reference else ""
    return (
        "You are a strict reviewer of English translations of Chinese classical poetry. "
        "Translations from small models usually contain real problems, so give 5 only when nothing is wrong.\n\n"
        "Original:\n%s\n\n%sTranslation:\n%s\n\n"
        "Rate each dimension from 1 to 5.\n"
        "adequacy: 5 complete and accurate, 4 minor loss of detail, 3 obvious problems but core kept, "
        "2 core meaning damaged, 1 serious error or opposite meaning\n"
        "imagery: 5 all metaphor, personification and imagery kept, 4 most kept, 3 partly kept, 2 most lost, 1 all lost\n"
        "emotion: 5 emotional tone and tension fully conveyed, 4 mostly conveyed, 3 partly, 2 deviates, 1 opposite or wrong\n"
        "fluency: 5 perfectly natural English, 4 fluent with minor issues, 3 slightly awkward, "
        "2 obvious grammar or expression problems, 1 hard to understand\n"
        "poetic_quality: 5 excellent English verse, 4 good, 3 acceptable, 2 prosaic, 1 awkward or unpoetic\n\n"
        'Reply with JSON only: {"adequacy":n,"imagery":n,"emotion":n,"fluency":n,"poetic_quality":n}'
    ) % (original, ref, translation)


def suitability_prompt(original: str, translation: str) -> str:
    """Layer 3 prompt (05 guide, section 4): the same translation judged for two different purposes."""
    return (
        "Judge this English translation of a Chinese classical poem for two different uses. Be strict.\n\n"
        "Original:\n%s\n\nTranslation:\n%s\n\n"
        "academic_translation: suitability for English readers with no Chinese background who want to understand "
        "the original. 5 excellent, they fully grasp meaning, imagery and emotion; 4 good; 3 acceptable, basic meaning only; "
        "2 poor, significant gaps; 1 unsuitable, they would misunderstand.\n"
        "poetic_reading: suitability for reading aloud as English poetry at a literary event. "
        "5 excellent rhythm and beauty; 4 good; 3 readable but not poetic; 2 awkward to read aloud; 1 unreadable as poetry.\n\n"
        'Reply with JSON only: {"academic_translation":n,"poetic_reading":n}'
    ) % (original, translation)


LAYER_WEIGHTS = {"layer1": 0.50, "layer2": 0.30, "layer3": 0.10, "layer4": 0.10}


def _unit_scale(x):
    return None if x is None or x != x else (x - 1) / 4.0


def composite(l1=None, l2=None, l3=None, l4=None):
    """Four-layer weighted score on 0-1 (05 guide: 50/30/10/10).

    l1 and l4 are already in 0-1; l2 and l3 are 1-5 means. A layer that does not apply to a
    translation (None or NaN) is dropped and the remaining weights are renormalised.
    """
    vals = {"layer1": l1, "layer2": _unit_scale(l2), "layer3": _unit_scale(l3), "layer4": l4}
    avail = {k: v for k, v in vals.items() if v is not None and v == v}
    if not avail:
        return None
    w = sum(LAYER_WEIGHTS[k] for k in avail)
    return sum(LAYER_WEIGHTS[k] * v for k, v in avail.items()) / w


class Evaluator:
    def __init__(self, scorer: ModelConfig = SCORER):
        self.client = OllamaModel(scorer)
        self.scorer_name = scorer.model_path

    def check_unit(self, unit: dict, translation: str, seed: int = 0) -> dict:
        resp = self.client.generate(unit_prompt(unit, translation), seed=seed, fmt="json")
        data = parse_json(resp["text"])
        if not data or "pass" not in data:
            return {"unit_id": unit["id"], "pass": None, "valid": False, "raw": resp["text"], "error": resp.get("error")}
        passed = data["pass"] is True or str(data["pass"]).lower() == "true"
        evidence = str(data.get("evidence", ""))
        found = bool(evidence) and _norm(evidence) in _norm(translation)
        # A pass claim must be backed by a verbatim quote, otherwise it counts as a fail.
        final = passed and found
        return {
            "unit_id": unit["id"],
            "pass": final,
            "claimed_pass": passed,
            "evidence": evidence,
            "evidence_found": found,
            "reasoning": data.get("reasoning", ""),
            "valid": True,
        }

    def score_suitability(self, original: str, translation: str, seed: int = 0) -> dict:
        resp = self.client.generate(suitability_prompt(original, translation), seed=seed, fmt="json")
        data = parse_json(resp["text"])
        out = {"valid": False, "raw": resp["text"]}
        if not data:
            return out
        try:
            s = {k: int(data[k]) for k in ("academic_translation", "poetic_reading")}
        except (KeyError, ValueError, TypeError):
            return out
        if any(not 1 <= v <= 5 for v in s.values()):
            return out
        s["valid"] = True
        return s

    def score_quality(self, original: str, translation: str, seed: int = 0, reference: str = "") -> dict:
        resp = self.client.generate(quality_prompt(original, translation, reference), seed=seed, fmt="json")
        data = parse_json(resp["text"])
        out = {"valid": False, "raw": resp["text"]}
        if not data:
            return out
        try:
            scores = {k: int(data[k]) for k in QUALITY_DIMS}
        except (KeyError, ValueError, TypeError):
            return out
        if any(not 1 <= v <= 5 for v in scores.values()):
            return out
        scores["overall"] = round(sum(scores[k] * QUALITY_WEIGHTS[k] for k in QUALITY_DIMS), 3)
        scores["valid"] = True
        return scores

    def evaluate(self, rec: dict, poem: dict, seed: int = 0) -> dict:
        """rec: generation record; poem: corpus record. Strategy is stored but never shown to the scorer."""
        text = strip_notes(rec["translation"])
        units = [self.check_unit(u, text, seed) for u in poem["semantic_units"]]
        valid = [u for u in units if u["valid"]]
        passed = sum(1 for u in valid if u["pass"])
        return {
            "translation_id": rec["id"],
            "anon_code": anon_code(rec["id"]),
            "poem_id": rec["poem_id"],
            "strategy": rec["strategy"],
            "model": rec["model"],
            "repeat": rec["repeat"],
            "scorer": self.scorer_name,
            "layer1": {
                "total_units": len(units),
                "valid_units": len(valid),
                "passed_units": passed,
                "accuracy": passed / len(valid) if valid else None,
                "details": units,
            },
            "layer2": self.score_quality(poem["text"], text, seed, build_semantic_analysis(poem)),
            "layer3": self.score_suitability(poem["text"], text, seed),
        }
