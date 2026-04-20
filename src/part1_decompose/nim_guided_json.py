"""Direct NIM (OpenAI-compatible) caller with XGrammar `guided_json` for interactive
sociolinguistic decomposition. Use this during prompt tuning / debugging. For batch
dataset generation, use `data_designer_runner.py` instead."""

from __future__ import annotations

import argparse
import json
import os

from dotenv import load_dotenv
from openai import OpenAI

from src.part1_decompose.prompts import SYSTEM_PROMPT, USER_TEMPLATE
from src.part2_cultural.tools.dict_lookup import lookup as _dict_lookup
from src.schemas import Decomposed

_ALLOWED_REF_TYPES = {"holiday","brand","service","event","food","pop_culture","slang","other"}

# Map free-form LLM outputs onto our strict enums. Things not in a map are kept as-is
# (if still invalid, they drop to the defaults below).
_ALLOWED_SPEECH_ACTS = {"complaint","brag","question","empathy_seeking","sarcasm","joke","statement","greeting","request"}
_SPEECH_ACT_ALIASES = {
    "expressive": "statement", "exclamation": "statement", "assertive": "statement",
    "directive": "request", "commissive": "statement", "declaration": "statement",
    "inform": "statement", "informative": "statement",
    "thanks": "greeting", "farewell": "greeting",
    "empathy": "empathy_seeking", "sympathy_seeking": "empathy_seeking",
    "praise": "brag", "boast": "brag",
}
_ALLOWED_EMOTIONS = {"joy","anger","sadness","fear","surprise","disgust","neutral"}
_EMOTION_ALIASES = {"happiness": "joy", "love": "joy", "excitement": "joy",
                    "frustration": "anger", "annoyance": "anger", "rage": "anger",
                    "grief": "sadness", "melancholy": "sadness",
                    "anxiety": "fear", "worry": "fear", "nervousness": "fear",
                    "amazement": "surprise", "shock": "surprise"}
_ALLOWED_REGISTERS = {"intimate","casual","formal","public"}
_AGE_PATTERNS = [
    ("teen", ["teen", "13", "14", "15", "16", "17", "18", "19", "youth"]),
    ("20s", ["20", "college", "young adult"]),
    ("30s", ["30"]),
    ("40plus", ["40", "50", "60", "senior"]),
]
_ALLOWED_EMPHASIS = {"CAPS", "repetition", "punctuation", "emoji"}
_ALLOWED_LAUGHTER = {"lol", "lmao", "rofl", "haha", "none"}


def _normalize(data: dict, source_text: str) -> dict:
    data.setdefault("source_text", source_text)

    sa = str(data.get("speech_act", "")).lower().replace(" ", "_")
    sa = _SPEECH_ACT_ALIASES.get(sa, sa)
    data["speech_act"] = sa if sa in _ALLOWED_SPEECH_ACTS else "statement"

    reg = str(data.get("register", "casual")).lower()
    data["register"] = reg if reg in _ALLOWED_REGISTERS else "casual"

    em = data.get("emotion") or {}
    if isinstance(em, str):
        em = {"type": em, "intensity": 3}
    et = str(em.get("type", "")).lower()
    et = _EMOTION_ALIASES.get(et, et)
    em["type"] = et if et in _ALLOWED_EMOTIONS else "neutral"
    inten = em.get("intensity", 3)
    try:
        inten_i = int(round(float(inten)))
    except (TypeError, ValueError):
        inten_i = 3
    em["intensity"] = max(1, min(5, inten_i))
    data["emotion"] = em

    refs = data.get("cultural_refs") or []
    norm_refs = []
    for r in refs:
        if isinstance(r, str):
            entry = {"type": "other", "term": r}
        elif isinstance(r, dict) and "term" in r:
            entry = {"type": r.get("type", "other"), "term": r["term"]}
        else:
            continue
        if entry["type"] not in _ALLOWED_REF_TYPES:
            entry["type"] = "other"
        # If the term matches our cultural map, trust the map's type over the LLM's guess.
        hit = _dict_lookup(entry["term"])
        if hit and hit.get("type") in _ALLOWED_REF_TYPES:
            entry["type"] = hit["type"]
        norm_refs.append(entry)
    data["cultural_refs"] = norm_refs

    im = data.get("internet_markers") or {}
    if isinstance(im, str):
        im = {}
    laughter = str(im.get("laughter", "none")).lower()
    im["laughter"] = laughter if laughter in _ALLOWED_LAUGHTER else "none"
    emph = im.get("emphasis", [])
    if isinstance(emph, str):
        emph = [e.strip() for e in emph.replace("/", ",").split(",") if e.strip()]
    emph = [("CAPS" if e.lower() == "caps" else e) for e in emph]
    im["emphasis"] = [e for e in emph if e in _ALLOWED_EMPHASIS]
    im.setdefault("sarcasm_marker", False)
    data["internet_markers"] = im

    age_raw = str(data.get("estimated_age_group", "")).lower()
    age = "unknown"
    for target, markers in _AGE_PATTERNS:
        if any(m in age_raw for m in markers):
            age = target
            break
    if age_raw in {"teen", "20s", "30s", "40plus", "unknown"}:
        age = age_raw
    data["estimated_age_group"] = age

    allowed_platforms = {"twitter", "reddit", "instagram", "tiktok", "discord", "sms"}
    plats = data.get("platform_fit") or []
    if isinstance(plats, str):
        plats = [p.strip() for p in plats.split(",") if p.strip()]
    data["platform_fit"] = [p.lower() for p in plats if p and p.lower() in allowed_platforms]

    return data


def _client() -> OpenAI:
    load_dotenv()
    return OpenAI(
        base_url=os.environ.get("NVIDIA_API_BASE", "https://integrate.api.nvidia.com/v1"),
        api_key=os.environ["NVIDIA_API_KEY"],
    )


def decompose(text: str, *, model: str | None = None, temperature: float = 0.2) -> Decomposed:
    client = _client()
    model = model or os.environ.get("NEMOTRON_MODEL", "nvidia/nemotron-3-nano-30b-a3b")
    schema = Decomposed.model_json_schema()
    resp = client.chat.completions.create(
        model=model,
        messages=[
            {"role": "system", "content": SYSTEM_PROMPT},
            {"role": "user", "content": USER_TEMPLATE.format(text=text)},
        ],
        extra_body={"nvext": {"guided_json": schema}},
        temperature=temperature,
    )
    raw = resp.choices[0].message.content or "{}"
    data = json.loads(raw)
    data = _normalize(data, text)
    return Decomposed.model_validate(data)


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--text", required=True, help="English SNS text to annotate")
    ap.add_argument("--model", default=None)
    args = ap.parse_args()
    out = decompose(args.text, model=args.model)
    print(out.model_dump_json(indent=2))


if __name__ == "__main__":
    main()
