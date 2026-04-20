from typing import Literal

from pydantic import BaseModel, Field

SpeechAct = Literal[
    "complaint",
    "brag",
    "question",
    "empathy_seeking",
    "sarcasm",
    "joke",
    "statement",
    "greeting",
    "request",
]
Register = Literal["intimate", "casual", "formal", "public"]
EmotionType = Literal["joy", "anger", "sadness", "fear", "surprise", "disgust", "neutral"]
CulturalRefType = Literal[
    "holiday", "brand", "service", "event", "food", "pop_culture", "slang", "other"
]
Laughter = Literal["lol", "lmao", "rofl", "haha", "none"]
Emphasis = Literal["CAPS", "repetition", "punctuation", "emoji"]
AgeGroup = Literal["teen", "20s", "30s", "40plus", "unknown"]
Platform = Literal["twitter", "reddit", "instagram", "tiktok", "discord", "sms"]
MapSource = Literal["dict", "retriever", "web+llm"]


class Emotion(BaseModel):
    type: EmotionType
    intensity: int = Field(ge=1, le=5)


class CulturalRef(BaseModel):
    type: CulturalRefType
    term: str


class InternetMarkers(BaseModel):
    laughter: Laughter = "none"
    emphasis: list[Emphasis] = Field(default_factory=list)
    sarcasm_marker: bool = False


class Decomposed(BaseModel):
    source_text: str
    speech_act: SpeechAct
    register: Register
    emotion: Emotion
    cultural_refs: list[CulturalRef] = Field(default_factory=list)
    internet_markers: InternetMarkers
    estimated_age_group: AgeGroup
    platform_fit: list[Platform] = Field(default_factory=list)


class MappedRef(BaseModel):
    term: str
    ko: str
    type: str
    source: MapSource
    retrieved: bool = False
    notes: str = ""


class Stage12Output(BaseModel):
    source_text: str
    decomposed: Decomposed
    mapped_refs: list[MappedRef] = Field(default_factory=list)
