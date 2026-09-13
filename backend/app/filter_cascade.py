"""Epocrates-style cascading candidate filter: shape/color narrow the field,
imprint text is near-decisive, visual embedding similarity is the fallback and
tiebreaker.

This is deliberately NOT a hard elimination filter. Automatic shape/color
detection from a phone photo is imperfect (lighting, angle, partial occlusion),
so a wrong auto-detected shape must not permanently exclude the correct pill —
it should just be outweighed by strong signal on the other axes. Each stage
produces a soft score in [0, 1]; stages combine as a weighted sum with the
embedding similarity, so a single bad signal degrades gracefully instead of
producing a hard miss.

Cascade, in the order a pharmacist actually uses these cues:
  1. Shape match       (coarse; eliminates a large fraction of the database)
  2. Color match       (coarse; further narrows within the shape-consistent set)
  3. Imprint match     (near-decisive within the shape+color-consistent set)
  4. Embedding score    (visual prior/fallback + tiebreaker throughout)

Color can be auto-extracted from the photo (extract_dominant_color). Shape
currently has no reliable automatic detector in this codebase — pass it
through from a manual "what shape is it?" UI selection when available (this
mirrors how Epocrates' own pill identifier actually works: manual shape/color
entry, not automatic classification), and omit it (None) otherwise, in which
case that stage simply contributes zero rather than penalizing anything.
"""
from __future__ import annotations

import re
from collections import Counter
from dataclasses import dataclass
from difflib import SequenceMatcher
from typing import Optional

from PIL import Image

_TOKEN_RE = re.compile(r"[A-Z0-9]+")

# RxNav SPLSHAPE / SPLCOLOR values are free-text-ish but drawn from a fairly
# small controlled vocabulary. Canonicalize common synonyms so "ROUND" and
# "CIRCLE" (seen in different data sources) compare equal.
_SHAPE_SYNONYMS = {
    "CIRCLE": "ROUND",
    "CIRCULAR": "ROUND",
    "OVAL": "OVAL",
    "ELLIPSE": "OVAL",
    "CAPSULE": "CAPSULE",
    "OBLONG": "CAPSULE",
    "RECTANGLE": "RECTANGLE",
    "SQUARE": "SQUARE",
    "DIAMOND": "DIAMOND",
    "TRIANGLE": "TRIANGLE",
    "PENTAGON": "PENTAGON",
    "HEXAGON": "HEXAGON",
    "OCTAGON": "OCTAGON",
}

# A small set of named colors with approximate RGB anchors, used to map a
# photo's dominant color to the same vocabulary RxNav's COLORTEXT uses.
_NAMED_COLORS: dict[str, tuple[int, int, int]] = {
    "WHITE": (255, 255, 255),
    "OFF-WHITE": (245, 240, 230),
    "YELLOW": (255, 220, 60),
    "ORANGE": (255, 140, 40),
    "PINK": (255, 170, 190),
    "RED": (210, 40, 40),
    "BROWN": (120, 80, 50),
    "GREEN": (60, 150, 80),
    "BLUE": (50, 90, 190),
    "PURPLE": (130, 70, 160),
    "GRAY": (140, 140, 140),
    "BLACK": (25, 25, 25),
    "TAN": (200, 175, 135),
    "MAROON": (110, 30, 40),
    "TURQUOISE": (60, 180, 175),
}


def _normalize_tokens(text: str) -> set[str]:
    if not text:
        return set()
    return set(_TOKEN_RE.findall(text.upper()))


def _canonical_shape(shape: Optional[str]) -> Optional[str]:
    if not shape:
        return None
    key = shape.strip().upper()
    return _SHAPE_SYNONYMS.get(key, key)


@dataclass
class CascadeScores:
    shape_score: float = 0.0
    color_score: float = 0.0
    imprint_score: float = 0.0
    embedding_score: float = 0.0
    fused_score: float = 0.0
    ocr_text: Optional[str] = None
    query_color: Optional[str] = None


@dataclass
class CascadeWeights:
    """How many points of embedding-cosine-similarity each matching signal is
    worth. Shape/color are coarse (small weight, mainly useful for breaking
    ties and demoting implausible candidates); imprint is the strongest signal
    once legible, matching how a pharmacist actually resolves an unknown pill.
    """

    shape: float = 0.10
    color: float = 0.10
    imprint: float = 0.30


def extract_dominant_color(image: Image.Image, sample_size: int = 64) -> Optional[str]:
    """Best-effort dominant-color read, mapped to RxNav's named-color
    vocabulary. Downsamples and takes the modal quantized color, which is
    robust enough for a roughly-cropped pill photo without needing a
    foreground/background segmentation model.
    """
    try:
        small = image.convert("RGB").resize((sample_size, sample_size))
    except Exception:
        return None
    pixels = list(small.getdata())
    if not pixels:
        return None
    # Quantize to reduce noise, then take the most common bucket.
    quantized = Counter((r // 24 * 24, g // 24 * 24, b // 24 * 24) for r, g, b in pixels)
    dominant_rgb = quantized.most_common(1)[0][0]

    best_name, best_dist = None, float("inf")
    for name, rgb in _NAMED_COLORS.items():
        dist = sum((a - b) ** 2 for a, b in zip(dominant_rgb, rgb))
        if dist < best_dist:
            best_name, best_dist = name, dist
    return best_name


def score_shape(query_shape: Optional[str], candidate_shape: Optional[str]) -> float:
    q, c = _canonical_shape(query_shape), _canonical_shape(candidate_shape)
    if not q or not c:
        return 0.0
    return 1.0 if q == c else 0.0


def score_color(query_color: Optional[str], candidate_colortext: Optional[str]) -> float:
    """candidate_colortext is RxNav COLORTEXT, e.g. "white(opaque white),blue"."""
    if not query_color or not candidate_colortext:
        return 0.0
    candidate_names = _normalize_tokens(candidate_colortext.replace("(", " ").replace(")", " "))
    return 1.0 if query_color.upper() in candidate_names else 0.0


def score_imprint(ocr_text: Optional[str], candidate_imprint: Optional[str]) -> float:
    """Fuzzy token-overlap match between OCR'd query text and RxNav's
    semicolon-separated IMPRINT_CODE, e.g. "LILLY;3229;40;mg"."""
    if not ocr_text or not candidate_imprint:
        return 0.0
    query_tokens = _normalize_tokens(ocr_text)
    cand_tokens = _normalize_tokens(candidate_imprint.replace(";", " "))
    if not query_tokens or not cand_tokens:
        return 0.0
    overlap = query_tokens & cand_tokens
    jaccard = len(overlap) / len(query_tokens | cand_tokens)
    best_ratio = max(
        (SequenceMatcher(None, qt, ct).ratio() for qt in query_tokens for ct in cand_tokens),
        default=0.0,
    )
    return max(0.0, min(1.0, 0.6 * jaccard + 0.4 * best_ratio))


@dataclass
class CandidateAttributes:
    """A retrieval candidate's known structured metadata (from RxNav)."""

    imprint: Optional[str] = None
    color: Optional[str] = None
    shape: Optional[str] = None


DEFAULT_WEIGHTS = CascadeWeights()


def fuse(
    embedding_score: float,
    candidate: CandidateAttributes,
    ocr_text: Optional[str],
    query_color: Optional[str],
    query_shape: Optional[str] = None,
    weights: Optional[CascadeWeights] = None,
) -> CascadeScores:
    weights = weights or DEFAULT_WEIGHTS
    shape_s = score_shape(query_shape, candidate.shape)
    color_s = score_color(query_color, candidate.color)
    imprint_s = score_imprint(ocr_text, candidate.imprint)

    fused = (
        embedding_score
        + weights.shape * shape_s
        + weights.color * color_s
        + weights.imprint * imprint_s
    )
    return CascadeScores(
        shape_score=shape_s,
        color_score=color_s,
        imprint_score=imprint_s,
        embedding_score=embedding_score,
        fused_score=fused,
        ocr_text=ocr_text,
        query_color=query_color,
    )
