#!/usr/bin/env python3
"""
scripts/align_audio.py — Blocks 2.1:1 (Audio Sync) & 2.5:2 (Final Assembly)

Runs in the outer Python environment with two subcommands:
  - align    (2.1:1): Seeded faster-whisper transcription + fuzzy sequential
                      word-stream alignment, per-visual_cue trigger timestamps,
                      rough_segment_id grouping, visual hold / breath-pause
                      snapping, and contiguous integer frame math.
  - assemble (2.5:2): ffprobe verification of rendered rough-segment clips,
                      FFmpeg concat-demuxer stitching + narration muxing +
                      abrupt_end hard trim, 2-3 word .srt captions, and post
                      Foley/SFX cue sheet export.

Dependencies:
  - faster-whisper (with PyAV pinned to av<19)
  - rapidfuzz (falls back cleanly to difflib.SequenceMatcher if absent)
  - ffmpeg / ffprobe on PATH (required for assemble; optional duration probe in align)

Examples:
  python scripts/align_audio.py align \
      --storyboard batches/batch_001/ep_01_souffle/0_input/storyboard.json \
      --audio batches/batch_001/ep_01_souffle/0_input/narration.mp3 \
      --raw-out batches/batch_001/ep_01_souffle/1_manifests/whisper_raw.json \
      --out batches/batch_001/ep_01_souffle/1_manifests/timed_storyboard.json \
      --fps 30

  python scripts/align_audio.py assemble \
      --manifest batches/batch_001/ep_01_souffle/1_manifests/timed_storyboard.json \
      --audio batches/batch_001/ep_01_souffle/0_input/narration.mp3 \
      --clips-dir batches/batch_001/ep_01_souffle/4_renders \
      --out-dir batches/batch_001/ep_01_souffle/5_output \
      --ep ep_01_souffle
"""

from __future__ import annotations

import argparse
import copy
import difflib
import json
import logging
import re
import shutil
import subprocess
import sys
import tempfile
import unicodedata
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any

try:
    from rapidfuzz import fuzz as _rapidfuzz_fuzz  # type: ignore

    HAS_RAPIDFUZZ = True
except ImportError:
    _rapidfuzz_fuzz = None
    HAS_RAPIDFUZZ = False


logger = logging.getLogger("align_audio")

LOW_CONFIDENCE_THRESHOLD = 0.65

STOPWORDS = {
    "a",
    "an",
    "the",
    "of",
    "in",
    "to",
    "is",
    "it",
    "its",
    "and",
    "or",
    "on",
    "at",
    "by",
    "as",
    "so",
    "for",
    "be",
    "if",
    "up",
    "my",
    "we",
    "he",
    "she",
    "do",
    "no",
    "oh",
    "uh",
    "um",
}

EQUIVALENT_WORD_PAIRS = {
    frozenset(("mr", "mister")),
    frozenset(("mrs", "missus")),
    frozenset(("dr", "doctor")),
    frozenset(("vs", "versus")),
    frozenset(("ok", "okay")),
    frozenset(("gonna", "goingto")),
    frozenset(("wanna", "wantto")),
    frozenset(("gotta", "gotto")),
}

_ONES = [
    "zero",
    "one",
    "two",
    "three",
    "four",
    "five",
    "six",
    "seven",
    "eight",
    "nine",
    "ten",
    "eleven",
    "twelve",
    "thirteen",
    "fourteen",
    "fifteen",
    "sixteen",
    "seventeen",
    "eighteen",
    "nineteen",
]
_TENS = [
    "",
    "",
    "twenty",
    "thirty",
    "forty",
    "fifty",
    "sixty",
    "seventy",
    "eighty",
    "ninety",
]
_ORDINAL_ONES = {
    1: "first",
    2: "second",
    3: "third",
    4: "fourth",
    5: "fifth",
    6: "sixth",
    7: "seventh",
    8: "eighth",
    9: "ninth",
    10: "tenth",
    11: "eleventh",
    12: "twelfth",
    13: "thirteenth",
    14: "fourteenth",
    15: "fifteenth",
    16: "sixteenth",
    17: "seventeenth",
    18: "eighteenth",
    19: "nineteenth",
}
_ORDINAL_TENS = {
    2: "twentieth",
    3: "thirtieth",
    4: "fortieth",
    5: "fiftieth",
    6: "sixtieth",
    7: "seventieth",
    8: "eightieth",
    9: "ninetieth",
}

_SPLIT_DELIM_RE = re.compile(r"[-–—/]+")
_INT_RE = re.compile(r"^\d+$")
_ORD_RE = re.compile(r"^(\d+)(?:st|nd|rd|th)$", re.IGNORECASE)
_DEC_RE = re.compile(r"^(\d+)\.(\d+)$")
_PCT_RE = re.compile(r"^(\d+(?:\.\d+)?)%$")


# ---------------------------------------------------------------------------
# Data Structures
# ---------------------------------------------------------------------------


@dataclass
class ScriptWord:
    """A canonical word belonging to a storyboard beat."""

    word_idx: int
    beat_idx: int
    display: str
    raw_token: str
    norm_units: list[str]
    join_hyphen_prev: bool = False
    ends_sentence: bool = False
    ends_clause: bool = False
    start_s: float = 0.0
    end_s: float = 0.0
    interpolated: bool = True
    similarity: float = 0.0
    whisper_word: str = ""


@dataclass
class ScriptAlignUnit:
    """Single normalized unit in the sequential alignment stream."""

    unit_idx: int
    script_word_idx: int
    beat_idx: int
    norm: str
    start_s: float = 0.0
    end_s: float = 0.0
    matched: bool = False
    similarity: float = 0.0
    matched_whisper_indices: list[int] = field(default_factory=list)


@dataclass
class WhisperUnit:
    """Normalized Whisper heard word (split on hyphens/numbers if needed)."""

    unit_idx: int
    raw_word: str
    norm: str
    start_s: float
    end_s: float
    probability: float


# ---------------------------------------------------------------------------
# Normalization & Number Expansion
# ---------------------------------------------------------------------------


def _int_to_words(n: int) -> list[str]:
    if n < 0:
        return ["minus"] + _int_to_words(-n)
    if n < 20:
        return [_ONES[n]]
    if n < 100:
        tens, rem = divmod(n, 10)
        return [_TENS[tens]] + ([_ONES[rem]] if rem else [])
    if n < 1000:
        hundreds, rem = divmod(n, 100)
        words = [_ONES[hundreds], "hundred"]
        if rem:
            words.extend(_int_to_words(rem))
        return words
    if n < 1_000_000:
        thousands, rem = divmod(n, 1000)
        words = _int_to_words(thousands) + ["thousand"]
        if rem:
            words.extend(_int_to_words(rem))
        return words
    return [str(n)]


def _ordinal_to_words(n: int) -> list[str]:
    if n in _ORDINAL_ONES:
        return [_ORDINAL_ONES[n]]
    if n < 100:
        tens, rem = divmod(n, 10)
        if rem == 0 and tens in _ORDINAL_TENS:
            return [_ORDINAL_TENS[tens]]
        if tens >= 2 and rem in _ORDINAL_ONES:
            return [_TENS[tens], _ORDINAL_ONES[rem]]
    words = _int_to_words(n)
    if words[-1] == "hundred":
        words[-1] = "hundredth"
    elif words[-1] == "thousand":
        words[-1] = "thousandth"
    else:
        words[-1] = words[-1] + "th"
    return words


def _strip_accents(text: str) -> str:
    nkfd = unicodedata.normalize("NFKD", text)
    return "".join(ch for ch in nkfd if not unicodedata.combining(ch))


def _normalize_atom(atom: str) -> list[str]:
    """Normalize a single non-hyphenated token into canonical matching units."""
    cleaned = _strip_accents(atom).lower().strip()
    if not cleaned:
        return []

    pct_m = _PCT_RE.match(cleaned)
    if pct_m:
        return _normalize_atom(pct_m.group(1)) + ["percent"]

    cleaned = cleaned.strip("$€£¥\"'“”‘’()[]{}.,!?;:~*_")
    if not cleaned:
        return []

    ord_m = _ORD_RE.match(cleaned)
    if ord_m:
        val = int(ord_m.group(1))
        if val <= 9999:
            return _ordinal_to_words(val)

    dec_m = _DEC_RE.match(cleaned)
    if dec_m:
        whole = int(dec_m.group(1))
        frac = dec_m.group(2)
        if whole <= 9999 and len(frac) <= 4:
            frac_words = [_ONES[int(d)] for d in frac]
            return _int_to_words(whole) + ["point"] + frac_words

    if _INT_RE.match(cleaned):
        val = int(cleaned)
        if val <= 999_999:
            return _int_to_words(val)
        return [cleaned]

    alnum = re.sub(r"[^a-z0-9]", "", cleaned)
    return [alnum] if alnum else []


def _split_surface_parts(raw_chunk: str) -> list[tuple[str, bool]]:
    """Split a whitespace token on hyphens/slashes/dashes."""
    parts: list[tuple[str, bool]] = []
    tokens = re.split(r"([-–—/]+)", raw_chunk)
    pending_hyphen = False
    for tok in tokens:
        if not tok:
            continue
        if _SPLIT_DELIM_RE.fullmatch(tok):
            pending_hyphen = tok == "-"
            continue
        parts.append((tok, pending_hyphen and len(parts) > 0))
        pending_hyphen = False
    return parts


def _clean_display_word(raw_sub: str) -> tuple[str, bool, bool]:
    """Produce clean caption token while tracking sentence/clause endings."""
    stripped = raw_sub.strip()
    ends_sentence = any(ch in stripped for ch in ".!?…")
    ends_clause = ends_sentence or any(ch in stripped for ch in ",;:")

    disp = re.sub(r"^[\"“”‘’'()\[\]{}.,;:—–\-…]+", "", stripped)
    disp = re.sub(r"[\"“”‘’()\[\]{}.,;:—–\-…]+$", "", disp)
    if not disp:
        disp = re.sub(r"[^\w!?']", "", stripped)
    return disp, ends_sentence, ends_clause


def _tokenize_script_beats(
    beats: list[dict[str, Any]],
) -> tuple[list[ScriptWord], list[ScriptAlignUnit]]:
    """Tokenize all beats[].vo_line into canonical ScriptWords and ScriptAlignUnits."""
    script_words: list[ScriptWord] = []
    align_units: list[ScriptAlignUnit] = []

    for beat_idx, beat in enumerate(beats):
        vo_line = str(beat.get("vo_line") or "").strip()
        if not vo_line:
            continue
        for raw_chunk in vo_line.split():
            for sub_raw, join_hyphen in _split_surface_parts(raw_chunk):
                norm_list = _normalize_atom(sub_raw)
                disp, ends_sentence, ends_clause = _clean_display_word(sub_raw)
                if not norm_list and not disp:
                    continue
                if not disp:
                    disp = sub_raw.strip()
                word_idx = len(script_words)
                sw = ScriptWord(
                    word_idx=word_idx,
                    beat_idx=beat_idx,
                    display=disp,
                    raw_token=sub_raw,
                    norm_units=norm_list if norm_list else [re.sub(r"\W+", "", disp.lower()) or disp.lower()],
                    join_hyphen_prev=join_hyphen,
                    ends_sentence=ends_sentence,
                    ends_clause=ends_clause,
                )
                script_words.append(sw)
                for n_tok in sw.norm_units:
                    align_units.append(
                        ScriptAlignUnit(
                            unit_idx=len(align_units),
                            script_word_idx=word_idx,
                            beat_idx=beat_idx,
                            norm=n_tok,
                        )
                    )

    return script_words, align_units


def _tokenize_whisper_words(raw_words: list[dict[str, Any]]) -> list[WhisperUnit]:
    """Expand Whisper raw word tokens into normalized WhisperUnits with sub-timestamps."""
    units: list[WhisperUnit] = []
    for item in raw_words:
        raw_text = str(item.get("word") or item.get("text") or "").strip()
        if not raw_text:
            continue
        start_s = max(0.0, float(item.get("start", 0.0)))
        end_s = max(start_s, float(item.get("end", start_s)))
        prob = float(item.get("probability", 1.0))

        sub_norm_items: list[tuple[str, str]] = []
        for sub_raw, _ in _split_surface_parts(raw_text):
            for n in _normalize_atom(sub_raw):
                sub_norm_items.append((sub_raw, n))

        if not sub_norm_items:
            continue

        total_chars = sum(max(1, len(n)) for _, n in sub_norm_items)
        dur = max(0.0, end_s - start_s)
        cursor = start_s
        for idx, (sub_raw, norm) in enumerate(sub_norm_items):
            if idx == len(sub_norm_items) - 1:
                sub_end = end_s
            else:
                frac = max(1, len(norm)) / total_chars
                sub_end = cursor + dur * frac
            units.append(
                WhisperUnit(
                    unit_idx=len(units),
                    raw_word=sub_raw,
                    norm=norm,
                    start_s=cursor,
                    end_s=max(cursor, sub_end),
                    probability=prob,
                )
            )
            cursor = sub_end

    return units


# ---------------------------------------------------------------------------
# Fuzzy Similarity & Sequential Word-Stream Alignment
# ---------------------------------------------------------------------------


def _phonetic_key(word: str) -> str:
    """Compact English consonant skeleton for noisy Pokémon/culinary matching."""
    w = word.lower()
    if len(w) <= 3:
        return w
    for src, dst in (
        ("ph", "f"),
        ("qu", "kw"),
        ("ck", "k"),
        ("dg", "j"),
        ("kn", "n"),
        ("wr", "r"),
        ("gh", "g"),
        ("v", "f"),
        ("w", "f"),
        ("z", "s"),
        ("c", "k"),
    ):
        w = w.replace(src, dst)
    dedup = [w[0]]
    for ch in w[1:]:
        if ch != dedup[-1]:
            dedup.append(ch)
    head = dedup[0]
    tail = [ch for ch in dedup[1:] if ch not in "aeiouy"]
    key = head + "".join(tail)
    return key if len(key) >= 2 else "".join(dedup)


def _raw_ratio(a: str, b: str) -> float:
    if HAS_RAPIDFUZZ:
        return float(_rapidfuzz_fuzz.ratio(a, b)) / 100.0
    return difflib.SequenceMatcher(None, a, b).ratio()


def _word_similarity(a: str, b: str, cache: dict[tuple[str, str], float]) -> float:
    if a == b:
        return 1.0
    key = (a, b) if a <= b else (b, a)
    cached = cache.get(key)
    if cached is not None:
        return cached

    if frozenset((a, b)) in EQUIVALENT_WORD_PAIRS:
        cache[key] = 1.0
        return 1.0

    char_sim = _raw_ratio(a, b)
    sim = char_sim
    if min(len(a), len(b)) >= 4:
        blended = 0.65 * char_sim + 0.35 * _raw_ratio(_phonetic_key(a), _phonetic_key(b))
        if blended > sim:
            sim = blended

    cache[key] = sim
    return sim


def _delete_penalty(norm_word: str) -> float:
    if norm_word in STOPWORDS or len(norm_word) <= 2:
        return -0.35
    return -0.70


def _align_word_streams(
    script_units: list[ScriptAlignUnit],
    whisper_units: list[WhisperUnit],
    audio_duration_s: float,
) -> tuple[list[ScriptAlignUnit], set[int]]:
    """
    Global sequential Needleman-Wunsch alignment between script_units and whisper_units.
    Supports 1-to-1 fuzzy matches, 2-to-1 merges, 1-to-2 / 1-to-3 splits, and
    gap interpolation for swallowed short words.
    """
    n = len(script_units)
    m = len(whisper_units)
    if n == 0:
        return script_units, set(range(m))

    if m == 0:
        span = audio_duration_s if audio_duration_s > 0.0 else 0.30 * n
        step = span / n
        for idx, u in enumerate(script_units):
            u.start_s = idx * step
            u.end_s = (idx + 1) * step
            u.matched = False
            u.similarity = 0.0
        return script_units, set()

    sim_cache: dict[tuple[str, str], float] = {}
    neg_inf = -1e12

    dp: list[list[float]] = [[neg_inf] * (m + 1) for _ in range(n + 1)]
    bp: list[list[tuple[int, int, float]]] = [[(0, 0, 0.0)] * (m + 1) for _ in range(n + 1)]

    dp[0][0] = 0.0
    for i in range(1, n + 1):
        dp[i][0] = dp[i - 1][0] + _delete_penalty(script_units[i - 1].norm)
        bp[i][0] = (1, 0, 0.0)
    for j in range(1, m + 1):
        dp[0][j] = dp[0][j - 1] - 0.60
        bp[0][j] = (0, 1, 0.0)

    ins_penalty = -0.60

    for i in range(1, n + 1):
        s_norm = script_units[i - 1].norm
        del_pen = _delete_penalty(s_norm)
        dp_i = dp[i]
        dp_im1 = dp[i - 1]
        bp_i = bp[i]
        s_prev_norm = script_units[i - 2].norm if i >= 2 else ""

        for j in range(1, m + 1):
            w_norm = whisper_units[j - 1].norm

            best_score = dp_im1[j] + del_pen
            best_step = (1, 0, 0.0)

            cand = dp_i[j - 1] + ins_penalty
            if cand > best_score:
                best_score = cand
                best_step = (0, 1, 0.0)

            sim_11 = _word_similarity(s_norm, w_norm, sim_cache)
            if sim_11 >= 0.40:
                score_11 = dp_im1[j - 1] + (2.0 if s_norm == w_norm else 2.0 * sim_11)
                if score_11 > best_score:
                    best_score = score_11
                    best_step = (1, 1, sim_11)

            if i >= 2:
                sim_21 = _word_similarity(s_prev_norm + s_norm, w_norm, sim_cache)
                if sim_21 >= 0.72:
                    score_21 = dp[i - 2][j - 1] + 3.4 * sim_21
                    if score_21 > best_score:
                        best_score = score_21
                        best_step = (2, 1, sim_21)

            if j >= 2 and len(s_norm) >= 4:
                sim_12 = _word_similarity(s_norm, whisper_units[j - 2].norm + w_norm, sim_cache)
                if sim_12 >= 0.68:
                    score_12 = dp_im1[j - 2] + 3.2 * sim_12
                    if score_12 > best_score:
                        best_score = score_12
                        best_step = (1, 2, sim_12)

            if j >= 3 and len(s_norm) >= 6:
                joined_w3 = whisper_units[j - 3].norm + whisper_units[j - 2].norm + w_norm
                sim_13 = _word_similarity(s_norm, joined_w3, sim_cache)
                if sim_13 >= 0.65:
                    score_13 = dp_im1[j - 3] + 3.5 * sim_13
                    if score_13 > best_score:
                        best_score = score_13
                        best_step = (1, 3, sim_13)

            dp_i[j] = best_score
            bp_i[j] = best_step

    ops: list[tuple[int, int, int, int, float]] = []
    i, j = n, m
    while i > 0 or j > 0:
        di, dj, sim = bp[i][j]
        if di == 0 and dj == 0:
            raise RuntimeError("DP backtrace stalled unexpectedly.")
        ops.append((i - di, i, j - dj, j, sim))
        i -= di
        j -= dj
    ops.reverse()

    unmatched_whisper: set[int] = set()
    for i0, i1, j0, j1, sim in ops:
        di = i1 - i0
        dj = j1 - j0
        if di == 0 and dj > 0:
            unmatched_whisper.update(range(j0, j1))
        elif di > 0 and dj == 0:
            for s_idx in range(i0, i1):
                script_units[s_idx].matched = False
                script_units[s_idx].similarity = 0.0
        elif di == 1 and dj >= 1:
            u = script_units[i0]
            u.matched = True
            u.similarity = sim
            u.matched_whisper_indices = list(range(j0, j1))
            u.start_s = whisper_units[j0].start_s
            u.end_s = max(whisper_units[j0].start_s, whisper_units[j1 - 1].end_s)
        elif di == 2 and dj == 1:
            w = whisper_units[j0]
            u0, u1 = script_units[i0], script_units[i0 + 1]
            len0 = max(1, len(u0.norm))
            len1 = max(1, len(u1.norm))
            mid = w.start_s + max(0.0, w.end_s - w.start_s) * (len0 / (len0 + len1))
            u0.matched = True
            u0.similarity = sim
            u0.matched_whisper_indices = [j0]
            u0.start_s = w.start_s
            u0.end_s = mid
            u1.matched = True
            u1.similarity = sim
            u1.matched_whisper_indices = [j0]
            u1.start_s = mid
            u1.end_s = w.end_s

    idx = 0
    while idx < n:
        if script_units[idx].matched:
            idx += 1
            continue
        run_start = idx
        while idx < n and not script_units[idx].matched:
            idx += 1
        run_end = idx - 1
        count = run_end - run_start + 1

        prev_w_idx = (
            script_units[run_start - 1].matched_whisper_indices[-1]
            if run_start > 0 and script_units[run_start - 1].matched_whisper_indices
            else -1
        )
        next_w_idx = (
            script_units[run_end + 1].matched_whisper_indices[0]
            if run_end + 1 < n and script_units[run_end + 1].matched_whisper_indices
            else m
        )

        intervening_w = [k for k in range(prev_w_idx + 1, next_w_idx) if k in unmatched_whisper]

        if intervening_w:
            left_t = whisper_units[intervening_w[0]].start_s
            right_t = max(left_t, whisper_units[intervening_w[-1]].end_s)
        else:
            if run_start > 0:
                left_t = script_units[run_start - 1].end_s
            else:
                first_next = whisper_units[next_w_idx].start_s if next_w_idx < m else 0.0
                left_t = max(0.0, first_next - 0.22 * count)

            if run_end + 1 < n:
                right_t = script_units[run_end + 1].start_s
            else:
                cap = max(audio_duration_s, left_t + 0.22 * count)
                right_t = min(cap, left_t + 0.25 * count)

            min_needed = 0.03 * count
            if right_t - left_t < min_needed:
                needed = min_needed - (right_t - left_t)
                if run_start > 0:
                    prev_u = script_units[run_start - 1]
                    prev_dur = prev_u.end_s - prev_u.start_s
                    steal_l = min(needed * 0.5, max(0.0, prev_dur - 0.04))
                    prev_u.end_s -= steal_l
                    left_t = prev_u.end_s
                    needed -= steal_l
                if run_end + 1 < n and needed > 0:
                    next_u = script_units[run_end + 1]
                    next_dur = next_u.end_s - next_u.start_s
                    steal_r = min(needed, max(0.0, next_dur - 0.04))
                    next_u.start_s += steal_r
                    right_t = next_u.start_s
                if right_t <= left_t:
                    right_t = left_t + min_needed

            gap = right_t - left_t
            if gap > 0.45 * count:
                same_prev = run_start > 0 and script_units[run_start].beat_idx == script_units[run_start - 1].beat_idx
                same_next = run_end + 1 < n and script_units[run_end].beat_idx == script_units[run_end + 1].beat_idx
                target_dur = min(gap, 0.22 * count)
                if same_prev and not same_next:
                    right_t = left_t + target_dur
                elif same_next and not same_prev:
                    left_t = right_t - target_dur
                else:
                    mid = 0.5 * (left_t + right_t)
                    left_t = mid - 0.5 * target_dur
                    right_t = mid + 0.5 * target_dur

        weights = [max(2, len(script_units[k].norm)) for k in range(run_start, run_end + 1)]
        total_w = sum(weights)
        cursor = left_t
        span = max(0.02 * count, right_t - left_t)
        for offset, k in enumerate(range(run_start, run_end + 1)):
            u = script_units[k]
            u.start_s = cursor
            if k == run_end:
                u.end_s = max(cursor + 0.02, left_t + span)
            else:
                u.end_s = cursor + span * (weights[offset] / total_w)
            cursor = u.end_s

    return script_units, unmatched_whisper


def _aggregate_script_words(
    script_words: list[ScriptWord],
    align_units: list[ScriptAlignUnit],
    whisper_units: list[WhisperUnit],
) -> None:
    """Fold aligned ScriptAlignUnits back onto ScriptWords and enforce monotonicity."""
    units_by_word: dict[int, list[ScriptAlignUnit]] = {}
    for u in align_units:
        units_by_word.setdefault(u.script_word_idx, []).append(u)

    prev_end = 0.0
    for sw in script_words:
        u_list = units_by_word.get(sw.word_idx, [])
        if not u_list:
            sw.start_s = prev_end
            sw.end_s = prev_end + 0.04
            sw.interpolated = True
            sw.similarity = 0.0
            prev_end = sw.end_s
            continue

        start_s = max(prev_end, min(u.start_s for u in u_list))
        end_s = max(start_s + 0.02, max(u.end_s for u in u_list))

        sw.interpolated = not any(u.matched for u in u_list)
        sw.similarity = sum(u.similarity for u in u_list) / len(u_list)
        sw.start_s = start_s
        sw.end_s = end_s

        w_indices: list[int] = []
        for u in u_list:
            for w_idx in u.matched_whisper_indices:
                if w_idx not in w_indices:
                    w_indices.append(w_idx)
        sw.whisper_word = " ".join(whisper_units[w_idx].raw_word for w_idx in w_indices)
        prev_end = sw.end_s


def _score_beat_confidences(
    beats: list[dict[str, Any]],
    script_words: list[ScriptWord],
    align_units: list[ScriptAlignUnit],
    whisper_units: list[WhisperUnit],
    unmatched_whisper: set[int],
) -> dict[int, float]:
    """Compute alignment_confidence in [0.0, 1.0] per beat index."""
    words_by_beat: dict[int, list[ScriptWord]] = {}
    for sw in script_words:
        words_by_beat.setdefault(sw.beat_idx, []).append(sw)

    units_by_beat: dict[int, list[ScriptAlignUnit]] = {}
    for u in align_units:
        units_by_beat.setdefault(u.beat_idx, []).append(u)

    confidences: dict[int, float] = {}
    for beat_idx, beat in enumerate(beats):
        b_words = words_by_beat.get(beat_idx, [])
        if not b_words:
            confidences[beat_idx] = 1.0
            continue

        any_matched = any(not sw.interpolated for sw in b_words)
        token_scores: list[float] = []
        for sw in b_words:
            if not sw.interpolated:
                token_scores.append(sw.similarity)
            else:
                is_minor = len(sw.norm_units) == 1 and (
                    sw.norm_units[0] in STOPWORDS or len(sw.norm_units[0]) <= 2
                )
                token_scores.append(0.55 if (is_minor and any_matched) else 0.0)

        recall_score = sum(token_scores) / len(b_words)

        matched_w_indices = [
            w_idx
            for u in units_by_beat.get(beat_idx, [])
            for w_idx in u.matched_whisper_indices
        ]
        if matched_w_indices:
            w_lo, w_hi = min(matched_w_indices), max(matched_w_indices)
            extra_insertions = sum(
                1 for k in range(w_lo, w_hi + 1) if k in unmatched_whisper
            )
            heard_snippet = " ".join(whisper_units[k].raw_word for k in range(w_lo, w_hi + 1))
        else:
            extra_insertions = 0
            heard_snippet = "<no matched words>"

        insertion_penalty = extra_insertions / max(1, len(b_words) + extra_insertions)
        conf = round(max(0.0, min(1.0, recall_score * (1.0 - 0.5 * insertion_penalty))), 4)
        confidences[beat_idx] = conf

        if conf < LOW_CONFIDENCE_THRESHOLD:
            beat_label = beat.get("beat_index", beat.get("beat_id", beat_idx + 1))
            logger.warning(
                "Low alignment_confidence (%.2f < %.2f) on beat %s! "
                "Expected: %r | Heard span: %r",
                conf,
                LOW_CONFIDENCE_THRESHOLD,
                beat_label,
                beat.get("vo_line", ""),
                heard_snippet,
            )

    return confidences


# ---------------------------------------------------------------------------
# Whisper Seeding & Transcription (Block 2.1:1)
# ---------------------------------------------------------------------------


def _extract_domain_terms(value: Any) -> list[str]:
    terms: list[str] = []
    if value is None:
        return terms
    if isinstance(value, str):
        stripped = value.strip()
        if stripped:
            terms.append(stripped)
    elif isinstance(value, (list, tuple)):
        for item in value:
            terms.extend(_extract_domain_terms(item))
    elif isinstance(value, dict):
        for key in ("name", "subject", "delayed_reveal", "reveal", "term", "title", "text", "value"):
            if key in value:
                terms.extend(_extract_domain_terms(value[key]))
                break
        else:
            for item in value.values():
                if isinstance(item, str):
                    terms.extend(_extract_domain_terms(item))
    return terms


def _build_initial_prompt(storyboard: dict[str, Any]) -> str:
    """Seed Whisper initial_prompt with subject, delayed_reveal, and vo_line text."""
    domain_terms: list[str] = []
    seen_lower: set[str] = set()

    for key in ("subject", "delayed_reveal", "keywords", "glossary"):
        for term in _extract_domain_terms(storyboard.get(key)):
            low = term.lower()
            if low not in seen_lower:
                seen_lower.add(low)
                domain_terms.append(term)

    vo_phrases: list[str] = []
    for beat in storyboard.get("beats", []):
        line = str(beat.get("vo_line") or "").strip()
        if line:
            vo_phrases.append(line)

    parts: list[str] = []
    if domain_terms:
        parts.append(", ".join(domain_terms) + ".")
    if vo_phrases:
        parts.append(" ".join(vo_phrases))

    return " ".join(parts).strip()


def _probe_audio_duration(audio_path: Path) -> float | None:
    if not shutil.which("ffprobe"):
        return None
    try:
        res = subprocess.run(
            [
                "ffprobe",
                "-v",
                "error",
                "-show_entries",
                "format=duration",
                "-of",
                "json",
                str(audio_path),
            ],
            capture_output=True,
            text=True,
            check=True,
        )
        data = json.loads(res.stdout)
        dur = data.get("format", {}).get("duration")
        return float(dur) if dur is not None else None
    except Exception:
        return None


def _check_av_version() -> None:
    try:
        import av  # type: ignore

        ver_str = str(getattr(av, "__version__", "0"))
        major = int(ver_str.split(".", 1)[0])
        if major >= 19:
            logger.warning(
                "Detected PyAV %s (>=19). This pipeline is validated with 'av<19'; "
                "pin 'av<19' if faster-whisper decoding encounters issues.",
                ver_str,
            )
    except Exception:
        pass


def _transcribe_with_whisper(
    audio_path: Path,
    initial_prompt: str,
    model_size: str = "small",
    device: str = "cpu",
    compute_type: str = "int8",
    language: str | None = "en",
) -> dict[str, Any]:
    _check_av_version()
    try:
        from faster_whisper import WhisperModel  # type: ignore
    except ImportError as exc:
        raise RuntimeError(
            "faster-whisper is required for 'align'. Install with: "
            "pip install 'faster-whisper' 'av<19' 'rapidfuzz'"
        ) from exc

    logger.info(
        "Loading WhisperModel(%r, device=%r, compute_type=%r)...",
        model_size,
        device,
        compute_type,
    )
    model = WhisperModel(model_size, device=device, compute_type=compute_type)

    transcribe_kwargs: dict[str, Any] = {
        "word_timestamps": True,
        "vad_filter": False,  # Preserve quiet opening consonants at 0.0s
        "initial_prompt": initial_prompt or None,
    }
    if language and language.lower() != "auto":
        transcribe_kwargs["language"] = language

    logger.info("Transcribing %s (vad_filter=False, word_timestamps=True)...", audio_path)
    segments_gen, info = model.transcribe(str(audio_path), **transcribe_kwargs)

    segments_out: list[dict[str, Any]] = []
    flat_words: list[dict[str, Any]] = []

    for seg_idx, seg in enumerate(segments_gen):
        seg_words: list[dict[str, Any]] = []
        for w in getattr(seg, "words", None) or []:
            w_dict = {
                "word": str(getattr(w, "word", "")),
                "text": str(getattr(w, "word", "")).strip(),
                "start": round(float(getattr(w, "start", 0.0)), 6),
                "end": round(float(getattr(w, "end", 0.0)), 6),
                "probability": round(float(getattr(w, "probability", 0.0)), 6),
            }
            seg_words.append(w_dict)
            flat_words.append(
                {
                    "index": len(flat_words),
                    "segment_id": seg_idx,
                    **w_dict,
                }
            )

        segments_out.append(
            {
                "id": seg_idx,
                "start": round(float(getattr(seg, "start", 0.0)), 6),
                "end": round(float(getattr(seg, "end", 0.0)), 6),
                "text": str(getattr(seg, "text", "")),
                "avg_logprob": round(float(getattr(seg, "avg_logprob", 0.0)), 6),
                "no_speech_prob": round(float(getattr(seg, "no_speech_prob", 0.0)), 6),
                "words": seg_words,
            }
        )

    probed_dur = _probe_audio_duration(audio_path)
    info_dur = float(getattr(info, "duration", 0.0) or 0.0)
    audio_duration_s = probed_dur if probed_dur is not None else info_dur

    return {
        "audio_path": str(audio_path),
        "audio_duration_s": round(audio_duration_s, 6),
        "model": model_size,
        "device": device,
        "compute_type": compute_type,
        "vad_filter": False,
        "initial_prompt": initial_prompt,
        "language": getattr(info, "language", language),
        "language_probability": round(float(getattr(info, "language_probability", 0.0) or 0.0), 6),
        "segments": segments_out,
        "words": flat_words,
    }


# ---------------------------------------------------------------------------
# Segment Grouping, Frame Math & visual_cues[].vo_trigger Resolution
# ---------------------------------------------------------------------------


def _get_beat_transition(beat: dict[str, Any], is_last: bool) -> str:
    """Read Psyche's nested outgoing_transition.cut_type (with flat fallbacks)."""
    out_trans = beat.get("outgoing_transition")
    if isinstance(out_trans, dict):
        ct = out_trans.get("cut_type")
        if isinstance(ct, str) and ct.strip():
            return ct.strip().lower()

    for key in ("cut_type", "transition", "cut", "cut_after", "transition_out"):
        val = beat.get(key)
        if isinstance(val, str) and val.strip():
            return val.strip().lower()
    return "abrupt_end" if is_last else "smooth"


def _extract_trigger_query(vo_trigger: Any) -> tuple[str | None, float | None]:
    if vo_trigger is None:
        return None, None
    if isinstance(vo_trigger, dict):
        offset_raw = vo_trigger.get("offset_s")
        offset_s = float(offset_raw) if offset_raw is not None else None
        for key in ("vo_trigger", "phrase", "word", "keyword", "text"):
            val = vo_trigger.get(key)
            if isinstance(val, str) and val.strip():
                return val.strip(), offset_s
        return None, offset_s

    raw = str(vo_trigger).strip()
    if not raw:
        return None, None

    quoted = re.findall(r"['\"“”‘’]([^'\"“”‘’]+)['\"“”‘’]", raw)
    if quoted:
        return quoted[0].strip(), None

    cleaned = re.sub(
        r"^(?:on\s+word|on|at|word|trigger|when\s+narrator\s+says)\s*:?\s+",
        "",
        raw,
        flags=re.IGNORECASE,
    ).strip()
    return cleaned or raw, None


def _resolve_cue_span(
    vo_trigger: Any,
    beat_words: list[ScriptWord],
    min_word_idx: int,
    beat_start_s: float,
    beat_end_s: float,
    spoken_start_s: float,
    beat_conf: float,
) -> tuple[float, float, int, float, str]:
    """
    Locate exact (start_s, end_s, next_min_word_idx, confidence, method) for a
    single cue's vo_trigger inside beat_words, advancing monotonically from min_word_idx.
    """
    query, explicit_offset = _extract_trigger_query(vo_trigger)

    if explicit_offset is not None:
        t0 = max(beat_start_s, min(beat_end_s, beat_start_s + explicit_offset))
        return t0, min(beat_end_s, t0 + 0.25), min_word_idx, 1.0, "explicit_offset"

    if not beat_words:
        return beat_start_s, beat_end_s, min_word_idx, beat_conf, "silent_beat_fallback"

    if not query:
        sw = beat_words[min(min_word_idx, len(beat_words) - 1)]
        return sw.start_s, sw.end_s, min(len(beat_words), min_word_idx + 1), beat_conf, "sequential_default"

    q_low = query.lower().strip()
    if q_low in {"start", "on_start", "beat_start", "immediate"}:
        return beat_start_s, beat_words[0].end_s, min_word_idx, 1.0, "explicit_start"
    if q_low in {"end", "on_end", "beat_end"}:
        return beat_end_s, beat_end_s, len(beat_words), 1.0, "explicit_end"

    # Build normalized atom stream with local index into beat_words
    beat_atoms: list[tuple[str, int]] = []
    for bw_idx in range(min_word_idx, len(beat_words)):
        for n in beat_words[bw_idx].norm_units:
            beat_atoms.append((n, bw_idx))

    # Retry from start of beat if sequential cursor clipped an exact match
    if not beat_atoms:
        beat_atoms = [(n, bw_idx) for bw_idx, sw in enumerate(beat_words) for n in sw.norm_units]

    q_norms: list[str] = []
    for chunk in query.split():
        for sub_raw, _ in _split_surface_parts(chunk):
            q_norms.extend(_normalize_atom(sub_raw))

    if not q_norms:
        sw = beat_words[min(min_word_idx, len(beat_words) - 1)]
        return sw.start_s, sw.end_s, min_word_idx, beat_conf, "fallback_empty_query"

    sim_cache: dict[tuple[str, str], float] = {}
    k = len(q_norms)

    def _search_in(atoms: list[tuple[str, int]]) -> tuple[float, int, int] | None:
        if k > len(atoms):
            return None
        best_score = -1.0
        best_span: tuple[int, int] | None = None
        for idx in range(len(atoms) - k + 1):
            sims = [
                _word_similarity(q_norms[offset], atoms[idx + offset][0], sim_cache)
                for offset in range(k)
            ]
            avg_sim = sum(sims) / k
            if avg_sim > best_score:
                best_score = avg_sim
                best_span = (atoms[idx][1], atoms[idx + k - 1][1])
        if best_span is not None and best_score >= 0.72:
            return best_score, best_span[0], best_span[1]
        return None

    hit = _search_in(beat_atoms)
    if hit is None and min_word_idx > 0:
        full_atoms = [(n, bw_idx) for bw_idx, sw in enumerate(beat_words) for n in sw.norm_units]
        hit = _search_in(full_atoms)

    if hit is not None:
        score, w_start_i, w_end_i = hit
        sw0 = beat_words[w_start_i]
        sw1 = beat_words[w_end_i]
        cue_words = beat_words[w_start_i : w_end_i + 1]
        matched_sims = [w.similarity if not w.interpolated else 0.5 for w in cue_words]
        cue_conf = round(max(0.0, min(1.0, score * (sum(matched_sims) / len(matched_sims)))), 4)
        return (
            sw0.start_s,
            max(sw0.start_s, sw1.end_s),
            min(len(beat_words), w_end_i + 1),
            cue_conf,
            "phrase" if k > 1 else "word",
        )

    # Keyword fallback inside beat
    content_q = [qn for qn in q_norms if qn not in STOPWORDS and len(qn) >= 3] or q_norms
    best_kw_score = -1.0
    best_kw_idx = -1
    for qn in content_q:
        for atom_norm, bw_idx in beat_atoms:
            sim = _word_similarity(qn, atom_norm, sim_cache)
            if sim > best_kw_score:
                best_kw_score = sim
                best_kw_idx = bw_idx

    if best_kw_idx >= 0 and best_kw_score >= 0.78:
        sw = beat_words[best_kw_idx]
        return sw.start_s, sw.end_s, min(len(beat_words), best_kw_idx + 1), round(best_kw_score * beat_conf, 4), "keyword_fallback"

    sw_fb = beat_words[min(min_word_idx, len(beat_words) - 1)]
    return sw_fb.start_s, sw_fb.end_s, min_word_idx, round(beat_conf * 0.5, 4), "beat_cursor_fallback"


def _compute_beat_timings_and_frames(
    beats: list[dict[str, Any]],
    script_words: list[ScriptWord],
    confidences: dict[int, float],
    fps: float,
) -> list[dict[str, Any]]:
    """
    Assign rough_segment_id, beat timings, contiguous integer frames, and
    per-cue trigger_start_s / trigger_end_s / trigger_frame values.
    """
    n_beats = len(beats)
    if n_beats == 0:
        return []

    words_by_beat: dict[int, list[ScriptWord]] = {i: [] for i in range(n_beats)}
    for sw in script_words:
        words_by_beat[sw.beat_idx].append(sw)

    spoken_starts: list[float | None] = []
    spoken_ends: list[float | None] = []
    explicit_holds: list[float | None] = []

    for i, beat in enumerate(beats):
        bw = words_by_beat[i]
        if bw:
            spoken_starts.append(bw[0].start_s)
            spoken_ends.append(bw[-1].end_s)
        else:
            spoken_starts.append(None)
            spoken_ends.append(None)

        dur_val = beat.get("duration_seconds")
        explicit_holds.append(float(dur_val) if dur_val is not None else None)

    start_times: list[float] = [0.0] * n_beats
    end_times: list[float] = [0.0] * n_beats

    for i in range(n_beats):
        s_t = 0.0 if i == 0 else end_times[i - 1]
        start_times[i] = s_t
        sp_end = spoken_ends[i] if spoken_ends[i] is not None else s_t
        hold_dur = explicit_holds[i]
        vis_hold = max(sp_end, s_t + hold_dur) if hold_dur is not None else sp_end

        if i < n_beats - 1:
            next_sp_start = spoken_starts[i + 1]
            if next_sp_start is not None:
                if next_sp_start >= vis_hold:
                    end_times[i] = next_sp_start
                else:
                    if hold_dur is not None and next_sp_start < s_t + hold_dur - 0.05:
                        logger.warning(
                            "Beat %s visual_hold_end_s (%.3fs) exceeds next beat's spoken_start_s (%.3fs); "
                            "snapping boundary to %.3fs to preserve VO sync.",
                            beats[i].get("beat_index", beats[i].get("beat_id", i + 1)),
                            vis_hold,
                            next_sp_start,
                            max(sp_end, next_sp_start),
                        )
                    end_times[i] = max(sp_end, next_sp_start)
            else:
                end_times[i] = max(s_t + (1.0 / fps), vis_hold)
        else:
            end_times[i] = max(s_t + (1.0 / fps), vis_hold)

    # Quantize cumulative beat boundaries to contiguous integer frames (0-indexed durations)
    # Using 1-based start_frame (1..N) matching Contract 1C (`start_frame: 1, end_frame: 126` -> 126 frames)
    boundary_frames = [0]
    for end_t in end_times:
        f = max(boundary_frames[-1] + 1, int(round(end_t * fps)))
        boundary_frames.append(f)

    rough_segment_id = 1
    enriched_beats: list[dict[str, Any]] = []

    for i, orig_beat in enumerate(beats):
        beat = copy.deepcopy(orig_beat)
        is_last = i == n_beats - 1
        cut_type = _get_beat_transition(beat, is_last=is_last)

        b_start_idx_0 = boundary_frames[i]
        b_end_idx_0 = boundary_frames[i + 1]
        frame_count = b_end_idx_0 - b_start_idx_0

        start_frame = b_start_idx_0 + 1  # 1-based inclusive start frame
        end_frame = b_end_idx_0          # 1-based inclusive end frame

        snapped_start_s = round(b_start_idx_0 / fps, 6)
        snapped_end_s = round(b_end_idx_0 / fps, 6)
        duration_s = round(frame_count / fps, 6)

        bw = words_by_beat[i]
        sp_start = bw[0].start_s if bw else snapped_start_s
        sp_end = bw[-1].end_s if bw else snapped_start_s
        vis_hold_s = (
            max(sp_end, snapped_start_s + explicit_holds[i])
            if explicit_holds[i] is not None
            else sp_end
        )
        beat_conf = confidences.get(i, 1.0)

        # Resolve every cue inside visual_cues[] sequentially
        raw_cues = beat.get("visual_cues")
        if isinstance(raw_cues, list) and raw_cues:
            enriched_cues: list[dict[str, Any]] = []
            cursor_word_idx = 0
            prev_trig_frame = start_frame

            for c_idx, raw_cue in enumerate(raw_cues):
                cue = copy.deepcopy(raw_cue) if isinstance(raw_cue, dict) else {"vo_trigger": str(raw_cue)}
                t0_s, t1_s, cursor_word_idx, cue_conf, method = _resolve_cue_span(
                    vo_trigger=cue.get("vo_trigger"),
                    beat_words=bw,
                    min_word_idx=cursor_word_idx,
                    beat_start_s=snapped_start_s,
                    beat_end_s=snapped_end_s,
                    spoken_start_s=sp_start,
                    beat_conf=beat_conf,
                )
                # First cue snaps cleanly to beat start if it begins on the first word
                if c_idx == 0 and t0_s - snapped_start_s < 0.08:
                    t0_s = snapped_start_s

                t0_s = max(snapped_start_s, min(snapped_end_s, t0_s))
                t1_s = max(t0_s, min(snapped_end_s, t1_s))

                trig_frame = int(round(t0_s * fps)) + 1
                trig_frame = max(prev_trig_frame, min(end_frame, trig_frame))
                prev_trig_frame = trig_frame

                cue["trigger_start_s"] = round(t0_s, 6)
                cue["trigger_end_s"] = round(t1_s, 6)
                cue["trigger_frame"] = trig_frame
                cue["alignment_confidence"] = cue_conf
                cue["trigger_match_method"] = method
                enriched_cues.append(cue)

            beat["visual_cues"] = enriched_cues
            beat_trigger_frame = enriched_cues[0]["trigger_frame"]
            beat_trigger_s = enriched_cues[0]["trigger_start_s"]
        else:
            # Legacy fallback if a beat omits visual_cues[] and uses flat beat.vo_trigger
            t0_s, _, _, _, _ = _resolve_cue_span(
                vo_trigger=beat.get("vo_trigger"),
                beat_words=bw,
                min_word_idx=0,
                beat_start_s=snapped_start_s,
                beat_end_s=snapped_end_s,
                spoken_start_s=sp_start,
                beat_conf=beat_conf,
            )
            beat_trigger_frame = max(start_frame, min(end_frame, int(round(t0_s * fps)) + 1))
            beat_trigger_s = round(t0_s, 6)

        if "beat_index" not in beat:
            beat["beat_index"] = beat.get("beat_id", i + 1)

        beat["rough_segment_id"] = rough_segment_id
        beat["cut_type"] = cut_type
        beat["start_s"] = snapped_start_s
        beat["end_s"] = snapped_end_s
        beat["duration_s"] = duration_s
        beat["spoken_start_s"] = round(sp_start, 6)
        beat["spoken_end_s"] = round(sp_end, 6)
        beat["visual_hold_end_s"] = round(vis_hold_s, 6)
        beat["start_frame"] = start_frame
        beat["end_frame"] = end_frame
        beat["frame_count"] = frame_count
        beat["trigger_s"] = beat_trigger_s
        beat["trigger_frame"] = beat_trigger_frame
        beat["alignment_confidence"] = beat_conf
        beat["words"] = [
            {
                "word": sw.display,
                "raw_token": sw.raw_token,
                "norm": " ".join(sw.norm_units),
                "start_s": round(sw.start_s, 6),
                "end_s": round(sw.end_s, 6),
                "interpolated": sw.interpolated,
                "similarity": round(sw.similarity, 4),
                "whisper_word": sw.whisper_word,
                "join_hyphen_prev": sw.join_hyphen_prev,
                "ends_sentence": sw.ends_sentence,
                "ends_clause": sw.ends_clause,
            }
            for sw in bw
        ]

        enriched_beats.append(beat)

        if cut_type in {"rough", "rough_cut"} and not is_last:
            rough_segment_id += 1

    return enriched_beats


# ---------------------------------------------------------------------------
# Subcommand A: align (2.1:1)
# ---------------------------------------------------------------------------


def _resolve_default_audio(audio_arg: str | None) -> Path:
    if audio_arg:
        p = Path(audio_arg)
        if not p.exists():
            raise FileNotFoundError(f"Narration audio file not found: {p}")
        return p
    for candidate in (Path("0_input/narration.mp3"), Path("0_input/narration.wav")):
        if candidate.exists():
            return candidate
    raise FileNotFoundError(
        "Could not find narration audio at 0_input/narration.mp3 or 0_input/narration.wav. "
        "Specify explicitly via --audio."
    )


def run_align(args: argparse.Namespace) -> int:
    storyboard_path = Path(args.storyboard)
    if not storyboard_path.exists():
        raise FileNotFoundError(f"Storyboard not found: {storyboard_path}")

    audio_path = _resolve_default_audio(args.audio)
    fps = float(args.fps)
    if fps <= 0:
        raise ValueError(f"--fps must be > 0, got {fps}")

    raw_out_path = Path(args.raw_out)
    timed_out_path = Path(args.out)
    raw_out_path.parent.mkdir(parents=True, exist_ok=True)
    timed_out_path.parent.mkdir(parents=True, exist_ok=True)

    with storyboard_path.open("r", encoding="utf-8") as f:
        storyboard = json.load(f)

    beats = storyboard.get("beats", [])
    if not isinstance(beats, list) or not beats:
        raise ValueError(f"Storyboard {storyboard_path} contains no 'beats' array.")

    initial_prompt = _build_initial_prompt(storyboard)
    logger.info("Whisper initial_prompt (%d chars): %s", len(initial_prompt), initial_prompt[:140])

    if args.reuse_raw and raw_out_path.exists():
        logger.info("Reusing existing raw transcription: %s", raw_out_path)
        with raw_out_path.open("r", encoding="utf-8") as f:
            whisper_raw = json.load(f)
    else:
        whisper_raw = _transcribe_with_whisper(
            audio_path=audio_path,
            initial_prompt=initial_prompt,
            model_size=args.model,
            device=args.device,
            compute_type=args.compute_type,
            language=args.language,
        )
        with raw_out_path.open("w", encoding="utf-8") as f:
            json.dump(whisper_raw, f, indent=2, ensure_ascii=False)
        logger.info("Wrote raw Whisper timestamps -> %s", raw_out_path)

    audio_duration_s = float(whisper_raw.get("audio_duration_s") or 0.0)
    raw_words = whisper_raw.get("words") or [
        w for seg in whisper_raw.get("segments", []) for w in seg.get("words", [])
    ]

    logger.info(
        "Aligning script words against %d Whisper words (matcher=%s)...",
        len(raw_words),
        "rapidfuzz" if HAS_RAPIDFUZZ else "difflib",
    )

    script_words, script_units = _tokenize_script_beats(beats)
    whisper_units = _tokenize_whisper_words(raw_words)

    aligned_units, unmatched_whisper = _align_word_streams(
        script_units=script_units,
        whisper_units=whisper_units,
        audio_duration_s=audio_duration_s,
    )
    _aggregate_script_words(script_words, aligned_units, whisper_units)
    confidences = _score_beat_confidences(
        beats=beats,
        script_words=script_words,
        align_units=aligned_units,
        whisper_units=whisper_units,
        unmatched_whisper=unmatched_whisper,
    )

    enriched_beats = _compute_beat_timings_and_frames(
        beats=beats,
        script_words=script_words,
        confidences=confidences,
        fps=fps,
    )

    fps_clean: int | float = int(fps) if fps.is_integer() else fps
    low_conf_beats = [
        b.get("beat_index")
        for b in enriched_beats
        if b["alignment_confidence"] < LOW_CONFIDENCE_THRESHOLD
    ]
    conf_vals = [b["alignment_confidence"] for b in enriched_beats]
    matched_count = sum(1 for sw in script_words if not sw.interpolated)

    timed_manifest = copy.deepcopy(storyboard)
    timed_manifest["fps"] = fps_clean
    timed_manifest["audio_file"] = str(audio_path)
    timed_manifest["audio_duration_s"] = round(audio_duration_s, 6)
    timed_manifest["total_duration_s"] = enriched_beats[-1]["end_s"]
    timed_manifest["total_frames"] = enriched_beats[-1]["end_frame"]
    timed_manifest["initial_prompt"] = initial_prompt
    timed_manifest["alignment_summary"] = {
        "matcher": "rapidfuzz" if HAS_RAPIDFUZZ else "difflib",
        "total_beats": len(enriched_beats),
        "rough_segments": enriched_beats[-1]["rough_segment_id"],
        "low_confidence_threshold": LOW_CONFIDENCE_THRESHOLD,
        "low_confidence_beats": low_conf_beats,
        "min_confidence": min(conf_vals) if conf_vals else 1.0,
        "mean_confidence": round(sum(conf_vals) / len(conf_vals), 4) if conf_vals else 1.0,
        "total_script_words": len(script_words),
        "matched_words": matched_count,
        "interpolated_words": len(script_words) - matched_count,
    }
    timed_manifest["beats"] = enriched_beats

    with timed_out_path.open("w", encoding="utf-8") as f:
        json.dump(timed_manifest, f, indent=2, ensure_ascii=False)

    logger.info(
        "Wrote timed storyboard -> %s | %d beats | %d rough segments | %d frames (%.3fs @ %s fps)",
        timed_out_path,
        len(enriched_beats),
        enriched_beats[-1]["rough_segment_id"],
        timed_manifest["total_frames"],
        timed_manifest["total_duration_s"],
        fps_clean,
    )
    if low_conf_beats:
        logger.warning(
            "Alignment review recommended for %d beat(s) below %.2f confidence: %s",
            len(low_conf_beats),
            LOW_CONFIDENCE_THRESHOLD,
            low_conf_beats,
        )
    return 0


# ---------------------------------------------------------------------------
# Subcommand B: assemble (2.5:2) — Verification, FFmpeg Mux, SRT & Cue Sheet
# ---------------------------------------------------------------------------


def _require_tool(name: str) -> str:
    path = shutil.which(name)
    if not path:
        raise RuntimeError(f"Required executable '{name}' not found on PATH.")
    return path


def _natural_sort_key(path: Path) -> list[Any]:
    return [int(part) if part.isdigit() else part.lower() for part in re.split(r"(\d+)", path.name)]


def _resolve_episode_id(manifest: dict[str, Any], cli_ep: str | None) -> str:
    raw = cli_ep
    if not raw:
        for key in ("episode_id", "episode", "ep", "id", "slug", "title", "subject"):
            val = manifest.get(key)
            if isinstance(val, str) and val.strip():
                raw = val.strip()
                break
    if not raw:
        return "episode"
    sanitized = re.sub(r"[^A-Za-z0-9._-]+", "_", raw).strip("_")
    return sanitized or "episode"


def _group_rough_segments(beats: list[dict[str, Any]]) -> list[dict[str, Any]]:
    """Group beats by rough_segment_id for 2.5:1 clip verification."""
    groups: dict[int, list[dict[str, Any]]] = {}
    for b in beats:
        seg_id = int(b.get("rough_segment_id", 1))
        groups.setdefault(seg_id, []).append(b)

    segments: list[dict[str, Any]] = []
    for seg_id in sorted(groups):
        seg_beats = groups[seg_id]
        start_f = int(seg_beats[0]["start_frame"])
        end_f = int(seg_beats[-1]["end_frame"])
        start_s = float(seg_beats[0]["start_s"])
        end_s = float(seg_beats[-1]["end_s"])
        segments.append(
            {
                "segment_id": seg_id,
                "beats": seg_beats,
                "start_frame": start_f,
                "end_frame": end_f,
                "frame_count": sum(int(b.get("frame_count", 0)) for b in seg_beats) or (end_f - start_f + 1),
                "duration_s": round(end_s - start_s, 6),
            }
        )
    return segments


def _resolve_unit_clip_path(
    clips_dir: Path,
    unit_num: int,
    idx: int,
    fallback_sorted_clips: list[Path],
    total_units: int,
    unit_dict: dict[str, Any],
) -> Path:
    for key in ("clip_path", "render_path", "video_path", "render_file"):
        val = unit_dict.get(key)
        if isinstance(val, str) and val.strip():
            cand = Path(val)
            if cand.exists():
                return cand
            rel = clips_dir / cand
            if rel.exists():
                return rel

    for pat in (f"clip_{unit_num:02d}.mp4", f"clip_{unit_num:03d}.mp4", f"clip_{unit_num}.mp4"):
        cand = clips_dir / pat
        if cand.exists():
            return cand

    if (clips_dir / "clip_00.mp4").exists():
        cand0 = clips_dir / f"clip_{idx:02d}.mp4"
        if cand0.exists():
            return cand0

    if len(fallback_sorted_clips) == total_units:
        return fallback_sorted_clips[idx]

    raise FileNotFoundError(
        f"Missing clip_{unit_num:02d}.mp4 in {clips_dir} "
        f"(expected {total_units} clip(s), found {len(fallback_sorted_clips)})."
    )


def _parse_rate(rate_str: str | None) -> float:
    if not rate_str or rate_str == "0/0":
        return 0.0
    if "/" in rate_str:
        num, den = rate_str.split("/", 1)
        den_f = float(den)
        return float(num) / den_f if den_f else 0.0
    return float(rate_str)


def _probe_video_clip(clip_path: Path) -> dict[str, Any]:
    cmd = [
        "ffprobe",
        "-v",
        "error",
        "-select_streams",
        "v:0",
        "-count_packets",
        "-show_entries",
        "stream=codec_name,width,height,pix_fmt,r_frame_rate,avg_frame_rate,nb_frames,nb_read_packets,duration",
        "-show_entries",
        "format=duration",
        "-of",
        "json",
        str(clip_path),
    ]
    res = subprocess.run(cmd, capture_output=True, text=True, check=True)
    data = json.loads(res.stdout)
    streams = data.get("streams") or []
    if not streams:
        raise RuntimeError(f"No video stream found in {clip_path}")
    st = streams[0]
    fmt = data.get("format") or {}

    fps_val = _parse_rate(st.get("avg_frame_rate")) or _parse_rate(st.get("r_frame_rate"))
    duration_s = float(st.get("duration") or fmt.get("duration") or 0.0)

    nb_frames_raw = st.get("nb_frames")
    if nb_frames_raw and str(nb_frames_raw).isdigit():
        frames = int(nb_frames_raw)
    elif st.get("nb_read_packets") and str(st["nb_read_packets"]).isdigit():
        frames = int(st["nb_read_packets"])
    else:
        cnt_res = subprocess.run(
            [
                "ffprobe",
                "-v",
                "error",
                "-select_streams",
                "v:0",
                "-count_frames",
                "-show_entries",
                "stream=nb_read_frames",
                "-of",
                "json",
                str(clip_path),
            ],
            capture_output=True,
            text=True,
            check=True,
        )
        cnt_data = json.loads(cnt_res.stdout)
        nb_read = (cnt_data.get("streams") or [{}])[0].get("nb_read_frames")
        frames = int(nb_read) if nb_read and str(nb_read).isdigit() else int(round(duration_s * fps_val))

    return {
        "path": clip_path,
        "codec": st.get("codec_name", ""),
        "width": int(st.get("width", 0)),
        "height": int(st.get("height", 0)),
        "pix_fmt": st.get("pix_fmt", ""),
        "fps": fps_val,
        "frames": frames,
        "duration_s": duration_s,
    }


def _verify_clips(
    beats: list[dict[str, Any]],
    clips_dir: Path,
    expected_fps: float,
    frame_tolerance: int,
    allow_mismatch: bool,
) -> list[dict[str, Any]]:
    """
    Verify rendered clips in `clips_dir`. Expects one clip per `rough_segment_id`
    (matching 2.3:1 / 2.5:1), falling back cleanly if clips were rendered per beat.
    """
    fallback_sorted = sorted(clips_dir.glob("clip_*.mp4"), key=_natural_sort_key)
    segments = _group_rough_segments(beats)

    use_per_beat = (
        len(fallback_sorted) == len(beats)
        and len(segments) != len(beats)
    )
    if use_per_beat:
        units = [
            {
                "unit_num": int(b.get("beat_index", b.get("beat_id", idx + 1))),
                "frame_count": int(b["frame_count"]),
                "duration_s": float(b["duration_s"]),
                "raw": b,
            }
            for idx, b in enumerate(beats)
        ]
        logger.info("Detected per-beat clips (%d clips for %d beats).", len(fallback_sorted), len(beats))
    else:
        units = [
            {
                "unit_num": seg["segment_id"],
                "frame_count": seg["frame_count"],
                "duration_s": seg["duration_s"],
                "raw": seg["beats"][0],
            }
            for seg in segments
        ]
        logger.info("Verifying %d rough-segment clip(s) against manifest...", len(units))

    verified: list[dict[str, Any]] = []
    ref_res: tuple[int, int] | None = None
    errors: list[str] = []

    for idx, unit in enumerate(units):
        clip_path = _resolve_unit_clip_path(
            clips_dir=clips_dir,
            unit_num=unit["unit_num"],
            idx=idx,
            fallback_sorted_clips=fallback_sorted,
            total_units=len(units),
            unit_dict=unit["raw"],
        )
        info = _probe_video_clip(clip_path)

        exp_frames = unit["frame_count"]
        exp_dur = unit["duration_s"]
        frame_diff = info["frames"] - exp_frames
        dur_diff = info["duration_s"] - exp_dur

        if ref_res is None:
            ref_res = (info["width"], info["height"])
        elif (info["width"], info["height"]) != ref_res:
            msg = (
                f"{clip_path.name} resolution {info['width']}x{info['height']} "
                f"differs from first clip {ref_res[0]}x{ref_res[1]}"
            )
            if allow_mismatch:
                logger.warning(msg)
            else:
                errors.append(msg)

        if abs(info["fps"] - expected_fps) > 0.05:
            logger.warning(
                "%s FPS %.3f differs from manifest FPS %.3f (will normalize on encode).",
                clip_path.name,
                info["fps"],
                expected_fps,
            )

        status = "OK"
        max_dur_tol = max(0.08, (frame_tolerance + 0.5) / expected_fps)
        if abs(frame_diff) > frame_tolerance or abs(dur_diff) > max_dur_tol:
            status = "MISMATCH"
            msg = (
                f"{clip_path.name}: expected {exp_frames}f ({exp_dur:.3f}s), "
                f"got {info['frames']}f ({info['duration_s']:.3f}s)"
            )
            if allow_mismatch:
                logger.warning("[VERIFY] %s (proceeding due to --allow-mismatch)", msg)
            else:
                errors.append(msg)
        elif frame_diff != 0:
            status = f"WARN({frame_diff:+d}f)"

        logger.info(
            "[VERIFY] %-14s | %dx%d @ %.2ffps | frames: %4d/%4d | dur: %6.3fs/%6.3fs | %s",
            clip_path.name,
            info["width"],
            info["height"],
            info["fps"],
            info["frames"],
            exp_frames,
            info["duration_s"],
            exp_dur,
            status,
        )
        verified.append(info)

    if errors:
        raise RuntimeError(
            "Clip verification failed:\n  - " + "\n  - ".join(errors) + "\nUse --allow-mismatch to override."
        )
    return verified


def _concatenate_and_mux(
    clip_infos: list[dict[str, Any]],
    audio_path: Path,
    out_mp4: Path,
    target_end_s: float,
    total_frames: int,
    fps: float,
    copy_video: bool = False,
) -> None:
    """
    Stitch clips via FFmpeg concat demuxer, mux narration audio, and hard-trim
    cleanly on the abrupt_end payoff frame with zero trailing black frames/silence.
    """
    fps_str = str(int(fps)) if float(fps).is_integer() else f"{fps:.6f}"
    needs_fps_norm = any(abs(c["fps"] - fps) > 0.05 for c in clip_infos)

    with tempfile.NamedTemporaryFile(
        mode="w", suffix="_concat.txt", encoding="utf-8", delete=False
    ) as tf:
        concat_file = Path(tf.name)
        for info in clip_infos:
            posix_path = info["path"].resolve().as_posix().replace("'", "'\\''")
            tf.write(f"file '{posix_path}'\n")

    try:
        cmd: list[str] = [
            "ffmpeg",
            "-y",
            "-hide_banner",
            "-loglevel",
            "warning",
            "-f",
            "concat",
            "-safe",
            "0",
            "-i",
            str(concat_file),
            "-i",
            str(audio_path),
            "-map",
            "0:v:0",
            "-map",
            "1:a:0",
            "-af",
            f"apad,atrim=0:{target_end_s:.6f},asetpts=PTS-STARTPTS",
        ]

        if copy_video:
            cmd.extend(["-c:v", "copy"])
        else:
            vf_parts: list[str] = []
            if needs_fps_norm:
                vf_parts.append(f"fps={fps_str}")
            vf_parts.append(f"trim=start_frame=0:end_frame={total_frames}")
            vf_parts.append("setpts=PTS-STARTPTS")
            cmd.extend(
                [
                    "-vf",
                    ",".join(vf_parts),
                    "-r",
                    fps_str,
                    "-c:v",
                    "libx264",
                    "-preset",
                    "medium",
                    "-crf",
                    "18",
                    "-pix_fmt",
                    "yuv420p",
                ]
            )

        cmd.extend(
            [
                "-c:a",
                "aac",
                "-b:a",
                "192k",
                "-t",
                f"{target_end_s:.6f}",
                "-movflags",
                "+faststart",
                str(out_mp4),
            ]
        )

        logger.info("Running FFmpeg assembly -> %s (hard trim @ %.3fs / %df)...", out_mp4, target_end_s, total_frames)
        subprocess.run(cmd, check=True)
    finally:
        concat_file.unlink(missing_ok=True)

    final_info = _probe_video_clip(out_mp4)
    logger.info(
        "Master ready: %s | %dx%d | %d frames | %.3fs",
        out_mp4,
        final_info["width"],
        final_info["height"],
        final_info["frames"],
        final_info["duration_s"],
    )


# ---------------------------------------------------------------------------
# Shorts-Style .srt Generation (Zack D.-style 2-3 word chunks)
# ---------------------------------------------------------------------------


def _format_srt_timestamp(seconds: float) -> str:
    ms_total = max(0, int(round(seconds * 1000)))
    hours, rem = divmod(ms_total, 3_600_000)
    minutes, rem = divmod(rem, 60_000)
    secs, millis = divmod(rem, 1000)
    return f"{hours:02d}:{minutes:02d}:{secs:02d},{millis:03d}"


def _join_chunk_words(words: list[dict[str, Any]]) -> str:
    out: list[str] = []
    for i, w in enumerate(words):
        txt = str(w.get("word") or "").strip()
        if not txt:
            continue
        if i > 0 and w.get("join_hyphen_prev") and out:
            out[-1] = f"{out[-1]}-{txt}"
        else:
            out.append(txt)
    return " ".join(out)


def _partition_run_into_chunks(run: list[dict[str, Any]]) -> list[list[dict[str, Any]]]:
    """Partition a phrase run into punchy 2-3 word Shorts caption chunks."""
    n = len(run)
    if n == 0:
        return []
    if n == 1:
        return [run]

    inf = 1e9
    dp = [inf] * (n + 1)
    nxt = [n] * (n + 1)
    dp[n] = 0.0

    for i in range(n - 1, -1, -1):
        for size in (1, 2, 3):
            j = i + size
            if j > n:
                continue
            chunk = run[i:j]
            text = _join_chunk_words(chunk)
            char_len = len(text)

            cost = 0.0
            if size == 1:
                cost += 12.0 if char_len < 16 else 2.5
                if chunk[-1].get("ends_sentence"):
                    cost -= 3.0
            elif size == 2:
                cost += 1.0
                if char_len > 22:
                    cost += 2.0
            else:
                cost += 1.2
                if char_len > 22:
                    cost += 6.0
                elif char_len <= 18:
                    cost -= 0.4

            if j < n and chunk[-1].get("ends_clause"):
                cost -= 0.8

            if j < n and run[j].get("join_hyphen_prev") and char_len < 20:
                cost += 8.0

            total = cost + dp[j]
            if total < dp[i]:
                dp[i] = total
                nxt[i] = j

    chunks: list[list[dict[str, Any]]] = []
    idx = 0
    while idx < n:
        j = nxt[idx]
        chunks.append(run[idx:j])
        idx = j
    return chunks


def _generate_shorts_srt(manifest: dict[str, Any], out_srt: Path) -> int:
    """Group aligned word timestamps into punchy 2-3 word chunks and write .srt."""
    beats = manifest.get("beats", [])
    final_end_s = float(manifest.get("total_duration_s") or beats[-1]["end_s"])

    runs: list[list[dict[str, Any]]] = []
    for beat in beats:
        b_words = [w for w in beat.get("words", []) if str(w.get("word") or "").strip()]
        if not b_words:
            continue
        cur_run: list[dict[str, Any]] = []
        for w in b_words:
            if cur_run:
                gap = float(w["start_s"]) - float(cur_run[-1]["end_s"])
                if gap > 0.30 or cur_run[-1].get("ends_sentence"):
                    runs.append(cur_run)
                    cur_run = []
            cur_run.append(w)
        if cur_run:
            runs.append(cur_run)

    raw_chunks: list[dict[str, Any]] = []
    for run in runs:
        for chunk_words in _partition_run_into_chunks(run):
            text = _join_chunk_words(chunk_words)
            if not text:
                continue
            start_s = max(0.0, float(chunk_words[0]["start_s"]))
            end_s = min(final_end_s, float(chunk_words[-1]["end_s"]))
            if end_s <= start_s:
                end_s = min(final_end_s, start_s + 0.04)
            if start_s >= final_end_s:
                continue
            raw_chunks.append({"start_s": start_s, "end_s": end_s, "text": text})

    for idx in range(len(raw_chunks) - 1):
        cur = raw_chunks[idx]
        nxt = raw_chunks[idx + 1]
        if nxt["start_s"] < cur["end_s"]:
            cur["end_s"] = nxt["start_s"]
        elif nxt["start_s"] - cur["end_s"] < 0.12:
            cur["end_s"] = nxt["start_s"]

    srt_lines: list[str] = []
    cue_num = 1
    for ch in raw_chunks:
        if ch["end_s"] - ch["start_s"] < 0.02:
            continue
        srt_lines.append(str(cue_num))
        srt_lines.append(
            f"{_format_srt_timestamp(ch['start_s'])} --> {_format_srt_timestamp(ch['end_s'])}"
        )
        srt_lines.append(ch["text"])
        srt_lines.append("")
        cue_num += 1

    out_srt.write_text("\n".join(srt_lines), encoding="utf-8")
    count = cue_num - 1
    logger.info("Wrote Shorts captions (%d chunks) -> %s", count, out_srt)
    return count


# ---------------------------------------------------------------------------
# Post-Production Foley Cue Sheet (_cues.txt)
# ---------------------------------------------------------------------------


def _fmt_tc(seconds: float) -> str:
    ms_total = max(0, int(round(seconds * 1000)))
    mins, rem = divmod(ms_total, 60_000)
    secs, ms = divmod(rem, 1000)
    return f"{mins:02d}:{secs:02d}.{ms:03d}"


def _generate_cues_txt(manifest: dict[str, Any], ep: str, out_cues: Path) -> None:
    beats = manifest.get("beats", [])
    fps = manifest.get("fps", 30)
    total_dur = float(manifest.get("total_duration_s") or 0.0)
    total_frames = int(manifest.get("total_frames") or 0)

    lines: list[str] = [
        "=" * 88,
        f"POST-PRODUCTION FOLEY & TIMING CUE SHEET — {ep}",
        "=" * 88,
        f"FPS: {fps} | Total Duration: {_fmt_tc(total_dur)} ({total_dur:.3f}s) | Total Frames: {total_frames}",
        f"Subject: {manifest.get('subject', '-')} | Delayed Reveal: {manifest.get('delayed_reveal', '-')}",
        f"Audio: {manifest.get('audio_file', manifest.get('audio_path', '-'))}",
        "",
        "ROUGH SEGMENTS SUMMARY (clip_XX.mp4 boundaries)",
        "-" * 88,
    ]

    for seg in _group_rough_segments(beats):
        seg_beats = seg["beats"]
        b0 = seg_beats[0].get("beat_index", 1)
        b1 = seg_beats[-1].get("beat_index", b0)
        lines.append(
            f"  clip_{seg['segment_id']:02d}.mp4 | {_fmt_tc(seg_beats[0]['start_s'])} --> {_fmt_tc(seg_beats[-1]['end_s'])} "
            f"| F{seg['start_frame']:04d}-F{seg['end_frame']:04d} ({seg['frame_count']}f) | Beats {b0}..{b1}"
        )

    lines.extend(
        [
            "",
            "BEAT-BY-BEAT TIMELINE, VISUAL CUES & CUTS",
            "-" * 88,
        ]
    )

    chronological_markers: list[tuple[int, float, str]] = []

    for b in beats:
        b_idx = b.get("beat_index", b.get("beat_id"))
        seg_id = int(b.get("rough_segment_id", 1))
        b_role = b.get("beat_role", "-")
        cut_type = _get_beat_transition(b, is_last=(b is beats[-1]))
        bridge = ""
        if isinstance(b.get("outgoing_transition"), dict):
            bridge = str(b["outgoing_transition"].get("transition_bridge") or "")

        lines.append(
            f"[Beat {b_idx} | {b_role.upper()} | Seg {seg_id:02d}] "
            f"{_fmt_tc(b['start_s'])} --> {_fmt_tc(b['end_s'])} "
            f"(F{b['start_frame']:04d}-F{b['end_frame']:04d}) | conf={b.get('alignment_confidence', 1.0):.2f}"
        )
        lines.append(f"  VO: {b.get('vo_line', '')}")

        cues = b.get("visual_cues")
        if isinstance(cues, list) and cues:
            for c_i, cue in enumerate(cues, start=1):
                tf = int(cue.get("trigger_frame", b["start_frame"]))
                ts = float(cue.get("trigger_start_s", b["start_s"]))
                mode = cue.get("visual_mode", "standard_view")
                trig = cue.get("vo_trigger", "")
                focal = cue.get("focal_target", "")
                act = cue.get("action", "")
                lines.append(
                    f"    Cue {b_idx}.{c_i} @ {_fmt_tc(ts)} (F{tf:04d}) [{mode}] | Trigger: \"{trig}\""
                )
                lines.append(f"      Target: {focal} -> {act}")
                chronological_markers.append((tf, ts, f"SFX_CUE B{b_idx}.{c_i}: {act} (\"{trig}\")"))

        cut_label = f"CUT [{cut_type.upper()}]" + (f" ({bridge})" if bridge else "")
        lines.append(f"  --> Outgoing @ {_fmt_tc(b['end_s'])} (F{b['end_frame']:04d}): {cut_label}")
        lines.append("")
        chronological_markers.append((int(b["end_frame"]), float(b["end_s"]), cut_label))

    lines.extend(
        [
            "CHRONOLOGICAL FOLEY / CUT MARKER LIST",
            "-" * 88,
        ]
    )
    for frame_no, sec_val, desc in sorted(chronological_markers, key=lambda x: (x[0], x[1])):
        lines.append(f"  {_fmt_tc(sec_val)} | F{frame_no:04d} | {desc}")

    out_cues.write_text("\n".join(lines) + "\n", encoding="utf-8")
    logger.info("Wrote post-production cue sheet -> %s", out_cues)


def run_assemble(args: argparse.Namespace) -> int:
    _require_tool("ffmpeg")
    _require_tool("ffprobe")

    manifest_path = Path(args.manifest)
    if not manifest_path.exists():
        raise FileNotFoundError(f"Timed storyboard manifest not found: {manifest_path}")

    with manifest_path.open("r", encoding="utf-8") as f:
        manifest = json.load(f)

    beats = manifest.get("beats", [])
    if not beats:
        raise ValueError(f"Manifest {manifest_path} contains no beats.")

    audio_arg = args.audio or manifest.get("audio_file") or manifest.get("audio_path")
    audio_path = _resolve_default_audio(audio_arg)

    clips_dir = Path(args.clips_dir)
    if not clips_dir.exists():
        raise FileNotFoundError(f"Clips directory not found: {clips_dir}")

    out_dir = Path(args.out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)

    ep = _resolve_episode_id(manifest, args.ep)
    fps = float(manifest.get("fps") or 30)
    total_frames = int(manifest.get("total_frames") or beats[-1]["end_frame"])
    target_end_s = float(manifest.get("total_duration_s") or beats[-1]["end_s"])

    clip_infos = _verify_clips(
        beats=beats,
        clips_dir=clips_dir,
        expected_fps=fps,
        frame_tolerance=int(args.frame_tolerance),
        allow_mismatch=bool(args.allow_mismatch),
    )

    out_mp4 = out_dir / f"{ep}_master.mp4"
    out_srt = out_dir / f"{ep}.srt"
    out_cues = out_dir / f"{ep}_cues.txt"

    _concatenate_and_mux(
        clip_infos=clip_infos,
        audio_path=audio_path,
        out_mp4=out_mp4,
        target_end_s=target_end_s,
        total_frames=total_frames,
        fps=fps,
        copy_video=bool(args.copy_video),
    )
    _generate_shorts_srt(manifest, out_srt)
    _generate_cues_txt(manifest, ep, out_cues)
    return 0


# ---------------------------------------------------------------------------
# CLI Entrypoint
# ---------------------------------------------------------------------------


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description="Persephone audio alignment (2.1:1) & final FFmpeg assembly (2.5:2)."
    )
    parser.add_argument(
        "--log-level",
        default="INFO",
        choices=("DEBUG", "INFO", "WARNING", "ERROR"),
        help="Logging verbosity (default: INFO).",
    )
    sub = parser.add_subparsers(dest="command", required=True)

    p_align = sub.add_parser("align", help="Block 2.1:1 — Transcribe & align storyboard to audio.")
    p_align.add_argument("--storyboard", required=True, help="Path to 0_input/storyboard.json")
    p_align.add_argument("--audio", default=None, help="Path to narration .mp3/.wav")
    p_align.add_argument("--fps", type=float, default=30.0, help="Target render FPS (default: 30)")
    p_align.add_argument("--model", default="small", help="faster-whisper model size (default: small)")
    p_align.add_argument("--device", default="cpu", help="Inference device (default: cpu)")
    p_align.add_argument("--compute-type", default="int8", help="Compute type (default: int8)")
    p_align.add_argument("--language", default="en", help="Language code or 'auto' (default: en)")
    p_align.add_argument(
        "--raw-out",
        default="1_manifests/whisper_raw.json",
        help="Path for raw Whisper JSON cache",
    )
    p_align.add_argument(
        "--out",
        default="1_manifests/timed_storyboard.json",
        help="Output path for timed_storyboard.json",
    )
    p_align.add_argument(
        "--reuse-raw",
        action="store_true",
        help="Reuse existing --raw-out JSON instead of re-running faster-whisper.",
    )

    p_asm = sub.add_parser("assemble", help="Block 2.5:2 — Verify clips, mux audio, export SRT & cues.")
    p_asm.add_argument("--manifest", required=True, help="Path to 1_manifests/timed_storyboard.json")
    p_asm.add_argument("--audio", default=None, help="Path to narration .mp3/.wav")
    p_asm.add_argument("--clips-dir", default="4_renders", help="Directory containing clip_*.mp4")
    p_asm.add_argument("--out-dir", default="5_output", help="Output directory for master/srt/cues")
    p_asm.add_argument("--ep", default=None, help="Episode slug prefix (e.g. ep_01_souffle)")
    p_asm.add_argument(
        "--frame-tolerance",
        type=int,
        default=1,
        help="Allowed +/- frame variance per clip during ffprobe check (default: 1)",
    )
    p_asm.add_argument(
        "--allow-mismatch",
        action="store_true",
        help="Warn instead of failing if clip frame counts differ from manifest.",
    )
    p_asm.add_argument(
        "--copy-video",
        action="store_true",
        help="Stream-copy video instead of re-encoding libx264 (requires identical clip encoding).",
    )

    return parser


def main(argv: list[str] | None = None) -> int:
    parser = build_parser()
    args = parser.parse_args(argv)
    logging.basicConfig(
        level=getattr(logging, args.log_level),
        format="[align_audio] %(levelname)s: %(message)s",
    )

    if args.command == "align":
        return run_align(args)
    if args.command == "assemble":
        return run_assemble(args)
    parser.error(f"Unknown command: {args.command}")
    return 2


if __name__ == "__main__":
    raise SystemExit(main())