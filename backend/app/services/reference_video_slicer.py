"""Defensive FFmpeg scene-cut auto-slicer and anti-bleed guardrail for reference videos.

Prevents multi-shot reference videos (e.g., 16s commercials attached as project
defaults) from bleeding unrelated scenes into individual 2-5s shots, while
guaranteeing zero hangs, zero 422 range errors, and silent fallback on any error.
"""
from __future__ import annotations

import json
import math
import re
import subprocess
import tempfile
from pathlib import Path
from typing import Any

_CACHE_DIR = Path(tempfile.gettempdir()) / "vf_scene_cut_cache"
_MEMORY_CACHE: dict[str, list[tuple[float, float]]] = {}
_PTS_RE = re.compile(r"pts_time:([0-9]+(?:\.[0-9]+)?)")

MIN_SHOT_SECONDS = 0.8
GUARD_MARGIN_SECONDS = 0.15
MIN_RESULT_SECONDS = 0.5
SCENE_THRESHOLD = 0.25
DETECT_TIMEOUT_SECONDS = 30


def _merge_short_segments(cuts: list[float], duration: float) -> list[tuple[float, float]]:
    """Build clean [start, end] intervals and merge sub-0.8s flash fragments."""
    points = [0.0] + sorted(t for t in cuts if 0.05 < t < duration - 0.05) + [duration]
    raw = [(points[i], points[i + 1]) for i in range(len(points) - 1) if points[i + 1] > points[i]]
    if not raw:
        return [(0.0, duration)]
    merged: list[list[float]] = []
    for start, end in raw:
        if not merged:
            merged.append([start, end])
            continue
        if (end - start) < MIN_SHOT_SECONDS:
            merged[-1][1] = end
        elif (merged[-1][1] - merged[-1][0]) < MIN_SHOT_SECONDS:
            merged[-1][1] = end
        else:
            merged.append([start, end])
    return [(round(s, 3), round(e, 3)) for s, e in merged if e > s]


def detect_video_shots(
    ffmpeg_path: str,
    source_path: Path,
    source_duration: float,
    sha256: str,
    *,
    scene_threshold: float = SCENE_THRESHOLD,
    timeout_seconds: int = DETECT_TIMEOUT_SECONDS,
) -> list[tuple[float, float]]:
    """Detect clean single-shot intervals with fast downscaled FFmpeg probe + cache.

    Never raises: any FFmpeg error, timeout, or corrupt file silently falls back
    to [(0.0, source_duration)].
    """
    if not math.isfinite(source_duration) or source_duration <= 0:
        return []
    threshold = max(0.05, min(0.95, float(scene_threshold)))
    threshold_suffix = "" if abs(threshold - SCENE_THRESHOLD) < 1e-6 else f":t{round(threshold, 3)}"
    cache_key = f"{sha256}:{round(source_duration, 3)}{threshold_suffix}"
    if cache_key in _MEMORY_CACHE:
        return list(_MEMORY_CACHE[cache_key])

    file_suffix = "" if not threshold_suffix else f"-t{round(threshold, 3)}"
    cache_file = _CACHE_DIR / f"{sha256}-{round(source_duration, 3)}{file_suffix}.json"
    try:
        if cache_file.is_file():
            cached = json.loads(cache_file.read_text(encoding="utf-8"))
            if isinstance(cached, list) and cached:
                parsed = [(float(a), float(b)) for a, b in cached]
                if all(
                    math.isfinite(start)
                    and math.isfinite(end)
                    and 0 <= start < end <= source_duration + 0.01
                    for start, end in parsed
                ):
                    _MEMORY_CACHE[cache_key] = parsed
                    return list(parsed)
    except Exception:
        pass

    fallback = [(0.0, round(source_duration, 3))]
    try:
        if not source_path.is_file():
            return fallback
        cmd = [
            ffmpeg_path,
            "-nostdin",
            "-hide_banner",
            "-i",
            str(source_path),
            "-an",
            "-sn",
            "-dn",
            "-vf",
            f"scale=320:-1,select='gt(scene,{threshold})',showinfo",
            "-f",
            "null",
            "-",
        ]
        proc = subprocess.run(
            cmd,
            capture_output=True,
            text=True,
            timeout=max(10, int(timeout_seconds)),
            creationflags=getattr(subprocess, "CREATE_NO_WINDOW", 0),
        )
        if proc.returncode != 0:
            return fallback
        cuts = [float(m.group(1)) for m in _PTS_RE.finditer(proc.stderr or "")]
        segments = _merge_short_segments(cuts, source_duration)
        if not segments:
            segments = fallback
        _MEMORY_CACHE[cache_key] = segments
        try:
            _CACHE_DIR.mkdir(parents=True, exist_ok=True)
            cache_file.write_text(json.dumps(segments), encoding="utf-8")
        except Exception:
            pass
        return list(segments)
    except Exception:
        return fallback


def plan_long_video_shots(
    segments: list[tuple[float, float]],
    source_duration: float,
    *,
    min_shot_seconds: float = 2.5,
    max_shot_seconds: float = 8.0,
    target_shot_seconds: float = 5.0,
    guard_margin_seconds: float = 0.12,
) -> list[dict[str, Any]]:
    """Convert raw scene segments from a long video into H3-safe, physically stable shots.

    Guarantees:
    - Merges sub-``min_shot_seconds`` flash fragments into neighboring shots.
    - Subdivides long continuous takes exceeding ``max_shot_seconds`` (capped at 15s)
      into balanced sub-shots around ``target_shot_seconds`` to prevent physical drift.
    - Applies an interior boundary guard margin at scene cuts so adjacent transition
      frames never bleed into a shot's reference clip.
    """
    if not math.isfinite(source_duration) or source_duration <= 0:
        return []
    duration = round(float(source_duration), 3)
    safe_max = max(2.0, min(15.0, float(max_shot_seconds)))
    safe_min = max(1.0, min(safe_max, float(min_shot_seconds)))
    safe_target = max(safe_min, min(safe_max, float(target_shot_seconds)))
    safe_guard = max(0.0, min(0.5, float(guard_margin_seconds)))

    cleaned = sorted(
        (max(0.0, round(float(s), 3)), min(duration, round(float(e), 3)))
        for s, e in (segments or [(0.0, duration)])
        if math.isfinite(s) and math.isfinite(e) and min(duration, float(e)) > max(0.0, float(s)) + 0.05
    )
    if not cleaned:
        cleaned = [(0.0, duration)]

    # 1. Merge fragments shorter than safe_min with adjacent segments.
    merged: list[list[float]] = []
    for start, end in cleaned:
        if not merged:
            merged.append([start, end])
            continue
        seg_len = end - start
        prev_len = merged[-1][1] - merged[-1][0]
        if seg_len < safe_min or prev_len < safe_min:
            merged[-1][1] = end
        else:
            merged.append([start, end])
    if len(merged) >= 2 and (merged[-1][1] - merged[-1][0]) < safe_min:
        merged[-2][1] = merged[-1][1]
        merged.pop()

    # 2. Subdivide segments exceeding safe_max into balanced windows.
    planned_raw: list[tuple[float, float, str]] = []
    single_source = len(merged) == 1
    for start, end in merged:
        seg_len = end - start
        if seg_len <= safe_max + 1e-6:
            strategy = "single_segment" if single_source else "scene_cut"
            planned_raw.append((round(start, 3), round(end, 3), strategy))
            continue
        pieces = max(2, int(math.ceil(seg_len / safe_target)))
        while (seg_len / pieces) > safe_max + 1e-6:
            pieces += 1
        step = seg_len / pieces
        for idx in range(pieces):
            sub_start = start + idx * step
            sub_end = end if idx == pieces - 1 else start + (idx + 1) * step
            planned_raw.append((round(sub_start, 3), round(sub_end, 3), "subdivided_long_take"))

    # 3. Apply interior guard margin at cut boundaries without collapsing below safe floor.
    results: list[dict[str, Any]] = []
    total_count = len(planned_raw)
    min_floor = min(duration, max(0.5, min(2.0, safe_min)))
    for idx, (raw_start, raw_end, strategy) in enumerate(planned_raw, start=1):
        trim_start, trim_end = raw_start, raw_end
        if total_count > 1 and safe_guard > 0:
            left_pad = safe_guard if idx > 1 else 0.0
            right_pad = safe_guard if idx < total_count else 0.0
            if (trim_end - trim_start) - (left_pad + right_pad) >= min_floor:
                trim_start = round(trim_start + left_pad, 3)
                trim_end = round(trim_end - right_pad, 3)
        trim_start = max(0.0, min(trim_start, max(0.0, duration - 0.05)))
        trim_end = min(duration, max(trim_start + 0.05, trim_end))
        clip_duration = round(trim_end - trim_start, 2)
        shot_duration = round(max(1.0, min(15.0, clip_duration)), 1)
        results.append({
            "index": idx,
            "raw_start_seconds": raw_start,
            "raw_end_seconds": raw_end,
            "trim_start_seconds": round(trim_start, 3),
            "trim_end_seconds": round(trim_end, 3),
            "clip_duration_seconds": clip_duration,
            "duration_seconds": shot_duration,
            "source_cut_strategy": strategy,
        })
    return results


def _safe_clamp(
    start: float,
    end: float,
    bound_start: float,
    bound_end: float,
    source_duration: float,
    apply_guard: bool = True,
) -> tuple[float, float]:
    """Apply guard margin and strictly clamp within valid non-empty bounds."""
    orig_start, orig_end = start, end
    if apply_guard and (end - start) >= 1.2:
        start += GUARD_MARGIN_SECONDS
        end -= GUARD_MARGIN_SECONDS

    max_end = min(bound_end, max(0.0, source_duration - 0.01))
    if max_end <= bound_start + 0.05:
        return round(bound_start, 3), round(bound_end, 3)

    start = max(bound_start, min(start, max_end - MIN_RESULT_SECONDS))
    end = min(max_end, max(start + MIN_RESULT_SECONDS, end))
    if end <= start + 0.05:
        return round(orig_start, 3), round(orig_end, 3)
    return round(start, 3), round(end, 3)


def auto_slice_range_for_shot(
    *,
    ffmpeg_path: str,
    source_path: Path,
    source_duration: float,
    sha256: str,
    trim_start: float,
    trim_end: float,
    shot_index: int,
    shot_duration: float,
    all_shot_durations: list[float],
) -> tuple[float, float, dict[str, Any]]:
    """Auto-slice a multi-shot reference range to one clean segment for this shot.

    - Leaves short ranges and single-shot projects untouched.
    - Guarantees valid [new_start, new_end] within [trim_start, trim_end].
    - Never raises.
    """
    selected_dur = trim_end - trim_start
    shot_dur = max(1.0, float(shot_duration or 3.0))
    if (
        not math.isfinite(selected_dur)
        or selected_dur <= max(shot_dur * 1.5, 4.5)
        or source_duration <= 1.2
        or len(all_shot_durations) <= 1
    ):
        return trim_start, trim_end, {"strategy": "preserved_manual_or_short", "sliced": False}

    try:
        all_segments = detect_video_shots(ffmpeg_path, source_path, source_duration, sha256)
        in_range = [
            (max(s, trim_start), min(e, trim_end))
            for s, e in all_segments
            if min(e, trim_end) - max(s, trim_start) >= MIN_RESULT_SECONDS
        ]

        total_script = sum(max(0.5, float(d or 2.0)) for d in all_shot_durations)
        idx = max(0, min(shot_index, len(all_shot_durations) - 1))
        cum_start = sum(max(0.5, float(d or 2.0)) for d in all_shot_durations[:idx])
        cum_end = cum_start + max(0.5, float(all_shot_durations[idx] or 2.0))
        start_ratio = cum_start / total_script
        end_ratio = cum_end / total_script
        mid_ratio = (start_ratio + end_ratio) * 0.5

        if len(in_range) >= 2:
            target_mid = trim_start + mid_ratio * selected_dur
            chosen_idx = min(
                range(len(in_range)),
                key=lambda i: abs(((in_range[i][0] + in_range[i][1]) * 0.5) - target_mid),
            )
            seg_start, seg_end = in_range[chosen_idx]
            max_seg = max(shot_dur * 1.5, 4.0)
            if (seg_end - seg_start) > max_seg:
                win = min(seg_end - seg_start, max(shot_dur, 2.5))
                w_start = max(seg_start, min(target_mid - win * 0.5, seg_end - win))
                seg_start, seg_end = w_start, w_start + win
            new_start, new_end = _safe_clamp(
                seg_start, seg_end, trim_start, trim_end, source_duration, apply_guard=True
            )
            return new_start, new_end, {
                "strategy": "ffmpeg_scene_cut_aligned",
                "sliced": True,
                "detected_shots_count": len(in_range),
                "chosen_shot_index": chosen_idx,
                "original_range": [round(trim_start, 3), round(trim_end, 3)],
                "sliced_range": [new_start, new_end],
            }

        # Fallback: single-take / soft-dissolve video -> proportional timeline slice
        prop_start = trim_start + start_ratio * selected_dur
        prop_end = trim_start + end_ratio * selected_dur
        new_start, new_end = _safe_clamp(
            prop_start, prop_end, trim_start, trim_end, source_duration, apply_guard=False
        )
        return new_start, new_end, {
            "strategy": "timeline_proportional_fallback",
            "sliced": True,
            "detected_shots_count": len(in_range),
            "original_range": [round(trim_start, 3), round(trim_end, 3)],
            "sliced_range": [new_start, new_end],
        }
    except Exception as exc:
        return trim_start, trim_end, {
            "strategy": "error_fallback_original",
            "sliced": False,
            "error": type(exc).__name__,
        }
