"""Auto-split a long reference video (e.g., 3-5 min) into a multi-shot Studio project.

Each generated shot is bounded to an H3-safe, physically stable duration window
(default 2.5-8.0s) and bound directly to its corresponding native Ref2VA video
trim range via ``ShotDetail.source_json['production']['reference_video_overrides']``.
"""
from __future__ import annotations

import base64
import math
import os
from pathlib import Path
from typing import Any

import httpx
from sqlalchemy import select

from app import models as m
from app.core.errors import AppError
from app.core.paths import safe_media_path
from app.models.entities import new_id, utcnow
from app.schemas_cinematic import MotionReferenceInput
from app.services import reference_video_analysis as ref_library
from app.services.project_reference_videos import link_uploaded_video
from app.services.reference_video_slicer import detect_video_shots, plan_long_video_shots

LONG_VIDEO_DETECT_TIMEOUT_SECONDS = 120
VALID_ROLES = {'structure', 'action', 'camera', 'style', 'combined'}

PHYSICAL_REALISM_NEGATIVE_PROMPT = (
    'deformed hands, extra fingers, distorted limbs, broken anatomy, '
    'unnatural physics, floating objects, gravity violation, morphing artifacts, '
    'flickering, sudden subject identity change, warped face, text watermark'
)


def detect_aspect_ratio(width: int | None, height: int | None) -> str:
    """Map source pixel dimensions to the closest supported Studio aspect ratio."""
    if not width or not height or width <= 0 or height <= 0:
        return '16:9'
    ratio = width / height
    candidates = {'16:9': 16 / 9, '9:16': 9 / 16, '1:1': 1.0}
    return min(candidates, key=lambda key: abs(candidates[key] - ratio))


def _validate_split_params(
    *,
    min_shot_seconds: float,
    max_shot_seconds: float,
    target_shot_seconds: float,
    scene_threshold: float,
    reference_role: str,
) -> tuple[float, float, float, float, str]:
    if not (math.isfinite(min_shot_seconds) and 1.0 <= min_shot_seconds <= 10.0):
        raise AppError(422, 'invalid_min_shot_seconds', '单镜最小时长须在 1.0 至 10.0 秒之间')
    if not (math.isfinite(max_shot_seconds) and 2.0 <= max_shot_seconds <= 15.0):
        raise AppError(422, 'invalid_max_shot_seconds', '单镜最大时长须在 2.0 至 15.0 秒之间（符合 H3 原生参考限制）')
    if min_shot_seconds > max_shot_seconds:
        raise AppError(422, 'invalid_shot_duration_range', '单镜最小时长不能大于最大时长')
    if not (math.isfinite(target_shot_seconds) and min_shot_seconds <= target_shot_seconds <= max_shot_seconds):
        raise AppError(422, 'invalid_target_shot_seconds', '单镜目标时长须位于最小与最大时长之间')
    if not (math.isfinite(scene_threshold) and 0.05 <= scene_threshold <= 0.95):
        raise AppError(422, 'invalid_scene_threshold', '转场检测阈值须在 0.05 至 0.95 之间')
    role = (reference_role or 'combined').strip()
    if role not in VALID_ROLES:
        raise AppError(422, 'invalid_reference_role', '参考用途无效')
    return (
        float(min_shot_seconds),
        float(max_shot_seconds),
        float(target_shot_seconds),
        float(scene_threshold),
        role,
    )


def _local_vlm_endpoint(session) -> tuple[str, str] | None:
    """Resolve an optional loopback Qwen/local_llm endpoint for shot descriptions."""
    configured_base = os.getenv('LOCAL_SCREENPLAY_BASE_URL', '').strip().rstrip('/')
    if configured_base:
        url = configured_base if configured_base.endswith('/v1') else configured_base + '/v1'
        return url, 'Qwen/Qwen3-VL-2B-Instruct'
    service = session.scalar(
        select(m.AIService).where(
            m.AIService.module == 'language_processing',
            m.AIService.provider == 'local_llm',
            m.AIService.enabled.is_(True),
        ).order_by(m.AIService.is_default.desc(), m.AIService.created_at.desc())
    )
    if service and service.base_url:
        base = service.base_url.strip().rstrip('/')
        url = base if base.endswith('/v1') else base + '/v1'
        return url, service.default_model or 'Qwen/Qwen3-VL-2B-Instruct'
    return None


def _describe_frame_with_local_vlm(
    endpoint: tuple[str, str] | None,
    jpeg_bytes: bytes,
    *,
    timeout_seconds: float = 20.0,
) -> str | None:
    """Ask the local VLM for a concise, physically grounded single-shot caption.

    Never raises: returns None if the local VLM is offline, busy, or times out.
    """
    if not endpoint or not jpeg_bytes:
        return None
    base_url, model_id = endpoint
    encoded = base64.b64encode(jpeg_bytes).decode('ascii')
    payload = {
        'model': model_id,
        'temperature': 0.1,
        'max_tokens': 160,
        'messages': [
            {
                'role': 'user',
                'content': [
                    {
                        'type': 'text',
                        'text': (
                            '请用简洁的中文描述这张视频关键帧中的单一画面：主体外观与动作、'
                            '物体接触关系、场景环境、光线与镜头景别。不超过60字，不要猜测剧情，'
                            '不要输出编号或多余解释。'
                        ),
                    },
                    {'type': 'image_url', 'image_url': {'url': f'data:image/jpeg;base64,{encoded}'}},
                ],
            }
        ],
    }
    try:
        with httpx.Client(timeout=timeout_seconds, trust_env=False) as client:
            response = client.post(f'{base_url}/chat/completions', json=payload)
            if response.status_code != 200:
                return None
            data = response.json()
            text = (
                ((data.get('choices') or [{}])[0].get('message') or {}).get('content') or ''
            ).strip()
            return text[:240] if text else None
    except Exception:
        return None


def _build_shot_prompt(
    *,
    index: int,
    total_shots: int,
    trim_start: float,
    trim_end: float,
    reference_role: str,
    custom_prompt_prefix: str,
    vlm_observation: str | None = None,
) -> tuple[str, str, str]:
    """Return (visual_prompt, action, camera_move) with physical-realism guardrails."""
    role_labels = {
        'combined': '动作、运镜与光影氛围',
        'action': '主体动作与物理交互',
        'camera': '镜头运镜与景别节奏',
        'style': '色彩、光线与画面质感',
        'structure': '节奏结构与物理连贯性',
    }
    role_desc = role_labels.get(reference_role, '动作、运镜与光影氛围')
    parts: list[str] = []
    if custom_prompt_prefix.strip():
        parts.append(custom_prompt_prefix.strip())
    if vlm_observation:
        parts.append(vlm_observation)
    else:
        parts.append(
            f'第 {index}/{total_shots} 镜（参考原片 {trim_start:.1f}s–{trim_end:.1f}s）：'
            f'严格复刻参考片段的{role_desc}。'
        )
    parts.append(
        '保持单一镜头连贯稳定、主体结构与肢体自然、物体支撑与重力运动符合真实物理规律，'
        '画面中不出现突然切镜、畸变、多余手指或违背常识的形变。'
    )
    visual_prompt = ' '.join(parts)
    action = vlm_observation or f'复刻原片 {trim_start:.1f}s–{trim_end:.1f}s 段落动作，动作自然连贯'
    camera_move = '跟随参考片段运镜，保持画面稳定平滑'
    return visual_prompt, action, camera_move


def preview_long_video_split(
    session,
    settings,
    video_id: str,
    *,
    min_shot_seconds: float = 2.5,
    max_shot_seconds: float = 8.0,
    target_shot_seconds: float = 5.0,
    scene_threshold: float = 0.25,
    reference_role: str = 'combined',
) -> dict[str, Any]:
    """Preview how an existing ReferenceVideo will be split into H3-safe shots."""
    min_s, max_s, target_s, threshold, role = _validate_split_params(
        min_shot_seconds=min_shot_seconds,
        max_shot_seconds=max_shot_seconds,
        target_shot_seconds=target_shot_seconds,
        scene_threshold=scene_threshold,
        reference_role=reference_role,
    )
    video = ref_library.get_video(session, video_id, active=True)
    source_path = safe_media_path(settings.media_root, video.relative_path, must_exist=True)
    meta = video.metadata_json or {}
    duration = float(meta.get('duration_seconds') or 0)
    if duration <= 0:
        raise AppError(422, 'reference_video_metadata', '参考视频缺少有效时长')
    raw_segments = detect_video_shots(
        settings.ffmpeg_path,
        source_path,
        duration,
        video.sha256,
        scene_threshold=threshold,
        timeout_seconds=LONG_VIDEO_DETECT_TIMEOUT_SECONDS,
    )
    planned = plan_long_video_shots(
        raw_segments,
        duration,
        min_shot_seconds=min_s,
        max_shot_seconds=max_s,
        target_shot_seconds=target_s,
    )
    aspect_ratio = detect_aspect_ratio(meta.get('width'), meta.get('height'))
    return {
        'video_id': video.id,
        'video_name': video.name,
        'source_duration_seconds': round(duration, 2),
        'aspect_ratio': aspect_ratio,
        'detected_scene_cuts': len(raw_segments),
        'shot_count': len(planned),
        'total_planned_duration_seconds': round(sum(item['duration_seconds'] for item in planned), 2),
        'reference_role': role,
        'shots': planned,
    }


def replicate_long_video_to_project(
    session,
    settings,
    *,
    video_id: str | None = None,
    filename: str = '',
    data: bytes | None = None,
    project_name: str = '',
    min_shot_seconds: float = 2.5,
    max_shot_seconds: float = 8.0,
    target_shot_seconds: float = 5.0,
    scene_threshold: float = 0.25,
    reference_role: str = 'combined',
    include_audio: bool = False,
    custom_prompt_prefix: str = '',
    analyze_with_vlm: bool = False,
    quality: dict[str, Any] | None = None,
) -> dict[str, Any]:
    """Create a multi-shot Studio project from a long reference video.

    Either imports a newly uploaded video (when ``data`` is supplied) or reuses
    an active library video (when ``video_id`` is supplied). Each shot receives
    an explicit, project-scoped ``reference_video_overrides`` entry bounded to
    its clean slice so MiniMax H3 Ref2VA stays within its native limits.
    """
    min_s, max_s, target_s, threshold, role = _validate_split_params(
        min_shot_seconds=min_shot_seconds,
        max_shot_seconds=max_shot_seconds,
        target_shot_seconds=target_shot_seconds,
        scene_threshold=scene_threshold,
        reference_role=reference_role,
    )
    if not video_id and data is None:
        raise AppError(422, 'reference_video_required', '请上传参考视频或选择已有参考视频')

    uploaded_new = False
    video = None
    try:
        if video_id:
            video = ref_library.get_video(session, video_id, active=True)
        else:
            video, duplicate = ref_library.import_video(
                session,
                settings,
                filename or 'reference.mp4',
                data or b'',
                name=project_name.strip() or Path(filename or 'reference.mp4').stem,
                category='long_video_replication',
                source_note='long-video-auto-split',
                usage_scope='native_ref2va',
                commit=False,
            )
            uploaded_new = not duplicate

        source_path = safe_media_path(settings.media_root, video.relative_path, must_exist=True)
        meta = video.metadata_json or {}
        duration = float(meta.get('duration_seconds') or 0)
        if duration <= 0:
            raise AppError(422, 'reference_video_metadata', '参考视频缺少有效时长')

        raw_segments = detect_video_shots(
            settings.ffmpeg_path,
            source_path,
            duration,
            video.sha256,
            scene_threshold=threshold,
            timeout_seconds=LONG_VIDEO_DETECT_TIMEOUT_SECONDS,
        )
        planned = plan_long_video_shots(
            raw_segments,
            duration,
            min_shot_seconds=min_s,
            max_shot_seconds=max_s,
            target_shot_seconds=target_s,
        )
        if not planned:
            raise AppError(422, 'long_video_split_empty', '未能从该视频切分出有效分镜')

        aspect_ratio = detect_aspect_ratio(meta.get('width'), meta.get('height'))
        resolved_name = (project_name.strip() or f'{video.name} · 自动拆镜复刻')[:200]

        project = m.Project(
            id=new_id('prj'),
            name=resolved_name,
            source_type='long_video_replication',
            source_filename=video.filename,
            source_text='',
            status='draft',
        )
        session.add(project)
        session.flush()

        # Explicitly grant this project access to the uploaded/selected reference video.
        link_uploaded_video(session, project.id, video.id)

        vlm_endpoint = _local_vlm_endpoint(session) if analyze_with_vlm else None
        vlm_described_count = 0

        timeline_lines = [
            f'项目名称：{resolved_name}',
            f'来源参考视频：{video.name}（{duration:.2f} 秒，画幅 {aspect_ratio}，共切分 {len(planned)} 镜）',
            '',
        ]

        script_rev = m.ScriptRevision(
            id=new_id('scr'),
            project_id=project.id,
            revision=1,
            parsed_json={
                'source': 'long_video_replication',
                'video_id': video.id,
                'source_sha256': video.sha256,
                'source_duration_seconds': round(duration, 2),
                'aspect_ratio': aspect_ratio,
                'detected_scene_cuts': len(raw_segments),
                'shot_count': len(planned),
            },
            raw_text='',
            status='approved',
            approved_at=utcnow(),
            approved_by='long_video_replication',
        )
        session.add(script_rev)
        session.flush()

        scene = m.Scene(
            id=new_id('scene'),
            project_id=project.id,
            script_revision_id=script_rev.id,
            order=1,
            name=f'{video.name} · 自动拆解场景'[:240],
            environment_json={
                'source': 'long_video_replication',
                'aspect_ratio': aspect_ratio,
                'source_duration_seconds': round(duration, 2),
            },
            source_json={'video_id': video.id},
        )
        session.add(scene)
        session.flush()

        created_shots: list[dict[str, Any]] = []
        total_shots = len(planned)
        for item in planned:
            idx = item['index']
            trim_start = item['trim_start_seconds']
            trim_end = item['trim_end_seconds']
            vlm_text = None
            if vlm_endpoint is not None:
                midpoint = min(max(0.0, (trim_start + trim_end) * 0.5), max(0.0, duration - 0.1))
                try:
                    jpeg_bytes = ref_library.extract_frame(settings, source_path, midpoint)
                    vlm_text = _describe_frame_with_local_vlm(vlm_endpoint, jpeg_bytes)
                    if vlm_text:
                        vlm_described_count += 1
                except Exception:
                    vlm_text = None

            visual_prompt, action, camera_move = _build_shot_prompt(
                index=idx,
                total_shots=total_shots,
                trim_start=trim_start,
                trim_end=trim_end,
                reference_role=role,
                custom_prompt_prefix=custom_prompt_prefix,
                vlm_observation=vlm_text,
            )

            ref_binding = MotionReferenceInput(
                video_id=video.id,
                expected_revision=video.revision,
                source_sha256=video.sha256,
                trim_start_seconds=trim_start,
                trim_end_seconds=trim_end,
                authorized_for_reference=True,
                reference_role=role,
                include_audio=bool(include_audio),
            ).model_dump(exclude_none=True)

            shot_title = f'镜头 {idx:02d} ({trim_start:.1f}s–{trim_end:.1f}s)'
            shot = m.StoryboardShot(
                id=new_id('shot'),
                project_id=project.id,
                order=idx,
                scene_title=shot_title,
                visual_prompt=visual_prompt,
                negative_prompt=PHYSICAL_REALISM_NEGATIVE_PROMPT,
                action=action,
                camera_move=camera_move,
                duration_seconds=item['duration_seconds'],
                aspect_ratio=aspect_ratio,
                status='draft',
                revision=1,
            )
            session.add(shot)
            session.flush()

            detail = m.ShotDetail(
                id=new_id('detail'),
                shot_id=shot.id,
                scene_id=scene.id,
                photography_json={'camera_move': camera_move, 'aspect_ratio': aspect_ratio},
                source_json={
                    'production': {'reference_video_overrides': [ref_binding]},
                    'long_video_replication': {
                        'video_id': video.id,
                        'raw_start_seconds': item['raw_start_seconds'],
                        'raw_end_seconds': item['raw_end_seconds'],
                        'trim_start_seconds': trim_start,
                        'trim_end_seconds': trim_end,
                        'source_cut_strategy': item['source_cut_strategy'],
                        'vlm_described': bool(vlm_text),
                    },
                },
            )
            session.add(detail)
            timeline_lines.append(
                f'分镜 {idx} {shot_title}（{item["duration_seconds"]}s）：{visual_prompt}'
            )
            created_shots.append({
                'shot_id': shot.id,
                'order': idx,
                'title': shot_title,
                'trim_start_seconds': trim_start,
                'trim_end_seconds': trim_end,
                'duration_seconds': item['duration_seconds'],
                'source_cut_strategy': item['source_cut_strategy'],
                'vlm_described': bool(vlm_text),
                'visual_prompt': visual_prompt,
            })

        full_text = '\n'.join(timeline_lines)
        project.source_text = full_text
        script_rev.raw_text = full_text

        if quality:
            from app.schemas_quality import QualityProfileSave, QualitySettings
            from app.services.video_quality import save_profile

            merged_quality = QualitySettings.model_validate({
                **quality,
                'aspect_ratio': aspect_ratio,
            })
            save_profile(
                session,
                'project',
                project.id,
                QualityProfileSave(settings=merged_quality, expected_revision=0),
                settings,
            )

        session.commit()
        return {
            'status': 'imported',
            'project_id': project.id,
            'project_name': project.name,
            'video_id': video.id,
            'video_name': video.name,
            'aspect_ratio': aspect_ratio,
            'source_duration_seconds': round(duration, 2),
            'detected_scene_cuts': len(raw_segments),
            'shot_count': len(created_shots),
            'total_planned_duration_seconds': round(sum(s['duration_seconds'] for s in created_shots), 2),
            'reference_role': role,
            'include_audio': bool(include_audio),
            'vlm_described_count': vlm_described_count,
            'shots': created_shots,
        }
    except Exception:
        session.rollback()
        if uploaded_new and video is not None:
            candidate = safe_media_path(settings.media_root, video.relative_path)
            if candidate.is_file():
                candidate.unlink()
        raise
