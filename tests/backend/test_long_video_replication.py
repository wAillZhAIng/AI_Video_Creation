"""Tests for long reference video auto-splitting and multi-shot project replication."""
from __future__ import annotations

from pathlib import Path

import pytest
from fastapi.testclient import TestClient

from app import models as m
from app.core.config import Settings
from app.core.db import build_engine, build_session_factory
from app.main import create_app
from app.services import long_video_replication as service
from app.services import reference_video_analysis as ref_library
from app.services.reference_video_slicer import plan_long_video_shots


def test_plan_long_video_shots_handles_four_minute_mixed_cuts():
    # 240s video containing flash cuts (0.4s), normal cuts (4-6s), and long takes (35s, 60s)
    segments = [
        (0.0, 0.4),
        (0.4, 5.2),
        (5.2, 40.2),
        (40.2, 46.0),
        (46.0, 46.5),
        (46.5, 106.5),
        (106.5, 112.0),
        (112.0, 180.0),
        (180.0, 240.0),
    ]
    planned = plan_long_video_shots(
        segments,
        240.0,
        min_shot_seconds=2.5,
        max_shot_seconds=8.0,
        target_shot_seconds=5.0,
        guard_margin_seconds=0.12,
    )
    assert len(planned) >= 35
    total_raw = 0.0
    prev_end = -1.0
    for item in planned:
        assert item['trim_start_seconds'] >= 0.0
        assert item['trim_end_seconds'] <= 240.0
        assert item['trim_end_seconds'] > item['trim_start_seconds']
        assert item['trim_start_seconds'] >= prev_end - 1e-6
        assert 2.0 <= item['duration_seconds'] <= 8.0
        prev_end = item['trim_end_seconds']
        total_raw += item['raw_end_seconds'] - item['raw_start_seconds']
    assert abs(total_raw - 240.0) < 0.05


def test_detect_aspect_ratio_maps_common_dimensions():
    assert service.detect_aspect_ratio(1920, 1080) == '16:9'
    assert service.detect_aspect_ratio(1080, 1920) == '9:16'
    assert service.detect_aspect_ratio(1024, 1024) == '1:1'
    assert service.detect_aspect_ratio(None, None) == '16:9'


def test_import_long_video_api_creates_multishot_project_with_per_shot_references(tmp_path: Path, monkeypatch):
    db_path = tmp_path / 'test.db'
    media_root = tmp_path / 'media'
    settings = Settings(
        database_url=f'sqlite:///{db_path.as_posix()}',
        media_root=media_root,
        seed_demo_data=False,
        seed_local_model_services=False,
        h3_demo_mode=True,
    )

    # Mock FFprobe/FFmpeg decode and scene detection so the test is self-contained.
    monkeypatch.setattr(
        ref_library,
        'probe_video',
        lambda _settings, path, full_decode=True: {
            'duration_seconds': 210.0,
            'width': 1920,
            'height': 1080,
            'fps': 24.0,
            'format': 'mp4',
            'codec': 'h264',
            'has_audio': True,
            'audio_sample_rate': 48000,
            'audio_channels': 2,
            'byte_size': Path(path).stat().st_size,
            'full_decode': True,
            'evidence': 'mocked test probe',
        },
    )
    monkeypatch.setattr(
        service,
        'detect_video_shots',
        lambda _ffmpeg, _path, duration, _sha, **_kwargs: [
            (0.0, 24.0),
            (24.0, 90.0),
            (90.0, 150.0),
            (150.0, duration),
        ],
    )

    app = create_app(settings)
    with TestClient(app) as client:
        # 1. Invalid max_shot_seconds > 15 is rejected.
        bad = client.post(
            '/api/studio/import-long-video',
            files={'file': ('long_ref.mp4', b'fake-mp4-bytes', 'video/mp4')},
            data={'max_shot_seconds': '20'},
        )
        assert bad.status_code == 422

        # 2. Valid 3.5-minute (210s) reference video auto-splits into a multi-shot project.
        res = client.post(
            '/api/studio/import-long-video',
            files={'file': ('long_ref.mp4', b'fake-mp4-bytes-210s', 'video/mp4')},
            data={
                'project_name': '3分半复刻项目',
                'min_shot_seconds': '2.5',
                'target_shot_seconds': '5.0',
                'max_shot_seconds': '8.0',
                'reference_role': 'combined',
                'custom_prompt_prefix': '写实电影质感',
            },
        )
        assert res.status_code == 201, res.text
        payload = res.json()
        assert payload['project_name'] == '3分半复刻项目'
        assert payload['aspect_ratio'] == '16:9'
        assert payload['source_duration_seconds'] == 210.0
        assert payload['shot_count'] >= 30
        assert all(2.0 <= shot['duration_seconds'] <= 8.0 for shot in payload['shots'])

        # 3. Preview endpoint works for the uploaded library video.
        preview = client.post(
            '/api/studio/preview-long-video-split',
            data={
                'video_id': payload['video_id'],
                'min_shot_seconds': '2.5',
                'target_shot_seconds': '6.0',
                'max_shot_seconds': '8.0',
            },
        )
        assert preview.status_code == 200, preview.text
        assert preview.json()['shot_count'] >= 25

        # 4. Verify database records and per-shot reference_video_overrides.
        engine = build_engine(settings.database_url)
        session_factory = build_session_factory(engine)
        with session_factory() as session:
            project = session.get(m.Project, payload['project_id'])
            assert project is not None and project.source_type == 'long_video_replication'
            links = session.query(m.ProjectReferenceVideo).filter_by(project_id=project.id).all()
            assert len(links) == 1 and links[0].video_id == payload['video_id']
            shots = session.query(m.StoryboardShot).filter_by(project_id=project.id).order_by(m.StoryboardShot.order).all()
            assert len(shots) == payload['shot_count']
            for shot in shots:
                assert '写实电影质感' in shot.visual_prompt
                assert shot.negative_prompt == service.PHYSICAL_REALISM_NEGATIVE_PROMPT
                detail = session.query(m.ShotDetail).filter_by(shot_id=shot.id).one()
                overrides = detail.source_json['production']['reference_video_overrides']
                assert len(overrides) == 1
                ref = overrides[0]
                assert ref['video_id'] == payload['video_id']
                assert ref['authorized_for_reference'] is True
                assert 0 <= ref['trim_start_seconds'] < ref['trim_end_seconds'] <= 210.0
                assert (ref['trim_end_seconds'] - ref['trim_start_seconds']) <= 8.0 + 1e-3
