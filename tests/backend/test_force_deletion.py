"""Verify highest-privilege unconditional record & disk deletion across AI_Video_Creation."""
from __future__ import annotations

import os
import stat
from pathlib import Path
from types import SimpleNamespace

import pytest
from fastapi.testclient import TestClient

from app import models as m
from app.core.config import Settings
from app.core.db import Base, build_engine, build_session_factory
from app.core.errors import AppError
from app.main import create_app
from app.services.content_manager import build_plan, execute_delete, force_unlink_path


def _setup(tmp_path: Path):
    db_url = f"sqlite:///{tmp_path / 'force_del.db'}"
    media_root = tmp_path / "media"
    media_root.mkdir(parents=True, exist_ok=True)
    settings = Settings(app_env="test", database_url=db_url, media_root=media_root, h3_demo_mode=True)
    engine = build_engine(db_url)
    Base.metadata.create_all(engine)
    return settings, engine, build_session_factory(engine)


def test_force_delete_project_with_active_job_readonly_and_extra_files(tmp_path: Path):
    settings, engine, Session = _setup(tmp_path)
    media = Path(settings.media_root)

    vid_rel = "projects/p1/shots/s1/revisions/1/job1.mp4"
    vid_file = media / vid_rel
    vid_file.parent.mkdir(parents=True, exist_ok=True)
    vid_file.write_bytes(b"video-bytes-12345")
    os.chmod(vid_file, stat.S_IREAD)  # Read-only disk file must still be force-deleted!

    orphan_file = media / "projects/p1/tmp_cache/unindexed_frame.png"
    orphan_file.parent.mkdir(parents=True, exist_ok=True)
    orphan_file.write_bytes(b"png-bytes-999")

    sym_file = media / "projects/p1/link_to_vid.mp4"
    os.symlink(vid_file, sym_file)

    lora_cfg = media.parent / "custom_lora_configs" / "p1.json"
    lora_cfg.parent.mkdir(parents=True, exist_ok=True)
    lora_cfg.write_text('{"loras": []}', encoding="utf-8")

    try:
        with Session() as s:
            s.add(m.Project(id="p1", name="Active Project"))
            s.flush()
            s.add(m.StoryboardShot(id="s1", project_id="p1", order=1, status="approved"))
            s.flush()
            s.add(m.GenerationJob(
                id="job1", project_id="p1", shot_id="s1", provider="comfyui_h3",
                status="running", prompt="test prompt",
            ))
            s.flush()
            s.add(m.VideoRevision(
                id="v1", project_id="p1", shot_id="s1", generation_job_id="job1",
                relative_path=vid_rel, revision=1,
            ))
            s.add(m.StudioRun(
                id="run1", project_id="p1", status="running",
                snapshot_json={}, nodes_json=[], result_json={},
            ))
            s.commit()

            plan = build_plan(s, settings, "projects", ["p1"], "permanent", cascade=False)
            assert plan["blockers"] == [], f"Expected zero blockers, got: {plan['blockers']}"
            assert all(not f["protected"] for f in plan["files"])

            task = execute_delete(s, settings, SimpleNamespace(
                resource="projects", ids=["p1"], mode="permanent", cascade=False,
                fingerprint="stale-fingerprint-ignored", confirm=True, confirm_permanent=True,
            ))
            assert task["status"] == "completed", task
            assert not vid_file.exists(), "Read-only video file should be physically removed"
            assert not orphan_file.exists(), "Unindexed project disk file should be physically removed"
            assert not sym_file.exists() and not sym_file.is_symlink(), "Symlink inside media_root should be removed"
            assert not (media / "projects/p1").exists(), "Empty project dir should be removed"
            assert not lora_cfg.exists(), "Custom LoRA project config should be removed"
            assert s.get(m.Project, "p1") is None
            assert s.get(m.GenerationJob, "job1") is None
            assert s.get(m.StudioRun, "run1") is None
    finally:
        engine.dispose()


def test_direct_delete_endpoints_never_block_with_409(tmp_path: Path):
    settings, engine, Session = _setup(tmp_path)
    media = Path(settings.media_root)

    rv_rel = "asset_library/reference_videos/rv1/original.mp4"
    (media / rv_rel).parent.mkdir(parents=True, exist_ok=True)
    (media / rv_rel).write_bytes(b"ref-video-bytes")

    vl_rel = "asset_library/voices/voice1/sample.wav"
    (media / vl_rel).parent.mkdir(parents=True, exist_ok=True)
    (media / vl_rel).write_bytes(b"voice-bytes")

    sp_rel = "screenplay_jobs/sp1/source_frames/f1.jpg"
    (media / sp_rel).parent.mkdir(parents=True, exist_ok=True)
    (media / sp_rel).write_bytes(b"frame-bytes")

    with Session() as s:
        s.add(m.Project(id="p2", name="Busy Project"))
        s.add(m.ReferenceVideo(
            id="rv1", name="Ref1", filename="original.mp4", relative_path=rv_rel,
            sha256="a" * 64, state="active", revision=1, metadata_json={"duration_seconds": 5.0},
        ))
        s.add(m.VoiceLibraryAsset(
            id="voice1", name="V1", relative_path=vl_rel,
            sha256="b" * 64, file_size=11, state="active", revision=1,
        ))
        s.add(m.Product(id="prod1", name="Prod1", state="active", revision=1))
        s.add(m.ScreenplayGenerationJob(
            id="sp1", idempotency_key="sp-key-1", request_sha256="d" * 64,
            project_name="Running Screenplay", input_mode="text", creative_prompt="prompt",
            status="running", snapshot_json={},
            receipt_json={"source": {"frames": [{"relative_path": sp_rel}]}},
        ))
        s.flush()
        s.add(m.StoryboardShot(id="s2", project_id="p2", order=1, status="approved"))
        s.add(m.VideoAnalysis(
            id="va1", video_id="rv1", version=1, status="running",
            idempotency_key="va-key-1", request_sha256="c" * 64,
        ))
        s.add(m.ProjectProductBinding(id="ppb1", project_id="p2", product_id="prod1", snapshot_json={"images": []}))
        s.flush()
        s.add(m.GenerationJob(
            id="job2", project_id="p2", shot_id="s2", provider="comfyui_h3", status="running",
            prompt="test motion prompt", parameters_json={"motion_references": [{"video_id": "rv1"}]},
        ))
        s.add(m.StyleProfile(id="stp1", name="Style1", video_ids_json=["rv1"], analysis_ids_json=["va1"]))
        s.commit()

    app = create_app(settings)
    try:
        with TestClient(app) as client:
            # 1. DELETE shot with active running job -> must succeed (200), not 409
            r_shot = client.delete("/api/shots/s2")
            assert r_shot.status_code == 200, r_shot.text

            # 2. DELETE reference video referenced by active job + running analysis + style profile -> 200 & unlink
            r_rv = client.delete("/api/reference-videos/rv1")
            assert r_rv.status_code == 200, r_rv.text
            assert not (media / rv_rel).exists()

            # 3. DELETE voice library asset -> 200 & unlink
            r_vl = client.delete("/api/voice-library/voice1")
            assert r_vl.status_code == 200, r_vl.text
            assert not (media / vl_rel).exists()

            # 4. DELETE product binding on busy project -> 200
            r_unbind = client.delete("/api/product-library/prod1/project-bindings/ppb1")
            assert r_unbind.status_code == 200, r_unbind.text

            # 5. DELETE product -> 200
            r_pr = client.delete("/api/product-library/prod1")
            assert r_pr.status_code == 200, r_pr.text

            # 6. DELETE running screenplay job batch -> ok=True & disk file removed
            r_sp = client.request("DELETE", "/api/screenplay/jobs/batch", json={"ids": ["sp1"]})
            assert r_sp.status_code == 200, r_sp.text
            sp_body = r_sp.json()
            sp_items = sp_body.get("results", sp_body.get("items")) if isinstance(sp_body, dict) else sp_body
            assert sp_items and sp_items[0]["ok"] is True, sp_body
            assert not (media / sp_rel).exists()

            # 7. Trash + purge with read-only file -> succeeds without confirm_permanent blocker
            proj_rel = "projects/p2/extra.bin"
            (media / proj_rel).parent.mkdir(parents=True, exist_ok=True)
            (media / proj_rel).write_bytes(b"trash-me")
            os.chmod(media / proj_rel, stat.S_IREAD)
            r_tr = client.post("/api/deletions/execute", json={
                "resource": "projects", "ids": ["p2"], "mode": "trash", "cascade": False, "confirm": True,
            })
            assert r_tr.status_code == 200, r_tr.text
            tid = r_tr.json()["id"]
            assert r_tr.json()["status"] == "completed", r_tr.json()
            r_pg = client.post(f"/api/deletions/{tid}/purge", json={})
            assert r_pg.status_code == 200, r_pg.text
            assert r_pg.json()["status"] == "purged", r_pg.json()
            assert not (media / ".recycle" / tid).exists()
    finally:
        engine.dispose()


def test_path_traversal_still_rejected(tmp_path: Path):
    media = tmp_path / "media"
    media.mkdir()
    outside = tmp_path / "secret.txt"
    outside.write_text("do-not-delete", encoding="utf-8")
    with pytest.raises(AppError) as exc:
        force_unlink_path(media, "../secret.txt")
    assert exc.value.code == "unsafe_file"
    assert outside.exists()
