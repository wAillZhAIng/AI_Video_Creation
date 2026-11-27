"""Custom LoRA scanning and project/shot on-off switches."""
from typing import Any

import httpx
from fastapi import APIRouter, Depends, Request
from sqlalchemy import select
from sqlalchemy.orm import Session
from starlette.concurrency import run_in_threadpool

from app import models as m
from app.core.db import get_db
from app.core.errors import AppError
from app.core.trusted_service_origins import local_service_client
from app.services import custom_loras as service

router = APIRouter()


def _shot_ids(session: Session, project_id: str) -> list[str]:
    if not session.get(m.Project, project_id):
        raise AppError(404, 'project_missing', '项目不存在')
    return list(session.scalars(select(m.StoryboardShot.id).where(
        m.StoryboardShot.project_id == project_id, m.StoryboardShot.status != 'deleted')))


async def _comfy_lora_names(request: Request) -> list[str] | None:
    settings = request.app.state.settings
    if getattr(settings, 'h3_demo_mode', False):
        return None
    try:
        async with local_service_client([settings.comfyui_base_url], timeout=5.0) as client:
            response = await client.get(f'{settings.comfyui_base_url}/object_info/LoraLoaderModelOnly')
            response.raise_for_status()
            options = response.json()['LoraLoaderModelOnly']['input']['required']['lora_name'][0]
            return [name for name in options if isinstance(name, str)]
    except (httpx.HTTPError, ValueError, KeyError, IndexError, TypeError):
        return None


@router.get('/api/providers/h3/loras')
async def list_custom_loras(request: Request):
    """Live scan of the ComfyUI LoRA folder with H3 base compatibility."""
    names = await _comfy_lora_names(request)
    return await run_in_threadpool(service.scan_loras, None, names)


@router.get('/api/studio/projects/{project_id}/custom-loras')
def project_custom_loras(project_id: str, request: Request, session: Session = Depends(get_db)):
    return service.project_config(request.app.state.settings, project_id, _shot_ids(session, project_id))


@router.put('/api/studio/projects/{project_id}/custom-loras')
def save_project_custom_loras(project_id: str, payload: dict[str, Any], request: Request,
                              session: Session = Depends(get_db)):
    return service.save_project_config(request.app.state.settings, project_id, payload, _shot_ids(session, project_id))
