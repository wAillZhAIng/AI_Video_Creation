"""Default studio upload, with the existing detailed preview as exception UI."""
import asyncio
import json
from pathlib import Path

from fastapi import APIRouter, Depends, File, Form, Request, UploadFile
from sqlalchemy.orm import Session

from app.core.db import get_db
from app.core.errors import AppError
from app.services import long_video_replication as long_video_service
from app.services import reference_video_analysis as ref_library
from app.services.studio_import import import_word

router=APIRouter(prefix='/api/studio',tags=['studio'])
# One process owns the studio worker. Serializing uploads prevents two tabs from
# creating the same project while an optional language request is in flight.
_import_lock=asyncio.Lock()


@router.post('/import')
async def upload(request:Request,file:UploadFile=File(...),project_id:str|None=Form(None),
        auto_start:bool=Form(False),allow_cloud_costs:bool=Form(False),allow_paid_fallback:bool=Form(False),new_revision:bool=Form(False),
        quality_json:str|None=Form(None),
        session:Session=Depends(get_db)):
    settings=request.app.state.settings
    data=await file.read(settings.import_max_bytes+1)
    if len(data)>settings.import_max_bytes:raise AppError(413,'document_too_large','文档超过大小限制')
    filename=Path((file.filename or 'document.docx').replace('\\','/')).name[:240]
    quality=None
    if quality_json:
        from app.schemas_quality import QualitySettings
        try:quality=QualitySettings.model_validate(json.loads(quality_json)).model_dump()
        except (ValueError,TypeError):raise AppError(422,'invalid_quality_settings','制作画质配置无效')
    async with _import_lock:
        return await import_word(session,settings,filename,data,project_id=project_id,
            auto_start=auto_start,allow_cloud_costs=allow_cloud_costs,allow_paid_fallback=allow_paid_fallback,new_revision=new_revision,quality=quality)


@router.post('/preview-long-video-split')
async def preview_long_video_split(
    request: Request,
    video_id: str = Form(...),
    min_shot_seconds: float = Form(2.5),
    max_shot_seconds: float = Form(8.0),
    target_shot_seconds: float = Form(5.0),
    scene_threshold: float = Form(0.25),
    reference_role: str = Form('combined'),
    session: Session = Depends(get_db),
):
    """Preview scene-cut and long-take splitting for an existing library video."""
    settings = request.app.state.settings
    return await asyncio.to_thread(
        long_video_service.preview_long_video_split,
        session,
        settings,
        video_id,
        min_shot_seconds=min_shot_seconds,
        max_shot_seconds=max_shot_seconds,
        target_shot_seconds=target_shot_seconds,
        scene_threshold=scene_threshold,
        reference_role=reference_role,
    )


@router.post('/import-long-video', status_code=201)
async def import_long_video(
    request: Request,
    file: UploadFile | None = File(None),
    video_id: str | None = Form(None),
    project_name: str = Form('', max_length=200),
    min_shot_seconds: float = Form(2.5),
    max_shot_seconds: float = Form(8.0),
    target_shot_seconds: float = Form(5.0),
    scene_threshold: float = Form(0.25),
    reference_role: str = Form('combined'),
    include_audio: bool = Form(False),
    custom_prompt_prefix: str = Form('', max_length=1200),
    analyze_with_vlm: bool = Form(False),
    quality_json: str | None = Form(None),
    session: Session = Depends(get_db),
):
    """Upload or pick a long reference video and auto-split it into a multi-shot project."""
    settings = request.app.state.settings
    quality = None
    if quality_json:
        from app.schemas_quality import QualitySettings
        try:
            quality = QualitySettings.model_validate(json.loads(quality_json)).model_dump()
        except (ValueError, TypeError):
            raise AppError(422, 'invalid_quality_settings', '制作画质配置无效')
    data = None
    filename = ''
    if file is not None:
        try:
            data = await file.read(ref_library.MAX_BYTES + 1)
            filename = Path((file.filename or 'reference.mp4').replace('\\', '/')).name[:240]
        finally:
            await file.close()
    elif not (video_id and video_id.strip()):
        raise AppError(422, 'reference_video_required', '请上传参考视频或选择已有参考视频')

    async with _import_lock:
        return await asyncio.to_thread(
            long_video_service.replicate_long_video_to_project,
            session,
            settings,
            video_id=(video_id.strip() if video_id else None),
            filename=filename,
            data=data,
            project_name=project_name,
            min_shot_seconds=min_shot_seconds,
            max_shot_seconds=max_shot_seconds,
            target_shot_seconds=target_shot_seconds,
            scene_threshold=scene_threshold,
            reference_role=reference_role,
            include_audio=include_audio,
            custom_prompt_prefix=custom_prompt_prefix,
            analyze_with_vlm=analyze_with_vlm,
            quality=quality,
        )
