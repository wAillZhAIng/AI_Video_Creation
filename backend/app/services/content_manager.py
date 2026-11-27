"""Dependency-aware record deletion with a durable DB-first file outbox."""
import hashlib
import json
import os
import shutil
import stat
from datetime import datetime
from graphlib import TopologicalSorter
from pathlib import Path, PurePosixPath
from sqlalchemy import select, inspect, DateTime, LargeBinary
from app import models as m
from app.core.db import Base
from app.core.errors import AppError
from app.services.screenplay import row, get
from app.services.ai_services import public_service


RESOURCES = {
 'projects':(m.Project,'项目',['name','source_text']),
 'scripts':(m.ScriptRevision,'剧本版本',['raw_text']),
 'scenes':(m.Scene,'场景',['name','order','environment_json']),
 'shots':(m.StoryboardShot,'分镜',['scene_title','visual_prompt','action','camera_move','lighting','location','duration_seconds']),
 'participants':(m.ShotCharacter,'人物参与',['performance_json','order']),
 'dialogues':(m.DialogueLine,'对白',['text','order','delivery_json']),
 'audio':(m.AudioEvent,'音频事件与素材',['description','kind','start_seconds','end_seconds','volume']),
 'characters':(m.Character,'人物档案',['display_name','description']),
 'versions':(m.CharacterVersion,'人物身份版本',[]),
 'assets':(m.CharacterAsset,'人物图片及素材',['tags_json']),
 'image_batches':(m.CharacterBatch,'人物生成批次',[]),
 'image_jobs':(m.CharacterGenerationJob,'图片生成任务',[]),
 'video_jobs':(m.GenerationJob,'视频生成任务',[]),
 'videos':(m.VideoRevision,'视频片段',[]),
 'assemblies':(m.Assembly,'合成时间线',['name']),
 'renders':(m.AssemblyRender,'合成输出版本',[]),
 'compositions':(m.FinalComposition,'旧版成片',[]),
 'imports':(m.WordImport,'Word导入记录',[]),
 'sources':(m.SourceDocument,'原始文档',[]),
 'templates':(m.ContentTemplate,'创作模板',['name','kind','body']),
 'services':(m.AIService,'AI与本地服务配置',[]),
 'bindings':(m.ProjectServiceBinding,'项目服务覆盖',[]),
 'accounts':(m.PublishingAccount,'发布账号',['display_name']),
 'presets':(m.PublishingPreset,'发布预设',['name','title_template','caption_template','hashtags_json','visibility']),
 'publish_batches':(m.PublishBatch,'发布检查批次',[]),
 'files':(m.ManagedFile,'保留文件与导出登记',[]),
 'production_entities':(m.ProductionEntity,'服装道具与场景状态版本',[]),
 'production_visuals':(m.ProductionVisual,'场景参考与镜头关键画面',[]),
 'production_runs':(m.ProductionRun,'电影生产编排',[]),
 'studio_runs':(m.StudioRun,'自动电影生产任务',[]),
 'studio_decisions':(m.StudioDecision,'统一审片决定',[]),
 'screenplay_jobs':(m.ScreenplayGenerationJob,'剧本生成与审核记录',[]),
 'screenplay_documents':(m.ScreenplayDocument,'生成Word版本',[]),
 'screenplay_inputs':(m.ScreenplayInputAsset,'剧本创作原始素材',[]),
 'character_references':(m.CharacterReferenceAsset,'项目人物参考原图',[]),
 'character_reference_sets':(m.CharacterReferenceSet,'项目人物参考绑定',[]),
 'products':(m.Product,'产品图库',['name','brand','category','notes','selling_points_json','evidence_notes','target_audience']),
 'product_images':(m.ProductImage,'产品图片',['purpose','order','notes']),
 'product_project_bindings':(m.ProjectProductBinding,'产品项目引用',[]),
 'product_shot_bindings':(m.ShotProductBinding,'镜头产品引用',[]),
 'reference_videos':(m.ReferenceVideo,'参考视频库',['name','category','notes','source_note','usage_scope','tags_json']),
 'video_analyses':(m.VideoAnalysis,'参考视频分析任务',[]),
 'style_profiles':(m.StyleProfile,'多视频风格档案',['name','user_notes']),
 'voice_library':(m.VoiceLibraryAsset,'配音库',['name','language','tags_json']),
}
TABLE_RESOURCE={value[0].__tablename__:key for key,value in RESOURCES.items()}
TABLE_RESOURCE.update({'shot_details':'_shot_details','character_aliases':'_aliases','quality_checks':'_quality'})
TABLE_RESOURCE['screenplay_project_name_claims']='_screenplay_name_claims'
MODELS={c.__table__.name:c for c in Base.registry.mappers for c in [c.class_]}
ACTIVE={'queued','running','postprocessing','submitted','submitting','cancelling','unknown','reconciling','importing','parsing','waiting_review','waiting_design_review','waiting_visual_review','waiting_video_review','waiting_service_verification','rendering','ready'}
JOB_TABLES={'generation_jobs','character_generation_jobs','assembly_renders','word_imports','production_visuals','production_runs','studio_runs',
    'media_process_tasks','quality_renders'}
JOB_TABLES.add('screenplay_generation_jobs')
JOB_TABLES.add('video_analyses')
OWNED={
 'projects':{'script_revisions','storyboard_shots','scenes','source_documents','generation_jobs','video_revisions','assemblies','final_compositions','project_service_bindings'},
 'script_revisions':{'scenes'}, 'scenes':{'shot_details','audio_events'},
 'storyboard_shots':{'shot_details','shot_characters','dialogue_lines','audio_events','generation_jobs','video_revisions'},
 'shot_characters':{'dialogue_lines'}, 'characters':{'character_versions','character_aliases','character_batches'},
 'character_versions':{'character_assets','character_batches'}, 'character_batches':{'character_generation_jobs'},
 'generation_jobs':{'video_revisions'}, 'video_revisions':{'quality_checks'}, 'assemblies':{'assembly_renders'},
 'ai_services':{'project_service_bindings'}, 'publishing_accounts':{'publishing_presets'}}
OWNED['projects'].update({'production_entities','production_visuals','production_runs'})
OWNED['scenes'].update({'production_entities','production_visuals'})
OWNED['storyboard_shots'].add('production_visuals')
OWNED['production_entities']={'production_visuals'}
OWNED['projects'].update({'studio_runs','studio_decisions'})
OWNED['storyboard_shots'].add('studio_decisions')
OWNED['projects'].update({'character_reference_sets','character_reference_assets','screenplay_project_name_claims','screenplay_generation_jobs'})
OWNED['characters'].update({'character_reference_sets','character_reference_assets'})
OWNED['screenplay_generation_jobs']={'screenplay_documents','screenplay_project_name_claims'}
OWNED['projects'].update({'project_product_bindings','shot_product_bindings','project_production_preferences','project_reference_videos'})
OWNED['storyboard_shots'].add('shot_product_bindings')
OWNED['projects'].update({'media_process_tasks','audio_assets'})
OWNED['storyboard_shots'].update({'media_process_tasks','audio_assets'})
OWNED['media_process_tasks']={'audio_assets'}
OWNED['video_revisions'].add('quality_renders')
OWNED['products']={'product_images'}
OWNED['reference_videos']={'video_analyses','project_reference_videos'}
SCOPED_OWNERS={'projects':'project','storyboard_shots':'shot'}


def _owns_child(parent, foreign_key, child, selected):
    if isinstance(child,m.QualityRender):
        # The source owns this processing task. Its resulting video revision
        # is independently selectable and must also be included if it exists.
        return foreign_key.parent.name=='source_video_id' and (
            child.video_id is None or ('video_revisions',child.video_id) in selected)
    return child.__table__.name in OWNED.get(parent,set())


def _is_active_job(obj, selected):
    table=obj.__table__.name
    if table not in JOB_TABLES:return False
    status=getattr(obj,'status','')
    # The Studio scheduler stops at review. Project-wide removal still checks
    # every owned child job separately before allowing this finished run.
    if table=='studio_runs' and status=='waiting_review' and ('projects',obj.project_id) in selected:
        return False
    return status in ACTIVE


def _ordered_records(records, *, reverse=False):
    """Order tables by FK dependencies and screenplay revisions by lineage."""
    tables={table.name:index for index,table in enumerate(Base.metadata.sorted_tables)}
    parents={entry['id']:entry['values'].get('parent_job_id') for entry in records
             if entry['table']=='screenplay_generation_jobs'}
    dependencies={identifier:[parent] if parent in parents else [] for identifier,parent in parents.items()}
    lineage={identifier:index for index,identifier in enumerate(TopologicalSorter(dependencies).static_order())}
    return sorted(records,key=lambda entry:(tables[entry['table']],
        lineage[entry['id']] if entry['table']=='screenplay_generation_jobs' else 0),reverse=reverse)


def serialize(value):
    return json.loads(json.dumps(value,default=lambda x:x.isoformat() if isinstance(x,datetime) else None,ensure_ascii=False))


def label(obj):
    return str(next((getattr(obj,key) for key in ['name','display_name','scene_title','filename','relative_path','text'] if getattr(obj,key,None)),obj.id))[:240]


def public_record(obj):
    data=public_service(obj) if isinstance(obj,m.AIService) else serialize(row(obj))
    for key in ('credential_reference','encrypted_secret'): data.pop(key,None)
    return data


def safe_file(root,relative):
    parts=PurePosixPath(relative.replace('\\','/'))
    if parts.is_absolute() or '..' in parts.parts or ':' in relative or not parts.parts:
        raise AppError(422,'unsafe_file','文件路径非法，已保护')
    root=Path(root).absolute()
    path=root.joinpath(*parts.parts)
    for parent in [root,*path.parents,path]:
        if parent==root or root in parent.parents:
            if parent.is_symlink() or (hasattr(parent,'is_junction') and parent.is_junction()):
                raise AppError(422,'unsafe_link','符号链接或目录联接中的文件已保护')
    if not path.resolve().is_relative_to(root.resolve()): raise AppError(422,'unsafe_file','文件路径越界，已保护')
    if path.exists() and not path.is_file(): raise AppError(422,'not_regular_file','只允许处理普通文件')
    if path.exists() and path.stat().st_nlink>1: raise AppError(422,'shared_hardlink','多重硬链接文件已保护')
    return path


def _make_writable(path: Path) -> None:
    try:
        if path.exists() and not path.is_symlink():
            mode = path.stat().st_mode
            os.chmod(path, mode | stat.S_IWUSR | stat.S_IWGRP | stat.S_IWOTH | (stat.S_IXUSR if path.is_dir() else 0))
    except OSError:
        pass


def _rmtree_onerror(func, p, _exc_info):
    try:
        os.chmod(p, 0o777)
        func(p)
    except OSError:
        pass


def safe_media_target(root, relative):
    """Resolve a path strictly inside media_root without rejecting symlinks/hardlinks."""
    parts = PurePosixPath(str(relative).replace('\\', '/'))
    if parts.is_absolute() or '..' in parts.parts or ':' in str(relative) or not parts.parts:
        raise AppError(422, 'unsafe_file', '文件路径非法，已保护')
    root_abs = Path(root).resolve()
    path = root_abs.joinpath(*parts.parts)
    try:
        parent_resolved = path.parent.resolve()
        if not (parent_resolved == root_abs or root_abs in parent_resolved.parents):
            raise AppError(422, 'unsafe_file', '文件路径越界，已保护')
    except OSError:
        raise AppError(422, 'unsafe_file', '文件路径越界，已保护')
    return path


def force_unlink_path(root, relative) -> tuple[str, int]:
    """Highest-privilege file/dir removal inside media_root; never refuses read-only/symlink/hardlink."""
    path = safe_media_target(root, relative)
    if not path.exists() and not path.is_symlink():
        return 'missing', 0
    size = 0
    try:
        if path.is_symlink():
            _make_writable(path.parent)
            path.unlink(missing_ok=True)
            return 'deleted', 0
        if path.is_dir():
            for child in path.rglob('*'):
                if child.is_file() and not child.is_symlink():
                    try:
                        size += child.stat().st_size
                    except OSError:
                        pass
            _make_writable(path.parent)
            _make_writable(path)
            shutil.rmtree(path, onerror=_rmtree_onerror)
        else:
            size = path.stat().st_size
            _make_writable(path.parent)
            _make_writable(path)
            path.unlink(missing_ok=True)
    except OSError:
        _make_writable(path.parent)
        _make_writable(path)
        if path.is_dir():
            shutil.rmtree(path, ignore_errors=True)
        else:
            path.unlink(missing_ok=True)
    root_abs = Path(root).resolve()
    cur = path.parent
    while cur != root_abs and root_abs in cur.parents:
        try:
            cur.rmdir()
            cur = cur.parent
        except OSError:
            break
    return 'deleted', size


OWNED_DIR_TEMPLATES = {
    'projects': [
        'projects/{id}',
        'quality/{id}',
        'media_processing/{id}',
        'production_visuals/{id}',
        'audio_assets/{id}',
        'character_references/{id}',
        'assemblies/{id}',
        'reference_video_slices/{id}',
        'revision_references/{id}',
    ],
    'characters': ['characters/{id}'],
    'products': ['product_library/{id}', 'asset_library/products/{id}'],
    'reference_videos': ['asset_library/reference_videos/{id}', 'reference_videos/{id}'],
    'voice_library_assets': ['asset_library/voices/{id}', 'voice_library/{id}'],
    'screenplay_generation_jobs': ['screenplay_jobs/{id}'],
    'word_imports': ['word_imports/{id}'],
}


def _owned_record_directories(records) -> list[str]:
    dirs: list[str] = []
    seen: set[str] = set()
    for rec in records:
        table = rec['table']
        rid = str(rec['id'])
        if not rid or '/' in rid or '\\' in rid or '..' in rid:
            continue
        for tmpl in OWNED_DIR_TEMPLATES.get(table, []):
            rel = tmpl.format(id=rid)
            if rel not in seen:
                seen.add(rel)
                dirs.append(rel)
    return dirs


def file_entries(obj):
    table=obj.__table__.name
    path=getattr(obj,'relative_path',None)
    paths=[]
    if path: paths.append(path)
    if isinstance(obj,m.ProductImage) and obj.thumbnail_relative_path:paths.append(obj.thumbnail_relative_path)
    if isinstance(obj,m.VoiceLibraryAsset) and obj.preview_relative_path:paths.append(obj.preview_relative_path)
    if isinstance(obj,m.CharacterBatch) and obj.contact_sheet_path: paths.append(obj.contact_sheet_path)
    if isinstance(obj,m.VideoRevision): paths.extend(obj.thumbnail_paths_json or [])
    if isinstance(obj,m.ScreenplayGenerationJob):
        source=(obj.receipt_json or {}).get('source') or {}
        derivatives=[frame.get('relative_path') for frame in (source.get('frames') or []) if isinstance(frame,dict)]
        derivatives.append(source.get('audio_relative_path'))
        directory=PurePosixPath('screenplay_jobs',obj.id,'source_frames')
        paths.extend(relative for relative in derivatives if isinstance(relative,str)
                     and PurePosixPath(relative).parent==directory)
    for p in paths:
        owned=False;reason='文件归属未验证'
        if table=='character_assets':
            job=(obj.metadata_json or {}).get('generation_job_id','')
            owned=obj.source=='generated' and bool(job) and job in p
            reason='原始上传或归属未验证' if not owned else ''
        elif table=='video_revisions':
            quality_render=(obj.metadata_json or {}).get('quality_render_id')
            owned=bool((obj.generation_job_id and obj.generation_job_id in p)
                       or (quality_render and p==f'quality/{obj.project_id}/{quality_render}.mp4'))
        elif table=='quality_renders':
            owned=p==f"quality/{(obj.snapshot_json or {}).get('project_id')}/{obj.id}.mp4"
        elif table=='audio_assets':
            path=PurePosixPath(p)
            owned=bool(obj.task_id and path.parent==PurePosixPath('media_processing',obj.project_id)
                       and (path.stem==obj.task_id or path.stem.startswith(obj.task_id+'.')))
        elif table=='assembly_renders': owned=obj.id in p
        elif table=='final_compositions': owned=obj.id in p
        elif table=='character_batches': owned=obj.id in p
        elif table=='managed_files':
            owned=obj.state in ('active','retained','cleanup_failed') and obj.metadata_json.get('verified_owned',False)
            reason='用户原始文件或受保护登记' if not owned else ''
        elif table=='production_visuals':
            owned=obj.id in p and obj.snapshot_json.get('source')!='uploaded'
            reason='原始上传视觉参考受保护' if not owned else ''
        elif table=='word_imports': reason='用户上传原始Word保留'
        elif table in {'screenplay_input_assets','character_reference_assets'}:
            reason='用户创作或人物参考原图保留；解除关系不删除共享原文件'
            if table=='screenplay_input_assets' and obj.metadata_json.get('library_owner')=='product_library':owned=obj.id in p
        elif table in {'product_images','voice_library_assets','reference_videos'}:
            owned=obj.id in p
            reason='图库受控副本归属未验证' if not owned else ''
        elif table=='screenplay_documents':
            owned=obj.id in p
            reason='生成Word归属未验证' if not owned else ''
        elif table=='screenplay_generation_jobs':
            owned=True  # Selected above from this job's own derivative receipt paths.
        yield {'path':p,'owned':owned,'protected_reason':reason if not owned else ''}


def find_references(session):
    index={}
    models=[model for resource,(model,_,_) in RESOURCES.items() if resource!='files']
    for model in [*models,m.QualityRender,m.AudioAsset]:
        for obj in session.scalars(select(model)):
            for entry in file_entries(obj): index.setdefault(entry['path'],[]).append((obj.__table__.name,obj.id,entry))
    return index


def build_plan(session,settings,resource,ids,mode,cascade=True,force=True):
    if resource not in RESOURCES: raise AppError(404,'resource_unknown','不支持管理此资源')
    if force:
        cascade = True
    model=RESOURCES[resource][0]
    selected={(model.__tablename__,identifier):get(session,model,identifier) for identifier in sorted(set(ids))}
    dependencies=[];blockers=[];related_files=[]
    # Fixed point over explicit ownership edges. All other foreign references block.
    changed=True
    while changed:
        changed=False
        for (table,identifier),obj in list(selected.items()):
            if cascade and table in SCOPED_OWNERS:
                for scoped in (m.QualityProfile,m.RoutePolicy):
                    for child in session.scalars(select(scoped).where(
                        scoped.scope==SCOPED_OWNERS[table],scoped.target_id==identifier)):
                        key=(scoped.__tablename__,child.id)
                        if key not in selected:selected[key]=child;changed=True
            if isinstance(obj,m.ProductImage) and mode!='keep' and cascade:
                asset=session.get(m.ScreenplayInputAsset,obj.asset_id)
                other=list(session.scalars(select(m.ProductImage).where(m.ProductImage.asset_id==obj.asset_id)))
                if asset and asset.metadata_json.get('library_owner')=='product_library' and all(('product_images',image.id) in selected for image in other):
                    key=('screenplay_input_assets',asset.id)
                    if key not in selected:selected[key]=asset;changed=True
            for child_table in Base.metadata.tables.values():
                cls=MODELS[child_table.name]
                for fk in child_table.foreign_keys:
                    if fk.column.table.name!=table or fk.column.name!='id':continue
                    for child in session.scalars(select(cls).where(getattr(cls,fk.parent.name)==identifier)):
                        key=(child_table.name,child.id)
                        if key in selected:continue
                        if cascade and (_owns_child(table,fk,child,selected) or (force and not fk.parent.nullable)):
                            selected[key]=child;changed=True
                            if isinstance(child,m.ShotDetail) and table=='scenes':
                                shot=session.get(m.StoryboardShot,child.shot_id)
                                if shot:selected[('storyboard_shots',shot.id)]=shot
    for registry in session.scalars(select(m.ManagedFile)):
        owners={(r.get('table'),r.get('id')) for r in registry.ownership_json or []}
        if owners and owners.intersection(selected) and ('managed_files',registry.id) not in selected:
            related_files.append((('managed_files',registry.id),registry))
            if owners.issubset(selected) and mode!='keep':
                if cascade:selected[('managed_files',registry.id)]=registry
                else:blockers.append({'id':registry.id,'reason':'关联文件登记需要明确级联处理，文件模式仍按所选策略执行'})
    for (table,identifier),obj in list(selected.items()):
        if table in {'products','product_images','voice_library_assets','reference_videos','video_analyses','style_profiles'}:
            from app.services.product_library import contains
            if table=='reference_videos':
                for job in session.scalars(select(m.GenerationJob)):
                    if ('generation_jobs',job.id) not in selected and any(isinstance(ref,dict) and ref.get('video_id')==identifier for ref in (job.parameters_json or {}).get('motion_references',[])):
                        blockers.append({'id':job.id,'reason':'仍被视频生成任务的动作参考快照引用'})
            for job in session.scalars(select(m.ScreenplayGenerationJob)):
                if ('screenplay_generation_jobs',job.id) not in selected and contains(job.snapshot_json,identifier):blockers.append({'id':job.id,'reason':'仍被剧本生成任务或其冻结来源引用'})
            for run in session.scalars(select(m.StudioRun)):
                if ('studio_runs',run.id) not in selected and contains([run.snapshot_json,run.nodes_json,run.result_json],identifier):blockers.append({'id':run.id,'reason':'仍被自动制作或历史快照引用'})
            if table=='voice_library_assets':
                for audio in session.scalars(select(m.AudioAsset)):
                    if audio.metadata_json.get('voice_library_asset_id')==identifier and ('audio_assets',audio.id) not in selected:blockers.append({'id':audio.id,'reason':'配音仍被项目音频引用；请先在角色设置解除绑定并处理项目音频'})
            if table in {'reference_videos','video_analyses'}:
                for profile in session.scalars(select(m.StyleProfile)):
                    if ('style_profiles',profile.id) not in selected and identifier in profile.video_ids_json+profile.analysis_ids_json:blockers.append({'id':profile.id,'reason':'仍被风格档案引用'})
        resource_key=TABLE_RESOURCE.get(table)
        if resource_key:
            note_resources=[resource_key]+(['product_file_analysis'] if resource_key=='products' else [])
            for note in session.scalars(select(m.ContentNote).where(m.ContentNote.resource.in_(note_resources),m.ContentNote.record_id==identifier)):
                selected[('content_notes',note.id)]=note
    for (table,identifier),obj in list(selected.items()):
        if _is_active_job(obj,selected):
            blockers.append({'resource':TABLE_RESOURCE.get(table,table),'id':identifier,'reason':'任务仍在运行或排队，请先取消'})
        if not cascade and table in SCOPED_OWNERS:
            for scoped in (m.QualityProfile,m.RoutePolicy):
                for child in session.scalars(select(scoped).where(
                    scoped.scope==SCOPED_OWNERS[table],scoped.target_id==identifier)):
                    blockers.append({'resource':scoped.__tablename__,'id':child.id,'parent_id':identifier,'reason':'需要明确级联处理子记录'})
        for detail in session.scalars(select(m.ShotDetail)):
            if identifier in str(detail.source_json.get('production',{})) and ('storyboard_shots',detail.shot_id) not in selected:
                blockers.append({'id':detail.shot_id,'reason':'仍被镜头制作版本或视觉参考引用，请先解除绑定'})
        for run in session.scalars(select(m.ProductionRun)):
            if identifier!=run.id and ('production_runs',run.id) not in selected and identifier in str([run.snapshot_json,run.steps_json,run.result_json]):
                blockers.append({'id':run.id,'reason':'仍被电影制作编排或其历史快照引用'})
        if isinstance(obj,m.ManagedFile):
            for owner in obj.ownership_json or []:
                owner_cls=MODELS.get(owner.get('table'));parent=session.get(owner_cls,owner.get('id')) if owner_cls else None
                if parent and (owner['table'],parent.id) not in selected:
                    dependencies.append({'resource':TABLE_RESOURCE.get(owner['table'],owner['table']),'id':parent.id,'reason':'此文件属于该记录；删除文件登记不会删除其所属记录'})
                    if _is_active_job(parent,selected):blockers.append({'id':parent.id,'reason':'运行中的任务仍拥有此文件，请先取消任务'})
        affected_shots=[]
        if isinstance(obj,(m.ShotCharacter,m.DialogueLine)) or isinstance(obj,m.AudioEvent) and obj.shot_id:affected_shots=[obj.shot_id]
        elif isinstance(obj,m.AudioEvent) and obj.scene_id:affected_shots=list(session.scalars(select(m.ShotDetail.shot_id).where(m.ShotDetail.scene_id==obj.scene_id)))
        for shot_id in affected_shots:
            if ('storyboard_shots',shot_id) not in selected:
                dependencies.append({'id':shot_id,'reason':'内容删除后，此分镜及已有视频需要重新审核'})
                if session.scalar(select(m.GenerationJob.id).where(m.GenerationJob.shot_id==shot_id,m.GenerationJob.status.in_(ACTIVE))):blockers.append({'id':shot_id,'reason':'运行中的分镜仍在使用该内容，请先取消任务'})
        for child_table in Base.metadata.tables.values():
            cls=MODELS[child_table.name]
            for fk in child_table.foreign_keys:
                if fk.column.table.name!=table: continue
                for child in session.scalars(select(cls).where(getattr(cls,fk.parent.name)==identifier)):
                    if (child_table.name,child.id) in selected:continue
                    item={'resource':TABLE_RESOURCE.get(child_table.name,child_table.name),'id':child.id,'label':label(child),'parent_id':identifier}
                    dependencies.append(item)
                    blockers.append({**item,'reason':'需要明确级联处理子记录' if _owns_child(table,fk,child,selected) else '仍被其他业务引用，请先解除或重新关联'})
        if isinstance(obj,m.CharacterAsset):
            for v in session.scalars(select(m.CharacterVersion).where(m.CharacterVersion.standard_face_asset_id==identifier)):
                if ('character_versions',v.id) not in selected:blockers.append({'id':v.id,'reason':'仍是人物标准脸，请建立新版本或解除引用'})
            for cls,field in [(m.StoryboardShot,'reference_asset_ids_json'),(m.ShotCharacter,'reference_asset_ids_json')]:
                for p in session.scalars(select(cls)):
                    if identifier in (getattr(p,field) or []) and (cls.__tablename__,p.id) not in selected:blockers.append({'id':p.id,'reason':'仍被分镜参考素材引用'})
        if isinstance(obj,m.CharacterVersion):
            for character in session.scalars(select(m.Character).where(m.Character.current_approved_version_id==identifier)):
                if ('characters',character.id) not in selected:blockers.append({'id':character.id,'reason':'仍是人物当前正式版本，请先重新指定正式版本或管理整个人物'})
        if isinstance(obj,m.VideoRevision):
            for assembly in session.scalars(select(m.Assembly)):
                if any(c.get('video_id')==identifier for c in assembly.timeline_json.get('clips',[])) and ('assemblies',assembly.id) not in selected:
                    blockers.append({'id':assembly.id,'reason':'仍被合成时间线使用'})
            for final in session.scalars(select(m.FinalComposition)):
                if identifier in final.video_ids_json and ('final_compositions',final.id) not in selected:blockers.append({'id':final.id,'reason':'仍被旧版成片引用'})
        if isinstance(obj,m.AIService):
            for cls in (m.GenerationJob,m.CharacterGenerationJob,m.AssemblyRender,m.ScreenplayGenerationJob):
                for job in session.scalars(select(cls)):
                    content=serialize(row(job))
                    if identifier in json.dumps(content) and job.status in ACTIVE:blockers.append({'id':job.id,'reason':'运行任务仍在使用此服务'})
            dependencies.append({'id':identifier,'reason':'默认配置' if obj.is_default else '服务配置','credential_cleanup':bool(obj.encrypted_secret),'history':'历史任务配置快照保留，密钥不进入快照'})
    references=find_references(session);files={}
    for (table,identifier),obj in list(selected.items())+[item for item in related_files if item[0] not in selected]:
        for entry in file_entries(obj):
            p=entry['path']
            external=[{'table':t,'id':i} for t,i,_ in references.get(p,[]) if (t,i) not in selected]
            if isinstance(obj,m.ManagedFile):
                owners=[ref for ref in obj.ownership_json if ref.get('table') in MODELS and session.get(MODELS[ref['table']],ref.get('id'))]
                if len(owners)>1:external.extend(ref for ref in owners if (ref.get('table'),ref.get('id')) not in selected)
            if p in files: continue
            protected=entry['protected_reason']; size=0;mtime=None;exists=False
            try:
                path=safe_media_target(settings.media_root,p);exists=path.exists() or path.is_symlink()
                if exists and not path.is_symlink() and path.is_file():size=path.stat().st_size;mtime=path.stat().st_mtime_ns
            except AppError as exc:protected=exc.message
            if external:protected='其他有效记录共用'
            # Historical generation reference snapshots also keep source images alive.
            for cls,field in [(m.GenerationJob,'reference_assets_json'),(m.CharacterGenerationJob,'snapshot_json'),
                              (m.ScreenplayGenerationJob,'snapshot_json'),(m.ScreenplayGenerationJob,'receipt_json')]:
                for job in session.scalars(select(cls)):
                    if (cls.__tablename__,job.id) not in selected and p in json.dumps(getattr(job,field),ensure_ascii=False):protected='历史或运行任务仍引用此文件'
            files[p]={'path':p,'bytes':size,'mtime_ns':mtime,'exists':exists,'protected':bool(protected),'reason':protected,'shared_references':external,'verified_owned':entry['owned']}
    credentials=[]
    from app.core.secrets import KEYCHAIN_PREFIX
    for (table,identifier),obj in selected.items():
        if isinstance(obj,m.AIService) and obj.encrypted_secret:
            shared=any(other.id!=identifier and other.encrypted_secret==obj.encrypted_secret and ('ai_services',other.id) not in selected for other in session.scalars(select(m.AIService)))
            credentials.append({'record_id':identifier,'shared':shared,'storage':'keychain' if obj.encrypted_secret.startswith(KEYCHAIN_PREFIX) else 'dpapi',
                'account':obj.encrypted_secret[len(KEYCHAIN_PREFIX):].decode('ascii') if obj.encrypted_secret.startswith(KEYCHAIN_PREFIX) else None})
    snapshots=[]
    for (table,identifier),obj in selected.items():
        values=serialize(row(obj))
        for col in obj.__table__.columns:
            if isinstance(col.type,LargeBinary) or col.name=='credential_reference': values[col.name]=None
        snapshots.append({'table':table,'id':identifier,'resource':TABLE_RESOURCE.get(table,table),'label':label(obj),'values':values})
    snapshots.sort(key=lambda x:(x['table'],x['id']))
    if force:
        for b in blockers:
            dependencies.append({**b, 'auto_resolved': True})
        blockers = []
        if mode != 'keep':
            root_abs = Path(settings.media_root).resolve()
            for dir_rel in _owned_record_directories(snapshots):
                try:
                    dpath = safe_media_target(settings.media_root, dir_rel)
                except AppError:
                    continue
                if dpath.exists() and dpath.is_dir():
                    for child in dpath.rglob('*'):
                        if child.is_file() or child.is_symlink():
                            try:
                                rel_child = child.relative_to(root_abs).as_posix()
                            except ValueError:
                                continue
                            if rel_child not in files:
                                csize = 0
                                cmtime = None
                                if not child.is_symlink():
                                    try:
                                        st = child.stat()
                                        csize = st.st_size
                                        cmtime = st.st_mtime_ns
                                    except OSError:
                                        pass
                                files[rel_child] = {
                                    'path': rel_child, 'bytes': csize, 'mtime_ns': cmtime,
                                    'exists': True, 'protected': False, 'reason': '',
                                    'shared_references': [], 'verified_owned': True,
                                }
        for fentry in files.values():
            fentry['protected'] = False
            fentry['verified_owned'] = True
            if fentry.get('reason'):
                fentry['reason'] = ''
    plan={'resource':resource,'ids':sorted(set(ids)),'mode':mode,'cascade':cascade,'force':force,'records':snapshots,'credential_cleanup':credentials,
        'dependencies':dependencies,'blockers':blockers,'files':sorted(files.values(),key=lambda x:x['path']),
        'record_count':len(snapshots),'file_count':len(files),'total_bytes':sum(x['bytes'] for x in files.values()),
        'exclusive_bytes':sum(x['bytes'] for x in files.values() if not x['protected']),
        'recoverable':mode=='trash','space_note':'回收站仍占用磁盘空间，清空后释放' if mode=='trash' else '最高权限强制删除：自动解除关联、终止任务并彻底释放全部关联磁盘文件'}
    plan['fingerprint']=hashlib.sha256(json.dumps(plan,ensure_ascii=False,sort_keys=True).encode()).hexdigest()
    return plan


def public_plan(plan):
    return {**plan,'records':[{k:v for k,v in r.items() if k!='values'} for r in plan['records']], 'credential_cleanup':[{k:v for k,v in c.items() if k!='account'} for c in plan.get('credential_cleanup',[])]}


def task_public(task):
    return {'id':task.id,'status':task.status,'mode':task.mode,'plan':public_plan(task.plan_json),'result':task.result_json,'error':task.error_message}


def file_outbox(session,settings,task):
    results=dict(task.result_json or {});states=dict(results.get('files',{}))
    for file in task.plan_json['files']:
        relative=file['path']
        if states.get(relative,{}).get('status') in ('kept','protected','missing','trashed','deleted'):continue
        outcome={'bytes':0}
        try:
            if task.mode=='keep':
                outcome={'status':'kept','bytes':0}
            elif task.mode=='trash':
                path=safe_media_target(settings.media_root,relative)
                recycle=safe_media_target(settings.media_root,'.recycle/'+task.id+'/'+relative)
                if not path.exists() and not path.is_symlink():
                    outcome={'status':'trashed' if recycle.exists() else 'missing','bytes':0}
                else:
                    recycle.parent.mkdir(parents=True,exist_ok=True)
                    if recycle.exists() or recycle.is_symlink():
                        force_unlink_path(settings.media_root,'.recycle/'+task.id+'/'+relative)
                    _make_writable(path.parent);_make_writable(path)
                    shutil.move(str(path),str(recycle))
                    outcome={'status':'trashed','bytes':0,'recycle_path':'.recycle/'+task.id+'/'+relative}
            else:
                st_code, released_size = force_unlink_path(settings.media_root, relative)
                outcome={'status':st_code,'bytes':released_size}
        except (AppError,OSError) as exc:outcome={'status':'failed','bytes':0,'error':exc.message if isinstance(exc,AppError) else type(exc).__name__+': '+str(exc)}
        states[relative]=outcome
        registry=session.scalar(select(m.ManagedFile).where(m.ManagedFile.relative_path==relative))
        if registry:registry.state={'kept':'retained','protected':'retained','missing':'missing','trashed':'trashed','deleted':'deleted','failed':'cleanup_failed'}[outcome['status']]
        results['files']=states;task.result_json=dict(results);session.commit()
    if task.mode == 'permanent':
        for dir_rel in _owned_record_directories(task.plan_json.get('records', [])):
            try:
                st_code, dir_bytes = force_unlink_path(settings.media_root, dir_rel)
                if st_code == 'deleted' and dir_bytes > 0 and dir_rel not in states:
                    states[dir_rel] = {'status': 'deleted', 'bytes': dir_bytes}
            except (AppError, OSError):
                pass
        results['files'] = states
    results['verified_released_bytes']=sum(x.get('bytes',0) for x in states.values() if x['status']=='deleted')
    results['failed_files']=sum(x['status']=='failed' for x in states.values())
    credential_results=[]
    from app.core.secrets import delete_keychain_reference
    for credential in task.plan_json.get('credential_cleanup',[]):
        try:
            if credential['shared']:status='shared_reference_preserved'
            elif credential['storage']=='keychain':delete_keychain_reference(credential['account']);status='keychain_deleted_verified'
            else:
                if session.get(m.AIService,credential['record_id']):raise AppError(409,'credential_record_exists','配置记录仍存在，尚未完成密钥删除')
                status='dpapi_record_removed_verified'
            credential_results.append({'record_id':credential['record_id'],'status':status})
        except AppError as exc:credential_results.append({'record_id':credential['record_id'],'status':'failed','error':exc.message})
    results['credentials']=credential_results
    results['failed_credentials']=sum(c['status']=='failed' for c in credential_results)
    task.result_json=results;task.status='files_failed' if results['failed_files'] or results['failed_credentials'] else 'completed'
    task.error_message='部分文件或凭据清理失败，可重试；清理清单仍保留' if results['failed_files'] or results['failed_credentials'] else None
    session.commit();return task_public(task)


def _force_unlink_external_references(session, selected):
    """Force-cancel active jobs and clear all external & circular FK/JSON references before row deletion."""
    selected_by_table: dict[str, set[str]] = {}
    for table, identifier in selected:
        selected_by_table.setdefault(table, set()).add(identifier)

    # 1. Cancel all active jobs inside selected records
    for table in JOB_TABLES:
        for identifier in selected_by_table.get(table, set()):
            obj = session.get(MODELS[table], identifier)
            if obj and getattr(obj, 'status', '') in ACTIVE:
                obj.status = 'cancelled'

    # 2. Clear soft/JSON references on surviving rows
    if 'character_versions' in selected_by_table:
        v_ids = selected_by_table['character_versions']
        for ch in session.scalars(select(m.Character).where(m.Character.current_approved_version_id.in_(v_ids))):
            ch.current_approved_version_id = None
    if 'character_assets' in selected_by_table:
        a_ids = selected_by_table['character_assets']
        for cv in session.scalars(select(m.CharacterVersion).where(m.CharacterVersion.standard_face_asset_id.in_(a_ids))):
            cv.standard_face_asset_id = None
        for cls in (m.StoryboardShot, m.ShotCharacter):
            for row_obj in session.scalars(select(cls)):
                if (cls.__tablename__, row_obj.id) not in selected and row_obj.reference_asset_ids_json:
                    filtered = [x for x in row_obj.reference_asset_ids_json if x not in a_ids]
                    if len(filtered) != len(row_obj.reference_asset_ids_json):
                        row_obj.reference_asset_ids_json = filtered
    if 'video_revisions' in selected_by_table:
        vr_ids = selected_by_table['video_revisions']
        for asm in session.scalars(select(m.Assembly)):
            if ('assemblies', asm.id) not in selected and isinstance(asm.timeline_json, dict):
                clips = asm.timeline_json.get('clips', [])
                filtered = [c for c in clips if c.get('video_id') not in vr_ids]
                if len(filtered) != len(clips):
                    asm.timeline_json = {**asm.timeline_json, 'clips': filtered}
        for fc in session.scalars(select(m.FinalComposition)):
            if ('final_compositions', fc.id) not in selected and fc.video_ids_json:
                filtered = [x for x in fc.video_ids_json if x not in vr_ids]
                if len(filtered) != len(fc.video_ids_json):
                    fc.video_ids_json = filtered
    if 'reference_videos' in selected_by_table or 'video_analyses' in selected_by_table:
        rv_ids = selected_by_table.get('reference_videos', set())
        va_ids = selected_by_table.get('video_analyses', set())
        for sp in session.scalars(select(m.StyleProfile)):
            if ('style_profiles', sp.id) not in selected:
                sp.video_ids_json = [x for x in (sp.video_ids_json or []) if x not in rv_ids]
                sp.analysis_ids_json = [x for x in (sp.analysis_ids_json or []) if x not in va_ids]
        if rv_ids:
            for job in session.scalars(select(m.GenerationJob)):
                if ('generation_jobs', job.id) not in selected and isinstance(job.parameters_json, dict):
                    mrefs = job.parameters_json.get('motion_references')
                    if isinstance(mrefs, list):
                        cleaned = [x for x in mrefs if not (isinstance(x, dict) and x.get('video_id') in rv_ids)]
                        if len(cleaned) != len(mrefs):
                            job.parameters_json = {**job.parameters_json, 'motion_references': cleaned}
            for pref in session.scalars(select(m.ProjectProductionPreference)):
                if ('project_production_preferences', pref.id) not in selected and pref.reference_video_defaults_json:
                    pref.reference_video_defaults_json = [
                        item for item in pref.reference_video_defaults_json
                        if not (isinstance(item, dict) and item.get('video_id') in rv_ids)
                    ]
            for detail in session.scalars(select(m.ShotDetail)):
                if ('storyboard_shots', detail.shot_id) not in selected and isinstance(detail.source_json, dict):
                    prod = dict(detail.source_json.get('production') or {})
                    changed = False
                    for key in ('motion_references', 'reference_video_overrides'):
                        if isinstance(prod.get(key), list):
                            cleaned = [x for x in prod[key] if not (isinstance(x, dict) and x.get('video_id') in rv_ids)]
                            if len(cleaned) != len(prod[key]):
                                prod[key] = cleaned
                                changed = True
                    if changed:
                        detail.source_json = {**detail.source_json, 'production': prod}
    if 'voice_library_assets' in selected_by_table:
        vl_ids = selected_by_table['voice_library_assets']
        for audio in session.scalars(select(m.AudioAsset)):
            if ('audio_assets', audio.id) not in selected and isinstance(audio.metadata_json, dict):
                if audio.metadata_json.get('voice_library_asset_id') in vl_ids:
                    meta = dict(audio.metadata_json)
                    meta.pop('voice_library_asset_id', None)
                    audio.metadata_json = meta

    # 3. Null out ALL nullable FKs pointing into selected (both external and circular self-refs)
    for table, ids_set in selected_by_table.items():
        for child_table in Base.metadata.tables.values():
            cls = MODELS.get(child_table.name)
            if not cls:
                continue
            for fk in child_table.foreign_keys:
                if fk.column.table.name != table or not fk.parent.nullable:
                    continue
                for child in session.scalars(select(cls).where(getattr(cls, fk.parent.name).in_(ids_set))):
                    setattr(child, fk.parent.name, None)
    session.flush()


def execute_delete(session,settings,payload):
    if not getattr(payload, 'confirm', True):
        raise AppError(409,'delete_confirmation','请明确确认删除操作')
    req_fp = getattr(payload, 'fingerprint', '') or ''
    if req_fp:
        existing = session.scalar(select(m.DeletionTask).where(m.DeletionTask.fingerprint == req_fp))
        if existing:
            return task_public(existing)
    plan=build_plan(session,settings,payload.resource,payload.ids,payload.mode,cascade=True,force=True)
    task_fp = req_fp if req_fp else f"{plan['fingerprint']}-{os.urandom(4).hex()}"
    if session.scalar(select(m.DeletionTask.id).where(m.DeletionTask.fingerprint == task_fp)):
        task_fp = f"{plan['fingerprint']}-{os.urandom(4).hex()}"
    task=m.DeletionTask(status='planned',mode=payload.mode,fingerprint=task_fp,plan_json=plan)
    session.add(task);session.commit()
    try:
        if session.bind.dialect.name=='sqlite':session.connection().exec_driver_sql('BEGIN IMMEDIATE')
        selected={(r['table'],r['id']) for r in plan['records']}
        _force_unlink_external_references(session, selected)
        for registry in session.scalars(select(m.ManagedFile)):
            previous=registry.ownership_json or []
            remaining=[ref for ref in previous if (ref.get('table'),ref.get('id')) not in selected]
            if remaining!=previous:
                registry.ownership_json=remaining
                if not remaining:registry.state='retained'
        for f in plan['files']:
            record=session.scalar(select(m.ManagedFile).where(m.ManagedFile.relative_path==f['path']))
            if not record:record=m.ManagedFile(relative_path=f['path']);session.add(record)
            record.state='retained' if payload.mode=='keep' or f['protected'] else 'deletion_pending'
            record.ownership_json=[{'table':t,'id':i} for t,i,_ in find_references(session).get(f['path'],[]) if (t,i) not in selected]
            record.metadata_json={'deletion_task_id':task.id,'verified_owned':f['verified_owned'],'reason':f['reason'],'original_bytes':f['bytes']}
        # Reverse FK order; rollback leaves every file untouched on database failure.
        from app.services.screenplay import touch_shot
        invalidated=set()
        for record in plan['records']:
            if record['table'] not in ('shot_characters','dialogue_lines','audio_events'):continue
            obj=session.get(MODELS[record['table']],record['id'])
            shot_ids=[obj.shot_id] if obj.shot_id else list(session.scalars(select(m.ShotDetail.shot_id).where(m.ShotDetail.scene_id==obj.scene_id)))
            for shot_id in shot_ids:
                if ('storyboard_shots',shot_id) in selected:continue
                shot=session.get(m.StoryboardShot,shot_id)
                if shot_id not in invalidated:shot=touch_shot(session,shot_id);invalidated.add(shot_id)
                if isinstance(obj,m.ShotCharacter) and shot.character_id==obj.character_id:
                    shot.character_id=None;shot.character_asset_version_id=None;shot.reference_asset_ids_json=[]
        removed_product_images={record['id'] for record in plan['records'] if record['table']=='product_images'}
        affected_products={image.product_id for identifier in removed_product_images if (image:=session.get(m.ProductImage,identifier))}
        for identifier in affected_products:
            if ('products',identifier) in selected:continue
            product=session.get(m.Product,identifier)
            if not product:continue
            remaining=list(session.scalars(select(m.ProductImage).where(m.ProductImage.product_id==identifier,m.ProductImage.id.not_in(removed_product_images)).order_by(m.ProductImage.order,m.ProductImage.id)))
            if product.cover_image_id in removed_product_images:product.cover_image_id=remaining[0].id if remaining else None
            product.revision+=1
        for item in _ordered_records(plan['records'],reverse=True):
            obj=session.get(MODELS[item['table']],item['id'])
            if obj:session.delete(obj);session.flush()
        task.status='files_pending';task.result_json={'records_deleted':len(plan['records']),'files':{}}
        session.add(m.AuditEvent(object_type='deletion_task',object_id=task.id,action='records_deleted',details_json={'mode':payload.mode,'count':len(plan['records'])}))
        session.commit()
        lora_cfg_root = Path(settings.media_root).parent / 'custom_lora_configs'
        for r in plan['records']:
            if r['table'] == 'projects':
                cfg = lora_cfg_root / f"{r['id']}.json"
                if cfg.exists():
                    try:
                        _make_writable(cfg)
                        cfg.unlink(missing_ok=True)
                    except OSError:
                        pass
    except Exception as exc:
        session.rollback();task=session.get(m.DeletionTask,task.id);task.status='database_failed';task.error_message=f'数据库删除异常: {type(exc).__name__}: {exc}';session.commit();return task_public(task)
    return file_outbox(session,settings,task)


def restore_task(session,settings,task):
    if task.mode!='trash' or task.status not in ('completed','files_failed'):raise AppError(409,'restore_unavailable','仅回收站任务可恢复，已清空或已恢复的任务不能重复恢复')
    moved=[]
    try:
        for relative,outcome in task.result_json.get('files',{}).items():
            if outcome['status']!='trashed':continue
            src=safe_file(settings.media_root,'.recycle/'+task.id+'/'+relative);dest=safe_file(settings.media_root,relative)
            if dest.exists():raise AppError(409,'restore_conflict','原位置已有文件，未覆盖')
            if not src.exists():raise AppError(409,'recycle_missing','回收站文件缺失，不能完整恢复')
            dest.parent.mkdir(parents=True,exist_ok=True);src.replace(dest);moved.append((dest,src))
        for item in _ordered_records(task.plan_json['records']):
            cls=MODELS[item['table']]
            if session.get(cls,item['id']):raise AppError(409,'record_conflict','记录已存在，不能覆盖恢复')
            values=dict(item['values'])
            if item['table']=='ai_services':values.update(enabled=False,is_default=False)
            for c in cls.__table__.columns:
                if isinstance(c.type,DateTime) and values.get(c.name):values[c.name]=datetime.fromisoformat(values[c.name])
            session.add(cls(**values));session.flush()
        task.status='restored';task.error_message=None
        for f in task.plan_json['files']:
            registry=session.scalar(select(m.ManagedFile).where(m.ManagedFile.relative_path==f['path']))
            if registry:registry.state='active'
        session.commit()
    except Exception:
        session.rollback()
        for dest,src in reversed(moved):
            if dest.exists() and not src.exists():dest.replace(src)
        raise
    return task_public(task)
