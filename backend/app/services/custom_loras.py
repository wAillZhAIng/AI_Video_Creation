"""Optional user LoRAs for the native MiniMax H3 Ref2VA graph.

* ``scan_loras`` lists every ``.safetensors`` in the ComfyUI LoRA folder (new
  files are picked up without code changes) and checks base-model
  compatibility from the safetensors headers only: the LoRA's target modules
  must exist in the H3 diffusion model and every ``up @ down`` product must
  match the logical weight shape (packed w6a8/int8 bases are supported).
* Project defaults and per-shot overrides are stored in one small JSON
  document per project.  A shot override of ``None`` inherits the project,
  ``{"enabled": false}`` switches LoRAs off for that shot only.
* ``inject_custom_loras`` chains ``LoraLoaderModelOnly`` nodes after the
  workflow's own acceleration LoRA (or directly after ``UNETLoader``) at
  submit time; with no LoRA selected the frozen graph is untouched.
"""
from __future__ import annotations

import copy
import json
import os
import re
import struct
import threading
from pathlib import Path
from typing import Any

from app.core.config import PROJECT_ROOT
from app.core.errors import AppError

MAX_LORAS = 4
MIN_STRENGTH = 0.05
MAX_STRENGTH = 2.0
DEFAULT_STRENGTH = 0.7
NODE_PREFIX = 'factory_custom_lora_'
PARAMETER_KEY = 'custom_loras'

_WORKFLOW_ROOT = Path(__file__).resolve().parents[3] / 'workflows'
_NAME_RE = re.compile(r'^[\w .()\[\]+-]{1,200}\.safetensors$')
_lock = threading.Lock()
_compat_cache: dict[tuple, dict[str, Any]] = {}
_header_cache: dict[tuple, dict[str, Any]] = {}


# --------------------------------------------------------------------------- paths
def lora_dir() -> Path:
    configured = os.getenv('COMFYUI_LORA_DIR', '').strip()
    if configured:
        return Path(configured)
    for candidate in (Path('/workspace/ComfyUI/models/loras'), PROJECT_ROOT / '.vast' / 'ComfyUI' / 'models' / 'loras'):
        if candidate.is_dir():
            return candidate
    return Path('/workspace/ComfyUI/models/loras')


def diffusion_model_dirs() -> list[Path]:
    configured = os.getenv('COMFYUI_DIFFUSION_MODEL_DIR', '').strip()
    if configured:
        return [Path(configured)]
    models = lora_dir().parent
    return [models / 'diffusion_models', models / 'unet']


def _config_root(settings) -> Path:
    return Path(settings.media_root).parent / 'custom_lora_configs'


def system_lora_names() -> set[str]:
    """LoRAs pinned inside shipped workflows (e.g. 4-step distillation)."""
    names: set[str] = set()
    for path in _WORKFLOW_ROOT.glob('*.json'):
        try:
            text = path.read_text(encoding='utf-8')
        except OSError:
            continue
        names.update(re.findall(r'"lora_name"\s*:\s*"([^"]+)"', text))
    return names


def valid_name(name: Any) -> bool:
    return isinstance(name, str) and bool(_NAME_RE.match(name)) and '/' not in name and '\\' not in name and '..' not in name


# --------------------------------------------------------------------------- compatibility
def _header(path: Path) -> dict[str, Any]:
    stat = path.stat()
    key = (str(path.resolve()), stat.st_size, stat.st_mtime_ns)
    with _lock:
        if key in _header_cache:
            return _header_cache[key]
    with path.open('rb') as handle:
        size = struct.unpack('<Q', handle.read(8))[0]
        if size <= 0 or size > 64 * 1024 * 1024:
            raise ValueError('invalid safetensors header')
        header = json.loads(handle.read(size))
        quant: dict[str, dict] = {}
        base = 8 + size
        for name, meta in header.items():
            if name.endswith('.comfy_quant') and isinstance(meta, dict):
                start, end = meta['data_offsets']
                handle.seek(base + start)
                try:
                    quant[name[:-len('.comfy_quant')]] = json.loads(handle.read(end - start))
                except ValueError:
                    quant[name[:-len('.comfy_quant')]] = {}
    header.pop('__metadata__', None)
    result = {'tensors': header, 'quant': quant}
    with _lock:
        _header_cache[key] = result
    return result


def _strip_prefix(key: str) -> str:
    for prefix in ('model.diffusion_model.', 'diffusion_model.', 'transformer.'):
        if key.startswith(prefix):
            return key[len(prefix):]
    return key


_LORA_SUFFIXES = (('.lora_A.weight', 'down'), ('.lora_B.weight', 'up'), ('.lora_down.weight', 'down'),
                  ('.lora_up.weight', 'up'), ('.lora.down.weight', 'down'), ('.lora.up.weight', 'up'))


def _lora_modules(tensors: dict[str, Any]) -> tuple[dict[str, dict[str, list[int]]], list[str]]:
    modules: dict[str, dict[str, list[int]]] = {}
    unknown: list[str] = []
    for key, meta in tensors.items():
        for suffix, role in _LORA_SUFFIXES:
            if key.endswith(suffix):
                modules.setdefault(_strip_prefix(key[:-len(suffix)]), {})[role] = list(meta.get('shape') or [])
                break
        else:
            if not key.endswith('.alpha') and not key.endswith('.dora_scale'):
                unknown.append(key)
    return modules, unknown


def _logical_shape(base: dict[str, Any], module: str) -> list[int] | None:
    meta = base['tensors'].get(module + '.weight')
    if not isinstance(meta, dict):
        return None
    shape = list(meta.get('shape') or [])
    if len(shape) != 2:
        return shape or None
    quant = base['quant'].get(module)
    if quant:
        scale = base['tensors'].get(module + '.weight_s_rel')
        group = quant.get('group_size')
        if isinstance(scale, dict) and isinstance(group, int) and len(scale.get('shape') or []) == 2:
            return [shape[0], scale['shape'][1] * group]
        match = re.match(r'w(\d)', str(quant.get('format', '')))
        if match and int(match.group(1)) < 8:
            return [shape[0], shape[1] * 8 // int(match.group(1))]
    return shape


def check_compatibility(lora_path: Path, base_path: Path) -> dict[str, Any]:
    base_name = base_path.name
    try:
        ls, bs = lora_path.stat(), base_path.stat()
    except OSError:
        return {'status': 'unknown', 'base_model': base_name, 'reason': '基座模型或 LoRA 文件不存在'}
    key = (str(lora_path), ls.st_size, ls.st_mtime_ns, str(base_path), bs.st_size, bs.st_mtime_ns)
    with _lock:
        if key in _compat_cache:
            return copy.deepcopy(_compat_cache[key])
    try:
        modules, unknown = _lora_modules(_header(lora_path)['tensors'])
        base = _header(base_path)
    except (OSError, ValueError, KeyError, struct.error) as exc:
        return {'status': 'unknown', 'base_model': base_name, 'reason': f'无法读取 safetensors 头: {type(exc).__name__}'}
    missing, mismatched, matched = [], [], 0
    for module, parts in modules.items():
        logical = _logical_shape(base, module)
        if logical is None:
            missing.append(module)
            continue
        up, down = parts.get('up'), parts.get('down')
        if not up or not down or len(up) < 2 or len(down) < 2 or len(logical) < 2 or up[0] != logical[0] or down[1] != logical[1]:
            mismatched.append(module)
            continue
        matched += 1
    total = len(modules)
    if total == 0:
        status, reason = 'incompatible', '文件中没有可识别的 LoRA 权重（lora_A/B 或 lora_up/down）'
    elif matched == total:
        status, reason = 'compatible', f'{matched}/{total} 个目标层全部命中 {base_name}，形状一致'
    elif matched / total >= 0.9:
        status, reason = 'partial', f'{matched}/{total} 个目标层命中；{len(missing)} 层缺失、{len(mismatched)} 层形状不符将被忽略'
    else:
        status, reason = 'incompatible', f'仅 {matched}/{total} 个目标层命中 {base_name}；该 LoRA 不是为 MiniMax H3 训练的'
    result = {'status': status, 'base_model': base_name, 'matched_modules': matched, 'total_modules': total,
              'missing_modules': missing[:10], 'shape_mismatches': mismatched[:10],
              'unrecognized_tensors': len(unknown), 'reason': reason}
    with _lock:
        _compat_cache[key] = copy.deepcopy(result)
    return result


def workflow_base_model(document: dict[str, Any] | None = None) -> str | None:
    if document is None:
        try:
            document = json.loads((_WORKFLOW_ROOT / 'h3_ref2va_native_fast4_v2.json').read_text(encoding='utf-8'))
        except (OSError, ValueError):
            return None
    prompt = document.get('prompt') if isinstance(document, dict) else None
    names = [node.get('inputs', {}).get('unet_name') for node in (prompt or {}).values()
             if isinstance(node, dict) and node.get('class_type') == 'UNETLoader']
    return names[0] if len(names) == 1 and isinstance(names[0], str) else None


def _base_path(name: str | None) -> Path | None:
    if not name:
        return None
    for folder in diffusion_model_dirs():
        candidate = folder / name
        if candidate.is_file():
            return candidate
    return None


def scan_loras(base_model: str | None = None, comfy_names: list[str] | None = None) -> dict[str, Any]:
    folder = lora_dir()
    base_model = base_model or workflow_base_model()
    base = _base_path(base_model)
    system = system_lora_names()
    items = []
    files = sorted(folder.glob('*.safetensors')) if folder.is_dir() else []
    for path in files:
        if path.name in system or not path.is_file():
            continue
        stat = path.stat()
        compat = (check_compatibility(path.resolve(), base) if base else
                  {'status': 'unknown', 'base_model': base_model, 'reason': '未找到当前工作流的 H3 基座权重文件，无法校验'})
        items.append({'name': path.name, 'size_bytes': stat.st_size, 'modified_at': stat.st_mtime,
                      'available_in_comfyui': None if comfy_names is None else path.name in comfy_names,
                      'compatibility': compat})
    return {'lora_dir': str(folder), 'base_model': base_model, 'system_loras': sorted(system), 'items': items,
            'limits': limits()}


def limits() -> dict[str, Any]:
    return {'max_loras': MAX_LORAS, 'min_strength': MIN_STRENGTH, 'max_strength': MAX_STRENGTH,
            'default_strength': DEFAULT_STRENGTH}


# --------------------------------------------------------------------------- selection normalisation
def normalize_selection(value: Any, *, field: str = 'custom_loras') -> dict[str, Any]:
    if not isinstance(value, dict):
        raise AppError(422, 'custom_lora_invalid', f'{field} 必须是对象')
    enabled = bool(value.get('enabled', False))
    raw = value.get('items') or []
    if not isinstance(raw, list) or len(raw) > MAX_LORAS:
        raise AppError(422, 'custom_lora_limit', f'每个层级最多 {MAX_LORAS} 个自定义 LoRA')
    items, seen = [], set()
    for entry in raw:
        if not isinstance(entry, dict) or not valid_name(entry.get('name')):
            raise AppError(422, 'custom_lora_invalid', 'LoRA 文件名无效')
        if entry['name'] in seen:
            raise AppError(422, 'custom_lora_duplicate', '同一层级不能重复选择同一个 LoRA')
        seen.add(entry['name'])
        try:
            strength = float(entry.get('strength', DEFAULT_STRENGTH))
        except (TypeError, ValueError) as exc:
            raise AppError(422, 'custom_lora_strength', 'LoRA 强度无效') from exc
        if not MIN_STRENGTH <= strength <= MAX_STRENGTH:
            raise AppError(422, 'custom_lora_strength', f'LoRA 强度需在 {MIN_STRENGTH}~{MAX_STRENGTH} 之间')
        items.append({'name': entry['name'], 'strength': round(strength, 3)})
    return {'enabled': enabled, 'items': items}


def effective_items(selection: dict[str, Any] | None) -> list[dict[str, Any]]:
    if not selection or not selection.get('enabled'):
        return []
    return [{'name': item['name'], 'strength': item['strength']} for item in selection.get('items') or []]


# --------------------------------------------------------------------------- project storage
def _config_path(settings, project_id: str) -> Path:
    if not re.fullmatch(r'[\w-]{1,64}', project_id or ''):
        raise AppError(422, 'project_id_invalid', '项目 ID 无效')
    return _config_root(settings) / f'{project_id}.json'


def _read(settings, project_id: str) -> dict[str, Any]:
    path = _config_path(settings, project_id)
    try:
        data = json.loads(path.read_text(encoding='utf-8'))
    except FileNotFoundError:
        data = {}
    except (OSError, ValueError) as exc:
        raise AppError(503, 'custom_lora_config_unreadable', '项目 LoRA 设置文件无法读取') from exc
    return {'revision': int(data.get('revision', 0)),
            'project_default': data.get('project_default') or {'enabled': False, 'items': []},
            'shot_overrides': data.get('shot_overrides') or {}}


def project_config(settings, project_id: str, shot_ids: list[str] | None = None) -> dict[str, Any]:
    data = _read(settings, project_id)
    overrides = data['shot_overrides']
    if shot_ids is not None:
        overrides = {key: value for key, value in overrides.items() if key in set(shot_ids)}
    return {'project_id': project_id, 'revision': data['revision'], 'project_default': data['project_default'],
            'shot_overrides': overrides, 'limits': limits()}


def save_project_config(settings, project_id: str, payload: dict[str, Any], shot_ids: list[str]) -> dict[str, Any]:
    with _lock:
        current = _read(settings, project_id)
        if int(payload.get('expected_revision', -1)) != current['revision']:
            raise AppError(409, 'custom_lora_config_conflict', 'LoRA 设置已在别处更新，请刷新后再保存')
        default = normalize_selection(payload.get('project_default') or {'enabled': False, 'items': []}, field='project_default')
        raw_overrides = payload.get('shot_overrides') or {}
        if not isinstance(raw_overrides, dict):
            raise AppError(422, 'custom_lora_invalid', 'shot_overrides 必须是对象')
        invalid = sorted(set(raw_overrides) - set(shot_ids))
        if invalid:
            raise AppError(422, 'custom_lora_shot_scope', 'LoRA 覆盖镜头必须属于当前项目', {'shot_ids': invalid})
        overrides = {shot_id: normalize_selection(value, field=f'shot {shot_id}') for shot_id, value in raw_overrides.items()}
        document = {'revision': current['revision'] + 1, 'project_default': default, 'shot_overrides': overrides}
        path = _config_path(settings, project_id)
        path.parent.mkdir(parents=True, exist_ok=True)
        temp = path.with_suffix('.tmp')
        temp.write_text(json.dumps(document, ensure_ascii=False, indent=2), encoding='utf-8')
        os.replace(temp, path)
    return project_config(settings, project_id, shot_ids)


def resolve_for_shot(settings, project_id: str, shot_id: str) -> list[dict[str, Any]]:
    data = _read(settings, project_id)
    override = data['shot_overrides'].get(shot_id)
    return effective_items(override if override is not None else data['project_default'])


def validate_for_generation(items: list[dict[str, Any]], workflow_document: dict[str, Any] | None) -> list[dict[str, Any]]:
    """Fail fast before a GPU job is queued with a missing/incompatible LoRA."""
    if not items:
        return []
    base_name = workflow_base_model(workflow_document)
    base = _base_path(base_name)
    system = system_lora_names()
    frozen = []
    for item in items:
        name = item.get('name')
        if not valid_name(name) or name in system:
            raise AppError(422, 'custom_lora_invalid', f'LoRA 无效或为工作流内置 LoRA: {name}')
        path = lora_dir() / name
        if not path.is_file():
            raise AppError(409, 'custom_lora_missing', f'LoRA 文件不存在: {name}；请重新扫描后再选择')
        compat = check_compatibility(path.resolve(), base) if base else {'status': 'unknown', 'reason': '基座权重未找到'}
        if compat['status'] == 'incompatible':
            raise AppError(409, 'custom_lora_incompatible', f'LoRA 与当前 H3 基座不兼容: {name}', {'compatibility': compat})
        frozen.append({'name': name, 'strength': float(item['strength']), 'compatibility_status': compat['status'],
                       'base_model': base_name})
    return frozen


# --------------------------------------------------------------------------- graph injection
def inject_custom_loras(prompt: dict[str, Any], items: list[dict[str, Any]] | None) -> list[dict[str, Any]]:
    if not items:
        return []
    if len(items) > MAX_LORAS:
        raise AppError(422, 'custom_lora_limit', f'最多 {MAX_LORAS} 个自定义 LoRA')
    loaders = [key for key, node in prompt.items() if node.get('class_type') == 'UNETLoader']
    if len(loaders) != 1:
        raise AppError(422, 'custom_lora_target_ambiguous', '工作流必须只有一个 UNETLoader 才能注入自定义 LoRA')
    tail = loaders[0]
    while True:
        following = [key for key, node in prompt.items() if node.get('class_type') == 'LoraLoaderModelOnly'
                     and list(node.get('inputs', {}).get('model') or []) == [tail, 0]]
        if len(following) != 1:
            break
        tail = following[0]
    consumers = [(key, field) for key, node in prompt.items() for field, value in node.get('inputs', {}).items()
                 if isinstance(value, list) and list(value) == [tail, 0]]
    if not consumers:
        raise AppError(422, 'custom_lora_target_missing', '工作流中没有消费模型输出的节点，无法注入 LoRA')
    previous, receipt = tail, []
    for index, item in enumerate(items):
        node_id = f'{NODE_PREFIX}{index}'
        if node_id in prompt:
            raise AppError(422, 'custom_lora_node_collision', '自定义 LoRA 节点 ID 冲突')
        prompt[node_id] = {'class_type': 'LoraLoaderModelOnly',
                           'inputs': {'model': [previous, 0], 'lora_name': item['name'],
                                      'strength_model': float(item['strength'])}}
        receipt.append({'node_id': node_id, 'lora_name': item['name'], 'strength_model': float(item['strength']),
                        'after_node': previous})
        previous = node_id
    for key, field in consumers:
        prompt[key]['inputs'][field] = [previous, 0]
    return receipt
