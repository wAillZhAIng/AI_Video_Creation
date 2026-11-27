import copy
import json
import struct
from pathlib import Path
from types import SimpleNamespace

import pytest

from app.core.errors import AppError
from app.services import custom_loras as svc

WORKFLOWS = Path(__file__).resolve().parents[2] / 'workflows'


def _safetensors(path: Path, tensors: dict[str, tuple[str, list[int], bytes | None]]):
    header, blobs, offset = {}, [], 0
    for name, (dtype, shape, data) in tensors.items():
        size = 1
        for dim in shape:
            size *= dim
        data = data if data is not None else b'\0' * (size * (1 if dtype in {'I8', 'U8'} else 4))
        header[name] = {'dtype': dtype, 'shape': shape, 'data_offsets': [offset, offset + len(data)]}
        blobs.append(data)
        offset += len(data)
    raw = json.dumps(header).encode()
    path.write_bytes(struct.pack('<Q', len(raw)) + raw + b''.join(blobs))
    return path


@pytest.fixture()
def lora_env(tmp_path, monkeypatch):
    loras = tmp_path / 'models' / 'loras'
    bases = tmp_path / 'models' / 'diffusion_models'
    loras.mkdir(parents=True)
    bases.mkdir(parents=True)
    monkeypatch.setenv('COMFYUI_LORA_DIR', str(loras))
    monkeypatch.delenv('COMFYUI_DIFFUSION_MODEL_DIR', raising=False)
    return loras, bases


def _h3_lora(path):
    return _safetensors(path, {
        'diffusion_model.blocks.0.attn.qkv_proj.lora_A.weight': ('F32', [4, 64], None),
        'diffusion_model.blocks.0.attn.qkv_proj.lora_B.weight': ('F32', [96, 4], None),
    })


def test_compatibility_accepts_matching_and_packed_w6_bases(lora_env):
    loras, bases = lora_env
    lora = _h3_lora(loras / 'style.safetensors')
    plain = _safetensors(bases / 'plain.safetensors', {'blocks.0.attn.qkv_proj.weight': ('F32', [96, 64], None)})
    quant = json.dumps({'format': 'w6a8_int8', 'group_size': 32}).encode()
    packed = _safetensors(bases / 'packed.safetensors', {
        'blocks.0.attn.qkv_proj.weight': ('I8', [96, 48], None),
        'blocks.0.attn.qkv_proj.weight_s_rel': ('I8', [96, 2], None),
        'blocks.0.attn.qkv_proj.comfy_quant': ('U8', [len(quant)], quant),
    })
    wrong = _safetensors(bases / 'other.safetensors', {'blocks.0.attn.qkv_proj.weight': ('F32', [96, 32], None)})
    alien = _safetensors(bases / 'alien.safetensors', {'double_blocks.0.img_attn.qkv.weight': ('F32', [96, 64], None)})
    assert svc.check_compatibility(lora, plain)['status'] == 'compatible'
    assert svc.check_compatibility(lora, packed)['status'] == 'compatible'
    assert svc.check_compatibility(lora, wrong)['status'] == 'incompatible'
    assert svc.check_compatibility(lora, alien)['status'] == 'incompatible'


def test_scan_lists_new_files_and_hides_workflow_system_loras(lora_env):
    loras, _ = lora_env
    _h3_lora(loras / 'new-style.safetensors')
    _h3_lora(loras / 'SMD-MiniMax-H3-Ref2VA-4step.safetensors')
    result = svc.scan_loras('missing-base.safetensors', comfy_names=['new-style.safetensors'])
    assert [item['name'] for item in result['items']] == ['new-style.safetensors']
    assert result['items'][0]['available_in_comfyui'] is True
    assert result['items'][0]['compatibility']['status'] == 'unknown'


def _workflow(name):
    return json.loads((WORKFLOWS / name).read_text(encoding='utf-8'))['prompt']


def test_injection_chains_after_acceleration_lora_and_is_noop_when_off():
    prompt = _workflow('h3_ref2va_native_fast4_v2.json')
    untouched = copy.deepcopy(prompt)
    assert svc.inject_custom_loras(prompt, []) == [] and prompt == untouched
    receipt = svc.inject_custom_loras(prompt, [{'name': 'a.safetensors', 'strength': 0.7},
                                               {'name': 'b.safetensors', 'strength': 0.4}])
    assert [item['node_id'] for item in receipt] == ['factory_custom_lora_0', 'factory_custom_lora_1']
    assert prompt['factory_custom_lora_0']['inputs']['model'] == ['140', 0]
    assert prompt['factory_custom_lora_1']['inputs']['model'] == ['factory_custom_lora_0', 0]
    assert prompt['141']['inputs']['model'] == ['factory_custom_lora_1', 0]
    assert prompt['140']['inputs']['lora_name'] == 'SMD-MiniMax-H3-Ref2VA-4step.safetensors'


def test_injection_rewires_every_model_consumer_without_acceleration_lora():
    prompt = _workflow('h3_ref2va_native_quality20_v2.json')
    svc.inject_custom_loras(prompt, [{'name': 'a.safetensors', 'strength': 1.0}])
    consumers = [key for key, node in prompt.items() if node['inputs'].get('model') == ['factory_custom_lora_0', 0]]
    assert consumers and not [key for key, node in prompt.items()
                              if node['inputs'].get('model') == ['127', 0] and key != 'factory_custom_lora_0']


def test_project_default_shot_override_and_revision_guard(tmp_path):
    settings = SimpleNamespace(media_root=tmp_path / 'media')
    saved = svc.save_project_config(settings, 'prj_1', {
        'expected_revision': 0,
        'project_default': {'enabled': True, 'items': [{'name': 'style.safetensors', 'strength': 0.7}]},
        'shot_overrides': {'shot_2': {'enabled': False, 'items': []},
                           'shot_3': {'enabled': True, 'items': [{'name': 'style.safetensors', 'strength': 1.2}]}},
    }, ['shot_1', 'shot_2', 'shot_3'])
    assert saved['revision'] == 1
    assert svc.resolve_for_shot(settings, 'prj_1', 'shot_1') == [{'name': 'style.safetensors', 'strength': 0.7}]
    assert svc.resolve_for_shot(settings, 'prj_1', 'shot_2') == []
    assert svc.resolve_for_shot(settings, 'prj_1', 'shot_3') == [{'name': 'style.safetensors', 'strength': 1.2}]
    with pytest.raises(AppError) as conflict:
        svc.save_project_config(settings, 'prj_1', {'expected_revision': 0}, ['shot_1'])
    assert conflict.value.code == 'custom_lora_config_conflict'
    with pytest.raises(AppError):
        svc.save_project_config(settings, 'prj_1', {'expected_revision': 1, 'project_default': {
            'enabled': True, 'items': [{'name': '../escape.safetensors', 'strength': 0.5}]}}, ['shot_1'])


def test_generation_validation_rejects_missing_and_incompatible(lora_env):
    loras, bases = lora_env
    _safetensors(bases / 'base.safetensors', {'blocks.0.attn.qkv_proj.weight': ('F32', [96, 32], None)})
    _h3_lora(loras / 'style.safetensors')
    document = {'prompt': {'1': {'class_type': 'UNETLoader', 'inputs': {'unet_name': 'base.safetensors'}}}}
    with pytest.raises(AppError) as missing:
        svc.validate_for_generation([{'name': 'nope.safetensors', 'strength': 0.7}], document)
    assert missing.value.code == 'custom_lora_missing'
    with pytest.raises(AppError) as incompatible:
        svc.validate_for_generation([{'name': 'style.safetensors', 'strength': 0.7}], document)
    assert incompatible.value.code == 'custom_lora_incompatible'
