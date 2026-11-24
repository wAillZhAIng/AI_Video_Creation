"""Regression contract for the scoped multi-reference Context-IR adapter.

These tests deliberately stop before H3 execution.  They prove that every
selected picture keeps an independent record through preparation and the
existing native compiler contract; the existing review and assembly flows are
not part of this adapter.
"""
from dataclasses import dataclass, field
import importlib.util
import json
from pathlib import Path
import sys
from types import ModuleType, SimpleNamespace

import pytest

from app.services.context_references import REFERENCE_SCHEMA_VERSION, reference_records
from app.services.h3_context_provider import CALLER_CONTRACT_POLICY, compiler_contract_supported, prepare_context_compilation
from app.services.h3_workflow_router import build_graph, freeze_workflow


ROOT = Path(__file__).resolve().parents[2]
NODE = ROOT / 'scripts' / 'local_inference' / 'bundle' / 'comfy_nodes' / 'VideoFactoryLocalVLMGuard'


def _load_node_module(monkeypatch):
    package = ModuleType('_context_multiref_node')
    package.__path__ = [str(NODE)]
    monkeypatch.setitem(sys.modules, package.__name__, package)
    spec = importlib.util.spec_from_file_location(package.__name__ + '.compile_node', NODE / 'compile_node.py')
    module = importlib.util.module_from_spec(spec)
    assert spec.loader is not None
    spec.loader.exec_module(module)
    return module


@dataclass
class _Slot:
    kind: str
    label: str


class _Tray:
    def asset_order(self, slots):
        return [(slot, 'file') for slot in slots]


def _image(index):
    return {
        'asset_id': f'image-{index}',
        'sha256': f'{index:064x}',
        'relative_path': f'products/view-{index}.jpg',
        'product_id': 'same-product' if index < 3 else f'product-{index}',
        'role': 'product',
        'product_name': 'test product',
        'image_notes': f'view note {index}',
    }


@pytest.mark.parametrize('count', [0, 1, 2, 3, 4, 6, 8])
def test_dynamic_image_records_survive_existing_compiler_contract(count, monkeypatch):
    images = [_image(index) for index in range(count)]
    tray, declarations, _, policy = prepare_context_compilation(
        'Show every declared product reference in a neutral studio.', '', images, [],
        {'duration_seconds': 5.167}, silent=True)

    assert len(tray) == count
    assert 'reference_schema_version' not in policy
    assert 'references' not in policy
    provenance = json.loads(declarations.get('reference_provenance') or json.dumps({
        'reference_schema_version': REFERENCE_SCHEMA_VERSION, 'references': []}))
    assert provenance['reference_schema_version'] == REFERENCE_SCHEMA_VERSION
    assert [item['sha256'] for item in provenance['references']] == [item['sha256'] for item in images]
    assert [item['image_description'] for item in provenance['references']] == [item['image_notes'] for item in images]
    assert [item['product_id'] for item in provenance['references']] == [item['product_id'] for item in images]
    assert [item['native_label'] for item in provenance['references']] == [f'<Picture {index}>' for index in range(1, count + 1)]

    module = _load_node_module(monkeypatch)
    slots = [_Slot('picture', f'picture{index}') for index in range(1, count + 1)]
    def execute():
        return None
    execute.__globals__['T'] = _Tray()
    result = module.reference_contract(
        SimpleNamespace(execute=execute), {'slots': [{'slot': slot} for slot in slots],
        }, declarations['reference_groups'], declarations['required_references'], declarations['motion_targets'],
        json.dumps(policy), declarations.get('reference_provenance', ''))
    assert [item['image_id'] for item in result.get('references', [])] == [item['asset_id'] for item in images]


def test_reference_schema_accepts_zero_through_eight_records():
    jsonschema = pytest.importorskip('jsonschema')
    schema = json.loads((ROOT / 'schemas' / 'context-reference-set-v1.json').read_text(encoding='utf-8'))
    for count in [0, 1, 2, 3, 4, 6, 8]:
        records = reference_records([_image(index) for index in range(count)])
        jsonschema.validate({'reference_schema_version': REFERENCE_SCHEMA_VERSION, 'references': records}, schema)


def test_provenance_is_a_new_optional_compiler_input_not_a_production_snapshot_mutation():
    images = [_image(index) for index in range(3)]
    tray, declarations, _, policy = prepare_context_compilation(
        'Show the three declared product references.', '', images, [], {'duration_seconds': 5}, silent=True)
    uploaded = [{**slot, 'file': f'jobs/test/picture-{index}.png [input]'}
                for index, slot in enumerate(tray, 1)]
    graph = build_graph(
        freeze_workflow('fast'), tray=uploaded, intent='One short daylight shot.', duration_seconds=5,
        aspect_ratio='16:9', seed=20261104, width=672, height=384,
        output_prefix='projects/project-1/shot-2/job-3', production_contract=policy, **declarations)
    assert 'references' not in policy and 'reference_schema_version' not in policy
    assert json.loads(graph['4']['inputs']['production_contract']) == policy
    assert json.loads(graph['4']['inputs']['reference_provenance'])['references'][2]['native_label'] == '<Picture 3>'


def test_same_hash_occurrences_never_overwrite_identity_records():
    records = reference_records([
        {'asset_id': 'angle-a', 'sha256': 'a' * 64, 'relative_path': 'cup/front.jpg'},
        {'asset_id': 'angle-a', 'sha256': 'a' * 64, 'relative_path': 'cup/side.jpg'},
    ])
    assert len({item['image_id'] for item in records}) == 2
    assert [item['source_path'] for item in records] == ['cup/front.jpg', 'cup/side.jpg']
    assert all(item['product_id'] is None and item['subject_id'] is None for item in records)


def test_stale_compiler_is_blocked_only_when_an_image_needs_the_new_parser():
    base = {
        'input': {'optional': {
            'production_contract': ['STRING', {}],
            'caller_contract_policy': ['STRING', {'default': CALLER_CONTRACT_POLICY}],
        }}
    }
    assert compiler_contract_supported(base)
    assert not compiler_contract_supported(base, multi_reference=True)
    base['input']['optional']['reference_provenance'] = ['STRING', {}]
    assert compiler_contract_supported(base, multi_reference=True)


def test_fusion_requires_explicit_group_and_visual_proof():
    spec = importlib.util.spec_from_file_location('_context_multiref_fusion', NODE / 'reference_fusion.py')
    fusion = importlib.util.module_from_spec(spec)
    assert spec.loader is not None
    spec.loader.exec_module(fusion)

    @dataclass
    class Subject:
        label: str
        descriptor: str
        sources: list[str]
        attributes: list[str] = field(default_factory=list)

    subjects = [
        Subject('<Subject 1>', 'bottle front', ['<Picture 1>'], ['orange pattern']),
        Subject('<Subject 2>', 'hand', ['<Picture 2>'], ['bare fingers']),
        Subject('<Subject 3>', 'bottle side', ['<Picture 2>'], ['black lid']),
        Subject('<Subject 4>', 'green toy', ['<Picture 3>'], ['green fabric']),
    ]
    contract = {'groups': [{'category': 'product', 'group': 'cup-sku',
                            'appearance_native': ['<Picture 1>', '<Picture 2>']}]}
    proof = {'status': 'PASS', 'approved_product_descriptors_by_native': {
        '<Picture 1>': ['bottle front'], '<Picture 2>': ['bottle side']}}
    fused, receipt = fusion.fuse_product_subjects(subjects, contract, proof)
    assert len(fused) == 3
    assert fused[0].sources == ['<Picture 1>', '<Picture 2>']
    assert fused[0].attributes == ['orange pattern', 'black lid']
    assert fused[1].descriptor == 'hand'
    assert fused[2].descriptor == 'green toy'
    assert receipt['image_inputs_removed'] is False
    assert receipt['unmatched_subject_policy'].startswith('retained')
    untouched, skipped = fusion.fuse_product_subjects(subjects, contract, {'status': 'BLOCKED'})
    assert untouched is subjects and skipped['status'] == 'NOT_APPLIED'


def test_cache_version_tracks_actual_qwen_bundle_sources(tmp_path, monkeypatch):
    spec = importlib.util.spec_from_file_location('_context_multiref_analysis', NODE / 'reference_analysis.py')
    analysis = importlib.util.module_from_spec(spec)
    assert spec.loader is not None
    spec.loader.exec_module(analysis)
    installed = tmp_path / 'comfy' / 'custom_nodes' / 'Guard' / 'reference_analysis.py'
    monkeypatch.setattr(analysis, '__file__', str(installed))
    bundle = tmp_path / 'bundle'
    (bundle / 'qwen-metadata').mkdir(parents=True)
    for relative in ('serve_screenplay_qwen.py', 'serve_h3ir_qwen4b.py', 'qwen-metadata/factory-model-manifest.json'):
        (bundle / relative).write_text(relative, encoding='utf-8')
    runtime = tmp_path / 'runtime'
    runtime.mkdir()
    (runtime / 'qwen-owned-process.json').write_text(
        json.dumps({'entry': str(bundle / 'serve_h3ir_qwen4b.py'), 'pid': 1}), encoding='utf-8')
    monkeypatch.setenv('VF_RUNTIME_ROOT', str(runtime))
    first = analysis.model_source_components()
    (bundle / 'qwen-metadata' / 'factory-model-manifest.json').write_text('changed', encoding='utf-8')
    assert analysis.model_source_components() != first
