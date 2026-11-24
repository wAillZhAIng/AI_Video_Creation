"""Unit coverage for the native per-image analysis adapter.

No model server, GPU, video output or assembly flow is used here.  The test
only replaces the audited compiler's analysis callback with an in-memory
fixture, then checks that the adapter preserves each original reference for
the existing compiler.
"""
from dataclasses import dataclass
import importlib
import importlib.util
from pathlib import Path
import sys
from types import ModuleType, SimpleNamespace

import pytest


ROOT = Path(__file__).resolve().parents[1]
NODE = ROOT / 'scripts' / 'local_inference' / 'bundle' / 'comfy_nodes' / 'VideoFactoryLocalVLMGuard'


def _load_profile(monkeypatch):
    package = ModuleType('_multiref_native_hook')
    package.__path__ = [str(NODE)]
    monkeypatch.setitem(sys.modules, package.__name__, package)
    spec = importlib.util.spec_from_file_location(package.__name__ + '.local_h3ir_backend_profile',
                                                  NODE / 'local_h3ir_backend_profile.py')
    module = importlib.util.module_from_spec(spec)
    assert spec.loader is not None
    spec.loader.exec_module(module)
    return module, package.__name__


def test_native_capture_delegates_to_per_image_analysis_and_keeps_all_refs(tmp_path, monkeypatch):
    compiler = pytest.importorskip('h3ir.compile')
    from h3ir.models import AssetKind
    profile, package = _load_profile(monkeypatch)
    analysis = importlib.import_module(package + '.reference_analysis')
    validation = importlib.import_module(package + '.product_reference_validation')

    @dataclass
    class Card:
        descriptor: str

    original = compiler.analyse_all
    observed = {}

    def legacy(backend, refs, *, use_cache=True, seed=None):
        return {'legacy': Card('legacy')}

    def per_image(original_callback, bound, scope):
        observed['refs'] = list(bound.arguments['refs'])
        observed['seed'] = scope.get('analysis_seed')
        observed['original'] = original_callback
        return {'a' * 64: Card('front'), 'b' * 64: Card('side'), 'c' * 64: Card('detail')}

    def verified(backend, refs, cards, contract, cache_root, seed, *, receipt_root):
        observed['verification_refs'] = list(refs)
        observed['verification_seed'] = seed
        return {'status': 'PASS', 'protocol': 'fixture', 'approved_product_descriptors_by_native': {}}

    monkeypatch.setattr(compiler, 'analyse_all', legacy)
    monkeypatch.setattr(analysis, 'analyse_references', per_image)
    monkeypatch.setattr(validation, 'verify_product_references', verified)
    profile._install_analysis_capture()

    refs = [SimpleNamespace(id='one', kind=AssetKind.IMAGE), SimpleNamespace(id='two', kind=AssetKind.IMAGE),
            SimpleNamespace(id='three', kind=AssetKind.IMAGE)]
    scope = {
        'directory': tmp_path,
        'seed': 99,
        'reference_contract': {
            'trace_id': 'trace-three',
            'identity_validation_required': True,
            'references': [
                {'image_id': 'one', 'sha256': 'a' * 64},
                {'image_id': 'two', 'sha256': 'b' * 64},
                {'image_id': 'three', 'sha256': 'c' * 64},
            ],
        },
    }
    token = profile._ACTIVE.set(scope)
    try:
        actual = compiler.analyse_all(SimpleNamespace(), refs, use_cache=True, seed=123)
    finally:
        profile._ACTIVE.reset(token)

    assert observed['refs'] == refs
    assert observed['verification_refs'] == refs
    assert observed['seed'] == analysis.ANALYSIS_SEED
    assert observed['verification_seed'] == analysis.ANALYSIS_SEED
    assert set(actual) == {'a' * 64, 'b' * 64, 'c' * 64}
    assert 'analysis_seed' not in scope
    assert scope['product_identity_verification']['status'] == 'PASS'
    assert (tmp_path / 'analysis-cards.json').is_file()
    assert 'product_identity_verification' in (tmp_path / 'reference-trace.jsonl').read_text(encoding='utf-8')
    monkeypatch.setattr(compiler, 'analyse_all', original)


def test_native_capture_preserves_existing_single_and_double_image_parser(tmp_path, monkeypatch):
    compiler = pytest.importorskip('h3ir.compile')
    from h3ir.models import AssetKind
    profile, package = _load_profile(monkeypatch)
    analysis = importlib.import_module(package + '.reference_analysis')
    validation = importlib.import_module(package + '.product_reference_validation')

    @dataclass
    class Card:
        descriptor: str

    observed = {}
    original = compiler.analyse_all

    def legacy(backend, refs, *, use_cache=True, seed=None):
        observed['legacy_refs'] = list(refs)
        observed['legacy_seed'] = seed
        return {'a' * 64: Card('existing native card')}

    def unexpected_per_image(*args, **kwargs):
        raise AssertionError('single/double image parser must remain native')

    def verified(backend, refs, cards, contract, cache_root, seed, *, receipt_root):
        observed['verification_seed'] = seed
        return {'status': 'PASS', 'protocol': 'fixture', 'approved_product_descriptors_by_native': {}}

    monkeypatch.setattr(compiler, 'analyse_all', legacy)
    monkeypatch.setattr(analysis, 'analyse_references', unexpected_per_image)
    monkeypatch.setattr(validation, 'verify_product_references', verified)
    profile._install_analysis_capture()
    refs = [SimpleNamespace(id='one', kind=AssetKind.IMAGE), SimpleNamespace(id='two', kind=AssetKind.IMAGE)]
    scope = {
        'directory': tmp_path,
        'seed': 99,
        'reference_contract': {'trace_id': 'trace-two', 'identity_validation_required': True,
                               'references': [{'image_id': 'one', 'sha256': 'a' * 64},
                                              {'image_id': 'two', 'sha256': 'b' * 64}]},
    }
    token = profile._ACTIVE.set(scope)
    try:
        actual = compiler.analyse_all(SimpleNamespace(), refs, use_cache=True, seed=123)
    finally:
        profile._ACTIVE.reset(token)

    assert observed['legacy_refs'] == refs
    assert observed['legacy_seed'] == 123
    assert observed['verification_seed'] == 99
    assert actual['a' * 64].descriptor == 'existing native card'
    assert 'analysis_seed' not in scope
    assert 'native_batch_analysis_preserved' in (tmp_path / 'reference-trace.jsonl').read_text(encoding='utf-8')
    monkeypatch.setattr(compiler, 'analyse_all', original)
