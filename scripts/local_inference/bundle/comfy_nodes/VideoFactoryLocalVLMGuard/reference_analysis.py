"""Bounded per-file analysis with complete source and cache provenance."""
from dataclasses import asdict
from copy import deepcopy
import hashlib
import json
import os
import inspect
from pathlib import Path
import time
import uuid


PROTOCOL = 'context-reference-analysis-v3'
ANALYSIS_SEED = 1247
CARD_PROMPT = (
    'Describe the principal foreground reference subjects in this photo. Printed cartoons are not physical subjects. '
    'Count each real foreground object once, an attached lid or handle is a component. '
    'Small background decorations belong to environment, not reusable subjects. '
    'Do not identify a brand from the product design. Describe color, shape, material, markings, structure, angle and pose. '
    'Text transcription is supplied separately below. Use ONLY those literal words for visible_text. '
    'Do not name brands or quote lettering in summary or attributes; keep literal text only in visible_text. '
    'Return only a concise JSON object with exactly: summary (one sentence), '
    'subjects (array of objects with kind: person/animal/object/environment/style, descriptor, '
    'attributes: array of observed visual features, pose: array of observed poses), '
    'environment (short), lighting (short), framing (short), style (short), '
    'visible_text (array), composition (bare_plate/composed_scene/unknown), is_reference_sheet (boolean). '
    'Use distinct concrete descriptors. When similar physical items are visibly separate, distinguish each by observed form or position; '
    'do not merge them or repeat an indistinguishable record. Never guess a missing detail.'
)
OCR_PROMPT = (
    'Literal OCR only. Transcribe the actual characters visible on this product, one line per visible word or phrase. '
    'Do not identify the brand from its product design. If a word is cropped, transcribe just the visible letters. '
    'If no characters are legible, return exactly [unclear] once and stop. Do not invent or complete any words.'
)


def validate_observation(obj):
    """Validate the complete native image-card shape, independently of decoding."""
    from h3ir.analyse import IMAGE_SCHEMA
    if not isinstance(obj, dict) or not set(IMAGE_SCHEMA['required']) <= set(obj) or set(obj) - set(IMAGE_SCHEMA['properties']):
        raise ValueError('Visual JSON does not match the native ImageCard fields')
    for field in ('summary', 'environment', 'lighting', 'framing', 'style'):
        if not isinstance(obj[field], str):
            raise ValueError('ImageCard string field invalid: ' + field)
    if type(obj['is_reference_sheet']) is not bool or obj['composition'] not in {'bare_plate','composed_scene','unknown'}:
        raise ValueError('ImageCard composition or reference-sheet flag invalid')
    if not isinstance(obj['subjects'], list) or not obj['subjects']:
        raise ValueError('ImageCard must contain at least one foreground subject')
    subjects_by_identity = {}
    for subject in obj['subjects']:
        if (not isinstance(subject, dict) or set(subject) != {'kind','descriptor','attributes','pose'}
                or subject['kind'] not in {'person','animal','object','environment','style'}
                or not isinstance(subject['descriptor'],str) or not subject['descriptor'].strip()):
            raise ValueError('ImageCard subject has invalid fields or kind')
        for key in ('attributes','pose'):
            if not isinstance(subject[key],list) or any(not isinstance(item,str) for item in subject[key]):
                raise ValueError('ImageCard subject observations must be string arrays')
        normalize = lambda value: ' '.join(value.casefold().split()).strip(' "\'.,:;')
        identity = (subject['kind'], normalize(subject['descriptor']))
        evidence = (tuple(normalize(item) for item in subject['attributes']),
                    tuple(normalize(item) for item in subject['pose']))
        prior = subjects_by_identity.setdefault(identity, [])
        if evidence in prior:
            raise ValueError('Visual model repeated an indistinguishable physical subject; no automatic deduplication')
        # Two separately visible copies may share a general noun (for example
        # two different straps in an accessory sheet).  Retain both only when
        # their model-observed visual evidence differs; Python never invents
        # an ordinal, merges attributes, or silently removes a subject.
        prior.append(evidence)
    for field in ('visible_text','palette'):
        if field in obj and (not isinstance(obj[field],list) or any(not isinstance(item,str) for item in obj[field])):
            raise ValueError('ImageCard field must be a string array: ' + field)


def visual_card(backend, ref, seed, inventory_path):
    from h3ir.analyse import ANALYZER_VERSION
    from h3ir.backend import user_message
    from h3ir.models import AssetCard, AssetKind

    ocr_recovery = None
    try:
        # No workflow-specific token ceiling: the local adapter derives the
        # generation budget from the remaining model context and VRAM.  The
        # stop string ends the known ``[unclear]`` loop without discarding a
        # real multi-line transcription.
        ocr = backend.chat([user_message(OCR_PROMPT,[ref.path])],
                           stop=['\n[unclear]'], temperature=0.2, thinking=False, seed=seed, retries=0)
        ocr_text = ocr.content
    except Exception as error:
        # This is deliberately narrow.  A literal OCR reply that reaches the
        # model boundary contains no verified lettering; recording it as
        # inconclusive is truthful.  Card analysis and every other compiler
        # failure remain loud and are never converted into a success.
        if type(error).__name__ != 'TruncatedResponse':
            raise
        ocr_text = '[unclear]'
        ocr_recovery = {'reason':'truncated_literal_ocr','error':str(error),
                        'action':'recorded no visible text; card analysis still runs'}
    write(inventory_path, {'stage':'literal_ocr','text':ocr_text,'model':backend.cfg.model,
                           'image_sha256':ref.sha256,'scope':'candidate literal transcription; no brand inference',
                           'recovery':ocr_recovery})
    prompt = CARD_PROMPT + '\nLiteral OCR candidate, do not invent other text:\n' + ocr_text
    if ref.note:
        prompt += '\nCaller reference purpose (not visual evidence): ' + ref.note
    reply = backend.chat([user_message(prompt,[ref.path])],
                         temperature=0.2, thinking=False, seed=seed, retries=0)
    write(inventory_path.with_name(inventory_path.stem+'-card-reply.json'),
          {'text':reply.content,'image_sha256':ref.sha256,'model':backend.cfg.model})
    raw = reply.content.strip()
    if raw.startswith('```json') and raw.endswith('```'):
        raw = raw[7:-3].strip()
    obj = json.loads(raw)
    validate_observation(obj)
    normalize = lambda value: ' '.join(value.casefold().split()).strip(' \"\'.,:;')
    transcription = normalize(ocr_text)
    if any(normalize(value) not in transcription for value in obj['visible_text'] if normalize(value)):
        raise ValueError('Visual JSON invented lettering absent from the separate literal OCR observation')
    sheet = obj['is_reference_sheet']
    return AssetCard(sha256=ref.sha256, kind=AssetKind.IMAGE, summary=obj['summary'], subjects=obj['subjects'],
        environment='' if sheet else obj['environment'], lighting=obj['lighting'], palette=obj.get('palette') or [],
        framing=obj['framing'], style=obj['style'], visible_text=[] if sheet else obj['visible_text'],
        composition=ref.composition if ref.composition != 'unknown' else obj['composition'],
        is_reference_sheet=sheet, analyzer_version=ANALYZER_VERSION, model_id=backend.cfg.model)


def digest(value):
    return hashlib.sha256(json.dumps(value, sort_keys=True, ensure_ascii=False, default=str).encode()).hexdigest()


def write(path, value):
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_name(path.name + '.' + uuid.uuid4().hex + '.tmp')
    try:
        temporary.write_text(json.dumps(value, ensure_ascii=False, indent=2, default=str), encoding='utf-8')
        temporary.replace(path)
    finally:
        temporary.unlink(missing_ok=True)


def trace(scope, stage, references, **details):
    contract = scope.get('reference_contract') or {}
    record = {'trace_id': contract.get('trace_id') or scope['directory'].name,
              'stage': stage, 'at_unix': time.time(), 'image_count': len(references),
              'references': references, **details}
    with (scope['directory'] / 'reference-trace.jsonl').open('a', encoding='utf-8') as stream:
        stream.write(json.dumps(record, ensure_ascii=False, default=str) + '\n')


def model_source_components():
    """Resolve source versions both in the source bundle and installed Comfy node.

    The supervised entry is metadata, never executed. Cache identity excludes
    process ids and timestamps so idle unload/reload retains valid analyses.
    """
    candidates=[Path(__file__).resolve().parent.parents[1]]
    managed_root=os.environ.get('VF_RUNTIME_ROOT')
    if managed_root:
        try:
            owner=json.loads((Path(managed_root)/'qwen-owned-process.json').read_text(encoding='utf-8'))
            entry=Path(owner['entry'])
            if entry.name=='serve_h3ir_qwen4b.py':candidates.append(entry.parent)
        except (OSError,ValueError,KeyError,TypeError):
            pass
    files=['serve_screenplay_qwen.py','serve_h3ir_qwen4b.py','qwen-metadata/factory-model-manifest.json']
    for directory in candidates:
        if all((directory/relative).is_file() for relative in files):
            return {relative:hashlib.sha256((directory/relative).read_bytes()).hexdigest() for relative in files}
    raise ValueError('Cannot verify local Qwen implementation and model revision for reference-analysis cache')


def analyse_references(original, bound, scope):
    from PIL import Image, ImageOps
    from h3ir.models import AssetCard, AssetKind
    from h3ir.analyse import IMAGE_SCHEMA, IMAGE_INSTRUCTIONS, ANALYZER_VERSION

    refs = bound.arguments['refs']
    contract = scope.get('reference_contract') or {}
    declared = contract.get('references') or []
    cards, records = {}, []
    source = model_source_components()
    picture_index = 0
    state_root = os.environ.get('H3IR_STATE_DIR')
    cache_dir = (Path(state_root) / 'factory-reference-analysis-v2' if state_root
                 else scope['directory'].parent / 'reference-analysis-cache')
    for ref in refs:
        if ref.kind is not AssetKind.IMAGE:
            call = dict(bound.arguments, refs=[ref])
            cards.update(original(**call))
            continue
        picture_index += 1
        native_label = f'<Picture {picture_index}>'
        candidates = [row for row in declared if row.get('native_label') == native_label and row.get('sha256') == ref.sha256]
        if declared and len(candidates) != 1:
            raise ValueError(f'{native_label} has no unique frozen source record matching its SHA256')
        record = dict(candidates[0]) if candidates else {
            'image_id': f'image-{ref.sha256}', 'source_path': ref.path, 'sha256': ref.sha256,
            'product_id': None, 'subject_id': None, 'reference_role': ref.role.value,
            'source_type': 'legacy_reference', 'image_description': ref.note or '', 'native_label': native_label,
        }
        records.append(record)
        started = time.monotonic()
        try:
            path = Path(ref.path)
            with path.open('rb') as stream:
                actual_sha = hashlib.file_digest(stream, 'sha256').hexdigest()
            if actual_sha != ref.sha256:
                raise ValueError('Reference SHA256 changed before image analysis')
            with Image.open(path) as image:
                image.load()
                display = ImageOps.exif_transpose(image)
                record.update(width=display.width, height=display.height)
            record['analysis_status'] = 'preflight_passed'
            trace(scope, 'image_preflight', [record], total_image_count=sum(r.kind is AssetKind.IMAGE for r in refs))
            key_input = {'protocol': PROTOCOL, 'sha256': ref.sha256, 'model': bound.arguments['backend'].cfg.model,
                         'analyser_version': ANALYZER_VERSION, 'schema': IMAGE_SCHEMA, 'prompt': IMAGE_INSTRUCTIONS,
                         'card_prompt': CARD_PROMPT, 'ocr_prompt': OCR_PROMPT,
                         'analysis_implementation': digest(inspect.getsource(visual_card)+inspect.getsource(validate_observation)),
                         'declared_identity': {key: record.get(key) for key in ('product_id', 'subject_id', 'reference_role')},
                         'note': ref.note, 'provenance': ref.provenance, 'composition': ref.composition,
                         'seed': ANALYSIS_SEED, 'source_components': source}
            key = digest(key_input)
            cache = cache_dir / (key + '.json')
            card = None
            if bound.arguments.get('use_cache', True) and cache.is_file():
                try:
                    saved = json.loads(cache.read_text(encoding='utf-8'))
                    if saved['key'] == key and saved['card_sha256'] == digest(saved['card']):
                        raw = saved['card']
                        card = AssetCard(**{**raw, 'kind': AssetKind(raw['kind'])})
                except (OSError, ValueError, KeyError, TypeError):
                    card = None
            cache_hit = card is not None
            if card is None:
                card = visual_card(bound.arguments['backend'], ref, ANALYSIS_SEED,
                                   scope['directory'] / f'image-{picture_index:03d}-inventory.json')
                raw = asdict(card)
                write(cache, {'key': key, 'inputs': key_input, 'card': raw, 'card_sha256': digest(raw)})
            cards[ref.sha256] = card
            record.update(analysis_status='analysed', image_description=card.summary, cache_hit=cache_hit,
                          analysis_cache_key=key, elapsed_seconds=round(time.monotonic()-started, 3))
            write(scope['directory'] / f'image-{picture_index:03d}.json', {'reference': record, 'analysis': asdict(card)})
            trace(scope, 'image_analysis_complete', [record])
        except Exception as error:
            record.update(analysis_status='failed', error_type=type(error).__name__, error=str(error))
            trace(scope, 'image_analysis_failed', [record], image_ordinal=picture_index,
                  function='analyse_references', error_type=type(error).__name__, error=str(error))
            write(scope['directory'] / 'reference-analyses.json', {'protocol': PROTOCOL, 'references': records, 'status': 'FAIL'})
            raise
    write(scope['directory'] / 'reference-analyses.json', {'protocol': PROTOCOL, 'references': records, 'status': 'PASS'})
    trace(scope, 'global_fusion_input', records, unique_image_hashes=len(cards),
          groups=contract.get('groups') or [], relationships_source='explicit caller product/subject ids; unbound references remain uncertain')
    return cards
