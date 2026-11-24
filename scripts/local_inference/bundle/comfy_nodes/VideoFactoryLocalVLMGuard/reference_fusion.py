"""Fuse verified reference views before native planning; never infer identity from order."""
from dataclasses import asdict, replace


def fuse_product_subjects(subjects, contract, proof):
    """Fuse only explicit, visually verified views of the same product.

    Only the exact, visually verified product views are merged.  Every other
    observed subject stays independent by default: a product photo can also
    contain a person, a hand, packaging or a second product, and none of those
    may be discarded without an explicit separate semantic decision.
    """
    if not contract or not proof or proof.get('status') != 'PASS':
        return subjects, {'status': 'NOT_APPLIED', 'reason': 'no complete visual product identity proof'}
    approved = proof['approved_product_descriptors_by_native']
    replacements, removed, groups = {}, set(), []
    for group in contract.get('groups', []):
        if group.get('category') != 'product':
            continue
        sources = group.get('appearance_native') or []
        matches = []
        for source in sources:
            candidates = [subject for subject in subjects
                          if source in subject.sources and subject.descriptor in approved.get(source, [])]
            if len(candidates) != 1:
                raise ValueError(f'Global fusion cannot uniquely bind {source} to the verified product {group["group"]}')
            if candidates[0] not in matches:
                matches.append(candidates[0])
        if not matches:
            continue
        if any(subject.label in removed or subject.label in replacements for subject in matches):
            raise ValueError('Global fusion found conflicting explicit product identities')
        anchor = matches[0]
        fused = replace(anchor,
            sources=list(dict.fromkeys(source for subject in matches for source in subject.sources)),
            attributes=list(dict.fromkeys(value for subject in matches for value in subject.attributes)))
        replacements[anchor.label] = fused
        removed.update(subject.label for subject in matches[1:])
        groups.append({'group_id': group['group'], 'category': 'product',
            'source': 'explicit product grouping plus actual cross-image identity verification',
            'native_sources': list(fused.sources), 'original_subjects': [asdict(subject) for subject in matches],
            'fused_subject': asdict(fused), 'attributes_discarded': False,
            'per_view_pose_source': 'original analysis cards retained independently'})
    result = [replacements.get(subject.label, subject) for subject in subjects
              if subject.label not in removed]
    result = [replace(subject, label=f'<Subject {index}>') for index, subject in enumerate(result, 1)]
    return result, {'status': 'PASS', 'groups': groups, 'original_subject_count': len(subjects),
        'fused_subject_count': len(result), 'image_inputs_removed': False,
        'unmatched_subjects': [asdict(subject) for subject in subjects
                               if subject.label not in set(replacements) | removed],
        'unmatched_subject_policy': 'retained as independent subjects; no person, package, prop or different product is discarded by source-image membership alone',
        'unbound_relationships': 'retained as independent subjects; never merged by filename, folder or array position',
        'subjects': [asdict(subject) for subject in result]}
