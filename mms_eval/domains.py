"""Explicit domain contracts and portable array-to-manifest ingestion."""
from pathlib import Path

import numpy as np

from .images import as_rgb_image, save_png_atomic, validate_policy
from .utils import sha256_file, stable_hash, write_json, write_jsonl


def validate_domain_contract(config, *, mode='semantic', evaluator_config=None):
    if mode not in ('semantic', 'quality_only'):
        raise ValueError('mode must be semantic or quality_only')
    if not str(config.get('domain', '')).strip():
        raise ValueError('A domain name is required')
    validate_policy(config.get('image_policy'))
    version = config.get('dataset_version', '')
    if not version or any(x in version.lower() for x in ('pending', 'replace_me', 'to_audit')):
        raise ValueError('Record the actual dataset release/source identity before freezing a domain')
    if mode == 'quality_only':
        return {'mode': mode, 'domain': config['domain'], 'MMS': 'unavailable'}
    classes = config.get('classes', [])
    if (config.get('category_mode') != 'exclusive' or len(classes) < 2 or
            any(not isinstance(c, str) or not c for c in classes) or len(set(classes)) != len(classes)):
        raise ValueError('MMS requires at least two distinct exclusive classes; independent attributes are not classes')
    if not config.get('semantic_definition') or config.get('registration_status') == 'template':
        raise ValueError('Define the domain semantics and resolve the template before registration')
    if config['domain'].lower() == 'mnist':
        if classes != [str(i) for i in range(10)] or config.get('digit_polarity') not in ('light_on_dark', 'dark_on_light'):
            raise ValueError('MNIST requires ordered digit classes 0..9 and an explicit fixed polarity')
    audit = config.get('pretraining_audit', {})
    if audit.get('status') not in ('disjoint_declared', 'verified_manifest_disjoint', 'from_scratch') or not audit.get('evidence'):
        raise ValueError('Record pretraining provenance and calibration/held-out separation with evidence')
    weights = evaluator_config.get('weights','implicit_pretrained_default') if evaluator_config is not None else None
    if evaluator_config is not None and audit['status'] == 'from_scratch' and weights is not None:
        raise ValueError('from_scratch declaration conflicts with evaluator initialization weights')
    if config['domain'].lower() == 'imagenet' and audit.get('calibration_source') == 'imagenet1k_train':
        if evaluator_config is None or weights is not None:
            raise ValueError('ImageNet-pretrained evaluators cannot calibrate on ImageNet-1k training images; use a demonstrably separate holdout or train from scratch')
    return {'mode': mode, 'domain': config['domain'], 'classes': classes,
            'pretraining_audit': audit,
            'limitation': 'A provenance declaration is not an independent verification of pretraining membership or human MM validity.'}


def import_arrays(images, output_dir, *, layout, value_range, dataset_id, labels=None,
                  classes=None, source_directory=None, image_policy=None, setting_id=None):
    """Export .npy arrays without pickle; identities preserve file hash and row index.

    Generated data omit labels. Real labeled data declare classes and original
    source partition, then their JSONL manifests can be combined for ``prepare``.
    No automatic inversion, sample selection, cropping, or class inference.
    """
    out, source = Path(output_dir).resolve(), Path(images).resolve()
    if not dataset_id or not str(dataset_id).strip():
        raise ValueError('dataset_id must identify the exact dataset or generation setting/version')
    if layout not in ('NHW', 'NHWC', 'NCHW'):
        raise ValueError('layout must be NHW, NHWC, or NCHW')
    if out.exists() and any(out.iterdir()):
        raise FileExistsError('Import output already contains files; preserve it and use a new revision')
    data = np.load(source, allow_pickle=False, mmap_mode='r')
    if not isinstance(data, np.ndarray) or data.ndim != (3 if layout == 'NHW' else 4) or not len(data):
        raise ValueError('Expected a nonempty .npy array with the declared batch layout')
    policy = validate_policy(image_policy)
    label_values = None
    if labels is not None:
        label_values = np.load(labels, allow_pickle=False)
        if (not classes or len(set(classes)) != len(classes) or not source_directory or
                label_values.shape != (len(data),) or not np.issubdtype(label_values.dtype, np.integer) or
                label_values.min() < 0 or label_values.max() >= len(classes)):
            raise ValueError('Real arrays require aligned integer labels, distinct class names, and source_directory')
    elif classes is not None or source_directory is not None:
        raise ValueError('Class/source partition declarations require real labels')
    # Validate the first conversion before opening a new output cohort.
    single_layout = 'CHW' if layout == 'NCHW' else 'HWC'
    as_rgb_image(data[0], value_range=value_range, layout=single_layout, policy=policy)
    source_hash, rows = sha256_file(source), []
    for i, item in enumerate(data):
        image = as_rgb_image(item, value_range=value_range, layout=single_layout, policy=policy)
        path = out/'images'/f'{i:08d}.png'
        save_png_atomic(image, path)
        row = {'image_id': f'{dataset_id}:{source_hash[:16]}:{i}', 'path': str(path),
               'sha256': sha256_file(path), 'source_array_sha256': source_hash, 'source_index': i,
               'dataset_id': dataset_id}
        if setting_id is not None:
            row['setting_id'] = setting_id
        if label_values is not None:
            row.update(label=int(label_values[i]), class_name=classes[int(label_values[i])],
                       source_directory=source_directory)
        rows.append(row)
    write_jsonl(out/'manifest.jsonl', rows)
    result = {'n': len(rows), 'source': str(source), 'source_sha256': source_hash, 'dataset_id': dataset_id,
              'layout': layout, 'value_range': value_range, 'image_policy': policy,
              'labels_sha256': sha256_file(labels) if labels is not None else None,
              'classes': classes, 'source_directory': source_directory,
              'manifest_sha256': sha256_file(out/'manifest.jsonl')}
    result['signature'] = stable_hash(result)
    write_json(out/'import.json', result)
    return result
