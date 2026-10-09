"""Metrics for raw CNN predictions and the complete acceptance policy."""
from __future__ import annotations

import numpy as np
from sklearn.metrics import confusion_matrix, precision_recall_fscore_support

from freshlens_ai.constants import CLASSES, FRUITS, STATUSES


def ratio(numerator, denominator):
    return float(numerator / denominator) if denominator else None


def classification(truth, predicted, labels):
    if not truth:
        return None
    precision, recall, f1, support = precision_recall_fscore_support(
        truth, predicted, labels=list(labels), zero_division=0)
    return {
        'accuracy': float(np.mean(np.asarray(truth) == np.asarray(predicted))),
        'macro_f1': float(f1.mean()),
        'balanced_accuracy': float(recall[support > 0].mean()),
        'labels': list(labels),
        'confusion_matrix': confusion_matrix(truth, predicted, labels=list(labels)).tolist(),
        'per_class': {label: dict(precision=float(p), recall=float(r), f1=float(f), support=int(s))
                      for label, p, r, f, s in zip(labels, precision, recall, f1, support)},
    }


def summarize(records):
    known = [r for r in records if r['known']]
    unknown = [r for r in records if not r['known']]
    accepted = [r for r in known if r['supported']]
    joint_correct = lambda r: r['true_joint'] == r['joint_class']
    fruit_correct = lambda r: r['true_fruit'] == r['fruit']
    bins = []
    ece = 0.0
    for index in range(10):
        selected = [r for r in known if min(int(r['joint_confidence'] * 10), 9) == index]
        accuracy = ratio(sum(joint_correct(r) for r in selected), len(selected))
        confidence = float(np.mean([r['joint_confidence'] for r in selected])) if selected else None
        if selected:
            ece += len(selected) / len(known) * abs(accuracy - confidence)
        bins.append(dict(lower=index / 10, upper=(index + 1) / 10, count=len(selected),
                         accuracy=accuracy, mean_confidence=confidence))
    good = [r for r in records if r.get('quality_label') == 'good']
    bad = [r for r in records if r.get('quality_label') == 'bad']
    latencies = [r['latency_ms'] for r in records]
    return {
        'total_images': len(records), 'total_known': len(known), 'total_unknown': len(unknown),
        'cnn_only': {
            'joint': classification([r['true_joint'] for r in known], [r['joint_class'] for r in known], CLASSES),
            'fruit': classification([r['true_fruit'] for r in known], [r['fruit'] for r in known], FRUITS),
            'condition': classification([r['true_condition'] for r in known], [r['condition'] for r in known], STATUSES),
        },
        'policy': {
            'known_accepted': len(accepted), 'known_rejected': len(known) - len(accepted),
            'known_coverage': ratio(len(accepted), len(known)),
            'known_quality_rejected': sum(r['status'] == 'quality_rejection' for r in known),
            'known_openset_rejected': sum(r['status'] == 'openset_rejection' for r in known),
            'end_to_end_fruit_success': ratio(sum(fruit_correct(r) for r in accepted), len(known)),
            'end_to_end_joint_success': ratio(sum(joint_correct(r) for r in accepted), len(known)),
            'accepted_only_joint_accuracy': ratio(sum(joint_correct(r) for r in accepted), len(accepted)),
            'unknown_false_acceptance_rate': ratio(sum(r['supported'] for r in unknown), len(unknown)),
            'unknown_rejection_rate': ratio(sum(not r['supported'] for r in unknown), len(unknown)),
            'unknown_false_accepted': sum(r['supported'] for r in unknown),
            'unknown_rejected': sum(not r['supported'] for r in unknown),
        },
        'gate_only': {
            'known_acceptance': ratio(sum(r['gate_supported'] for r in known), len(known)),
            'unknown_rejection': ratio(sum(not r['gate_supported'] for r in unknown), len(unknown)),
            'known_joint_outcomes': {
                f'cnn_{correct_name}_gate_{accept_name}': sum(
                    joint_correct(r) == correct and r['gate_supported'] == accept for r in known)
                for correct, correct_name in [(True, 'correct'), (False, 'wrong')]
                for accept, accept_name in [(True, 'accept'), (False, 'reject')]
            },
        },
        'quality_diagnostic': {
            'good_labeled': len(good), 'bad_labeled': len(bad),
            'good_false_rejection_rate': ratio(sum(not r['quality_passed'] for r in good), len(good)),
            'bad_rejection_rate': ratio(sum(not r['quality_passed'] for r in bad), len(bad)),
        },
        'calibration': {'joint_ece_10_bins': ece if known else None, 'bins': bins},
        'latency': {'mean_ms': float(np.mean(latencies)) if latencies else None,
                    'p95_ms': float(np.percentile(latencies, 95)) if latencies else None},
        'definitions': {
            'cnn_only': 'All decoded known images, including gate/quality rejections.',
            'policy': 'Quality rejection takes precedence over gate rejection when enabled.',
            'macro_f1': 'Unweighted average over the fixed supported class list, absent classes count as zero.',
            'balanced_accuracy': 'Mean recall of classes with ground-truth support.',
            'missing_denominator': 'Rates are null when the required population is absent.',
            'quality_diagnostic': 'Measured before resizing; human good/bad labels are optional. Thresholds are experimental.',
            'latency': 'Decode, quality measurements and full CNN/gate inference; one warm-up excluded, CUDA synchronized.',
        },
    }
