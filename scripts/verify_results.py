"""Verify archived Office-Home measurements without loading models or images."""
import json
from pathlib import Path

import numpy as np
import pandas as pd
from sklearn.metrics import accuracy_score, f1_score

ROOT = Path(__file__).resolve().parents[1]
DATA = ROOT / 'results'


def require(condition, message):
    if not condition:
        raise AssertionError(message)


def close(actual, expected, label):
    require(np.isclose(actual, expected, rtol=1e-10, atol=1e-12),
            f'{label}: computed {actual}, saved {expected}')


def main():
    config = json.loads((DATA / 'config.json').read_text())
    mapping = json.loads((DATA / 'class_to_idx.json').read_text())
    labels = list(range(65))
    require(sorted(mapping.values()) == labels, 'Class mapping must contain labels 0 through 64.')
    inverse = {v: k for k, v in mapping.items()}
    manifest = pd.read_csv(DATA / 'split_manifest.csv')
    require(manifest.relative_path.is_unique, 'Repeated image paths in manifest.')
    require(manifest.sha256.is_unique, 'Repeated image hashes after deduplication.')
    require(len(manifest) == 15176, 'Unexpected retained dataset size.')
    require(set(manifest.domain) == {'Art', 'Clipart', 'Product', 'Real_World'}, 'Unexpected domains.')
    require(manifest.split.isin(['train', 'val', 'test']).all(), 'Unknown split.')
    require((manifest.label == manifest.class_name.map(mapping)).all(), 'Manifest class mapping mismatch.')
    require((manifest.groupby(['domain', 'split']).label.nunique() == 65).all(), 'Missing class in a split.')
    require(len(pd.read_csv(DATA / 'exact_duplicates_removed.csv')) == 412, 'Deduplication count mismatch.')
    require((DATA / 'rejected_images.csv').read_text().strip() == '', 'Unexpected rejected images.')
    results = pd.read_csv(DATA / 'results_all.csv')
    expected = len(config['source_domains']) * len(config['models']) * len(config['modes']) * len(config['seeds']) * 4
    require(len(results) == expected, 'Unexpected number of evaluation rows.')
    require(not results.duplicated(['source', 'model', 'mode', 'seed', 'test_domain']).any(), 'Duplicate evaluation row.')
    require(results.groupby('experiment').test_domain.nunique().eq(4).all(), 'Incomplete experiment.')

    for row in results.itertuples(index=False):
        folder = DATA / row.experiment
        predictions = pd.read_csv(folder / f'predictions_{row.test_domain}.csv')
        test = manifest[(manifest.domain == row.test_domain) & (manifest.split == 'test')].set_index('relative_path')
        require(predictions.relative_path.is_unique, f'{row.experiment}: repeated predictions.')
        require(set(predictions.relative_path) == set(test.index), f'{row.test_domain}: test membership mismatch.')
        require(len(predictions) == row.n, 'Evaluation sample count mismatch.')
        require(np.array_equal(predictions.y_true, test.loc[predictions.relative_path, 'label']), 'Ground-truth mismatch.')
        require(predictions.y_pred.isin(labels).all(), 'Predicted class outside label space.')
        require((predictions.true_class == predictions.y_true.map(inverse)).all(), 'True class name mismatch.')
        require((predictions.predicted_class == predictions.y_pred.map(inverse)).all(), 'Predicted class name mismatch.')
        require(predictions.confidence.between(0, 1).all(), 'Invalid softmax score.')
        acc = accuracy_score(predictions.y_true, predictions.y_pred)
        f1 = f1_score(predictions.y_true, predictions.y_pred, labels=labels, average='macro', zero_division=0)
        close(acc, row.accuracy, 'accuracy')
        close(f1, row.macro_f1, 'macro-F1')
        report = json.loads((folder / f'classification_report_{row.test_domain}.json').read_text())
        close(report['accuracy'], acc, 'per-class report accuracy')
        close(report['macro avg']['f1-score'], f1, 'per-class report macro-F1')
        source = results[(results.experiment == row.experiment) & (results.test_domain == row.source)].iloc[0]
        close((source.accuracy - acc) * 100, row.accuracy_drop_pp, 'accuracy difference')
        history = pd.read_csv(folder / 'history.csv')
        best, epoch = -1., None
        for h in history.itertuples(index=False):
            if h.val_macro_f1 > best + 1e-6:
                best, epoch = h.val_macro_f1, h.epoch
        require(epoch == row.best_epoch, 'Validation-selected checkpoint epoch mismatch.')
        close(best, row.best_val_macro_f1, 'best validation macro-F1')
        done = json.loads((folder / 'training_complete.json').read_text())
        require(done['epochs_completed'] == len(history), 'Training completion record mismatch.')
        close(done['best_val_macro_f1'], best, 'training completion macro-F1')

    print(f'PASS: {len(manifest):,} retained images; 65 classes; four domains; '
          f'{results.experiment.nunique()} completed configurations; {len(results)} verified evaluations.')
    print('Accuracy, macro-F1, test membership, class mappings, domain differences, and checkpoint selection match.')
    print('Losses require logits and are not recomputed from top-1 predictions. No models were retrained.')


if __name__ == '__main__':
    main()
