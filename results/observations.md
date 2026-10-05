# Experiment observations

Smoke test: False
Seeds: [42]

Real_World / resnet18 / linear_probe: same-domain accuracy = 79.42%.
  Art: accuracy 55.59%, Macro-F1 0.5182, drop 23.83 pp.
  Clipart: accuracy 36.22%, Macro-F1 0.3345, drop 43.19 pp.
  Product: accuracy 67.34%, Macro-F1 0.6550, drop 12.08 pp.
Real_World / resnet18 / finetune_last_block: same-domain accuracy = 74.81%.
  Art: accuracy 49.00%, Macro-F1 0.4423, drop 25.81 pp.
  Clipart: accuracy 33.75%, Macro-F1 0.3160, drop 41.06 pp.
  Product: accuracy 60.68%, Macro-F1 0.5828, drop 14.13 pp.
Real_World / resnet50 / linear_probe: same-domain accuracy = 82.03%.
  Art: accuracy 62.75%, Macro-F1 0.5898, drop 19.28 pp.
  Clipart: accuracy 37.15%, Macro-F1 0.3837, drop 44.88 pp.
  Product: accuracy 72.91%, Macro-F1 0.7150, drop 9.12 pp.
Real_World / resnet50 / finetune_last_block: same-domain accuracy = 82.64%.
  Art: accuracy 61.60%, Macro-F1 0.5781, drop 21.04 pp.
  Clipart: accuracy 40.87%, Macro-F1 0.4372, drop 41.78 pp.
  Product: accuracy 71.83%, Macro-F1 0.7042, drop 10.82 pp.
Fine-tuning change for resnet18 (pp): Art -6.59, Clipart -2.48, Product -6.66, Real_World -4.61
Fine-tuning change for resnet50 (pp): Art -1.15, Clipart +3.72, Product -1.08, Real_World +0.61

Limitations:
- Source-only protocol; target labels were not used to train or select checkpoints.
- Same-domain and cross-domain test sets differ in difficulty and composition.
- Exact byte deduplication does not detect near-duplicate images.
- Office-Home measures visual-style differences; no medical/satellite/industrial datasets were evaluated.
- One seed does not measure training variability.
- Only the last residual block was fine-tuned; full-backbone fine-tuning was not evaluated.
- ImageNet pretraining overlap with collected web images cannot be ruled out.