# Contributing

This is a collaborative university project. Changes should preserve the measured experiment outputs and clearly describe what was reviewed or improved.

If each member wants their GitHub account to appear in the repository contributor history, they should make and push one real review or documentation improvement using their own Git account. Suggested review tasks are:

| Member | Suggested review | Suggested commit |
|---|---|---|
| Nguyễn Tiến Dũng | Review the experiment design and final integration. | `docs: refine experiment design and project overview` |
| Nguyễn Ngọc Hiếu | Review provenance and reproducibility. | `docs: refine dataset provenance and reproducibility` |
| Vũ Minh Châu | Review the preprocessing section. | `docs: clarify preprocessing and data pipeline` |
| Nguyễn Minh Hiếu | Review the model and training section. | `docs: clarify ResNet training configuration` |
| Lê Đức Anh | Review the results and evaluation section. | `docs: refine evaluation and domain-shift analysis` |
| Hoàng Lê Anh Đức | Review the report and references. | `docs: polish report and references` |

Do not change archived metrics unless a verified issue is found and documented. Before committing, run `python scripts/verify_results.py` and check that no datasets, checkpoints, credentials, caches, or temporary files are staged.
