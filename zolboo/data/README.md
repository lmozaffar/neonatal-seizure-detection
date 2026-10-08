# Data

`tprocessed_dataset_part1.parquet` + `tprocessed_dataset_part2.parquet` together
are the full processed neoSeer training dataset (170,025 epochs, 132 subjects,
preictal labels 0/1/2 + 190 QEEG features). Split in two (by subject, ~93MB
each) to stay under GitHub's 100MB per-file limit. Load both and concatenate:

```python
import pandas as pd
df = pd.concat([
    pd.read_parquet("tprocessed_dataset_part1.parquet"),
    pd.read_parquet("tprocessed_dataset_part2.parquet"),
], ignore_index=True)
```

`tprocessed_dataset_metadata.csv` is the column-by-column data dictionary
(one row per column: type, family, description).

Regenerate from source via `../scripts/export_processed_dataset.py` and
`../scripts/export_metadata.py`.
