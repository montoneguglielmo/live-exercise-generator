# live-exercise-generator

Generate the next exercise for a student on the EvidenceB platform from their recent
learning trajectory, the course knowledge graph and their mastery map.

## Step 0: what is a "good next exercise"?

Data (MIAAM V1/V2) lives on a remote server. Scripts here are run there and the
printed output is shared back.

### 1. Inspect the data

```bash
uv sync          # creates .venv and installs dependencies
uv run python scripts/inspect_data.py /path/to/MIAAM_V1 /path/to/MIAAM_V2 --out data_summary.txt
```

### 2. Explore structure for metric design

```bash
uv run python scripts/explore_miaam.py ../MIAAM/data --out explore_summary.txt
```
Prints only aggregates (no student-level rows). Use `--skip-full` to analyse only
`maths_data_filtered.parquet` if memory is tight.
