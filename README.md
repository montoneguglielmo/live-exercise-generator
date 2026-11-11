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
