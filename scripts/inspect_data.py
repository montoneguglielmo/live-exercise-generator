"""Inspect a dataset folder and print a compact, pasteable summary.

Usage:
    python scripts/inspect_data.py /path/to/MIAAM_V1 [/path/to/MIAAM_V2 ...] [--out summary.txt]

For every tabular file found (csv, tsv, parquet, json, jsonl, feather, pickle, xlsx)
it prints: size, row count, columns with dtype / %null / #unique, and 3 sample rows.
Other files are only listed. Nothing is modified.
"""
import argparse
import io
import os
import sys

import pandas as pd

TABULAR = {".csv", ".tsv", ".parquet", ".pq", ".json", ".jsonl", ".feather",
           ".pkl", ".pickle", ".xlsx", ".xls"}
MAX_FILES = 40          # per folder
SAMPLE_ROWS = 200_000   # rows loaded to profile columns of large text files
MAX_COLS = 60


def human(n):
    for unit in ["B", "KB", "MB", "GB", "TB"]:
        if n < 1024:
            return f"{n:.1f}{unit}"
        n /= 1024
    return f"{n:.1f}PB"


def count_lines(path):
    with open(path, "rb") as f:
        return sum(buf.count(b"\n") for buf in iter(lambda: f.read(1 << 24), b""))


def load(path, ext):
    """Return (df, total_rows, note)."""
    if ext in {".csv", ".tsv"}:
        sep = "\t" if ext == ".tsv" else None
        df = pd.read_csv(path, sep=sep, engine="python" if sep is None else "c",
                         nrows=SAMPLE_ROWS)
        total = count_lines(path) - 1
        note = f"(profiled first {len(df):,} rows)" if total > len(df) else ""
        return df, total, note
    if ext in {".parquet", ".pq"}:
        try:  # read only the first row group(s) so huge files don't blow memory
            import pyarrow.parquet as pq
            f = pq.ParquetFile(path)
            total = f.metadata.num_rows
            tables, n = [], 0
            for i in range(f.num_row_groups):
                t = f.read_row_group(i)
                tables.append(t)
                n += t.num_rows
                if n >= SAMPLE_ROWS:
                    break
            import pyarrow as pa
            df = pa.concat_tables(tables).to_pandas()
            note = f"(profiled first {len(df):,} rows)" if total > len(df) else ""
            return df, total, note
        except ImportError:
            df = pd.read_parquet(path)
    elif ext == ".jsonl":
        df = pd.read_json(path, lines=True, nrows=SAMPLE_ROWS)
        return df, count_lines(path), ""
    elif ext == ".json":
        df = pd.read_json(path)
    elif ext == ".feather":
        df = pd.read_feather(path)
    elif ext in {".pkl", ".pickle"}:
        obj = pd.read_pickle(path)
        if not isinstance(obj, pd.DataFrame):
            raise ValueError(f"pickle holds {type(obj).__name__}, not a DataFrame")
        df = obj
    else:
        df = pd.read_excel(path)
    return df, len(df), ""


def describe(df, out):
    cols = list(df.columns)[:MAX_COLS]
    out.write(f"  columns ({len(df.columns)}):\n")
    for c in cols:
        s = df[c]
        try:
            nun = s.nunique(dropna=True)
        except TypeError:  # unhashable (lists/dicts)
            nun = "n/a"
        out.write(f"    - {c!s:<35} {str(s.dtype):<15} null={s.isna().mean():6.1%}  unique={nun}\n")
    if len(df.columns) > MAX_COLS:
        out.write(f"    ... {len(df.columns) - MAX_COLS} more columns\n")
    with pd.option_context("display.max_columns", 30, "display.width", 250,
                           "display.max_colwidth", 60):
        out.write("  sample rows:\n")
        out.write("    " + df.head(3).to_string().replace("\n", "\n    ") + "\n")


def inspect_folder(root, out):
    out.write(f"\n{'=' * 80}\nFOLDER: {root}\n{'=' * 80}\n")
    if not os.path.exists(root):
        out.write("  !! path does not exist\n")
        return
    other, seen = [], 0
    for dirpath, dirs, files in os.walk(root):
        dirs[:] = sorted(d for d in dirs if not d.startswith("."))  # skip .cache, .git
        files = [f for f in files if not f.startswith(".")]
        for name in sorted(files):
            path = os.path.join(dirpath, name)
            rel = os.path.relpath(path, root)
            ext = os.path.splitext(name)[1].lower()
            size = os.path.getsize(path)
            if ext not in TABULAR:
                other.append(f"{rel} ({human(size)})")
                continue
            seen += 1
            if seen > MAX_FILES:
                out.write(f"\n-- {rel} ({human(size)}) [skipped, file limit reached]\n")
                continue
            out.write(f"\n-- {rel} ({human(size)})\n")
            try:
                df, total, note = load(path, ext)
                out.write(f"  rows: {total:,} {note}\n")
                describe(df, out)
            except Exception as e:  # keep going on bad files
                out.write(f"  !! could not read: {type(e).__name__}: {e}\n")
    if other:
        out.write(f"\n  other files ({len(other)}):\n")
        for o in other[:50]:
            out.write(f"    {o}\n")
        if len(other) > 50:
            out.write(f"    ... {len(other) - 50} more\n")


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("folders", nargs="+")
    ap.add_argument("--out", default="data_summary.txt")
    args = ap.parse_args()
    buf = io.StringIO()
    buf.write(f"python {sys.version.split()[0]}, pandas {pd.__version__}\n")
    for f in args.folders:
        inspect_folder(os.path.expanduser(f), buf)
    text = buf.getvalue()
    print(text)
    with open(args.out, "w") as fh:
        fh.write(text)
    print(f"\n[summary also written to {args.out}]")


if __name__ == "__main__":
    main()
