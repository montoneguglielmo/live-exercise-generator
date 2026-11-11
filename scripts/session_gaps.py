"""Histogram of time gaps between a student's consecutive interactions (text output, pasteable).

Usage:
    uv run python scripts/session_gaps.py ../MIAAM-v2/data --out session_gaps_summary.txt

Shows, on log-spaced bins, the gap distribution for (a) all consecutive interactions and
(b) only those where session_id changes. Used to choose the "sitting" threshold for the
Stimulating metric (a new sitting = gap longer than T, instead of a new session_id).
"""
import argparse
import os

import numpy as np
import pandas as pd
import pyarrow.parquet as pq

BINS = [0, 10, 30, 60, 120, 300, 600, 900, 1800, 3600, 2 * 3600, 6 * 3600, 12 * 3600,
        86400, 2 * 86400, 7 * 86400, 30 * 86400, 365 * 86400, np.inf]
LABELS = ["<10s", "10-30s", "30s-1m", "1-2m", "2-5m", "5-10m", "10-15m", "15-30m", "30m-1h",
          "1-2h", "2-6h", "6-12h", "12-24h", "1-2d", "2-7d", "7-30d", "30d-1y", ">1y"]


def hist(gaps, title, out, width=50):
    c = pd.cut(gaps, BINS, labels=LABELS, right=False).value_counts().reindex(LABELS).fillna(0)
    tot = c.sum()
    out.append(f"\n{title}  (n={int(tot):,})")
    cum = 0
    for lab, v in c.items():
        cum += v
        bar = "#" * int(round(width * v / c.max())) if c.max() else ""
        out.append(f"  {lab:>8} {int(v):>10,} {v / tot:6.1%}  cum {cum / tot:6.1%}  {bar}")


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("root")
    ap.add_argument("--file", default="maths_data_filtered.parquet")
    ap.add_argument("--out", default="session_gaps_summary.txt")
    a = ap.parse_args()
    path = os.path.join(os.path.expanduser(a.root), a.file)
    df = pq.read_table(path, columns=["user_id", "created_at", "session_id", "playlist_or_module_id"]).to_pandas()
    df = df.sort_values(["user_id", "created_at"], kind="stable").reset_index(drop=True)
    same_user = df.user_id.eq(df.user_id.shift())
    gap = (df.created_at - df.created_at.shift()).dt.total_seconds()[same_user]
    sid = df.session_id.fillna("<none>")
    changed = (sid != sid.shift())[same_user]
    mod_changed = (df.playlist_or_module_id != df.playlist_or_module_id.shift())[same_user]
    out = [f"file={a.file}"]
    hist(gap, "A. gap between ALL consecutive interactions of a student", out)
    hist(gap[changed], "B. gap where session_id CHANGES", out)
    hist(gap[~changed], "C. gap WITHIN the same session_id", out)
    out.append("\nsession_id changes with gap < 30 min: "
               f"{(gap[changed] < 1800).mean():.1%}; of those, module also changes: "
               f"{mod_changed[changed & (gap < 1800)].mean():.1%}")
    for t in [5, 10, 15, 30, 60, 120]:
        n_sit = int((gap > t * 60).sum() + df.user_id.nunique())
        out.append(f"threshold {t:>4} min -> sittings={n_sit:,}  "
                   f"(session_ids={df.session_id.nunique():,})")
    text = "\n".join(out)
    print(text)
    with open(a.out, "w") as f:
        f.write(text + "\n")
    print(f"\n[written to {a.out}]")


if __name__ == "__main__":
    main()
