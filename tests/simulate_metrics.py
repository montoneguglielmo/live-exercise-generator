"""Sanity check for metrics_v0 on simulated data with a KNOWN per-exercise effect.

    uv run python tests/simulate_metrics.py

Simulates students whose skill grows by a per-exercise amount ("true enabler") and who quit
the session with a per-exercise probability ("true stimulating"), runs metrics_v0 on it and
checks that the recovered exercise scores rank-correlate with the truth.
"""
import os
import subprocess
import sys
import tempfile

import numpy as np
import pandas as pd


def simulate(path, n_users=3000, seed=1):
    rng = np.random.default_rng(seed)
    acts = [(f"o{o}", f"o{o}a{a}") for o in range(4) for a in range(3)]
    ex = pd.DataFrame([dict(exercise_id=f"{a}e{e}", activity_id=a, objective_id=o,
                            enab=rng.normal(0, .15), stim=rng.normal(0, .6), diff=rng.normal(0, .5))
                       for o, a in acts for e in range(12)])
    by_act = {a: g.reset_index(drop=True) for a, g in ex.groupby("activity_id")}
    rows, t0 = [], pd.Timestamp("2000-01-01")
    for u in range(n_users):
        uid, t = f"u{u}", t0
        ability = {o: rng.normal(.5, 1) for o, _ in acts}
        for _ in range(rng.integers(1, 8)):
            t += pd.Timedelta(days=1)
            sid, o = f"am::{uid}::{t}", f"o{rng.integers(4)}"
            for _ in range(200):
                a = f"{o}a{rng.integers(3)}"
                e = by_act[a].iloc[rng.integers(len(by_act[a]))]
                c = rng.random() < 1 / (1 + np.exp(-(ability[o] - e["diff"])))
                ability[o] += .05 + e["enab"]
                t += pd.Timedelta(seconds=int(rng.integers(3, 30)))
                rows.append((uid, e.exercise_id, a, o, "m1", t, c, "zpdes", 1, sid, 5000.0))
                if rng.random() < 1 / (1 + np.exp(-(-3 - e["stim"] + (0 if c else .5)))):
                    break
    d = pd.DataFrame(rows, columns=["user_id", "exercise_id", "activity_id", "objective_id",
                                    "playlist_or_module_id", "created_at", "data_correct", "work_mode",
                                    "attempt_index", "session_id", "data_duration"])
    d["attempt_index"] = d.attempt_index.astype("uint32")
    d.to_parquet(os.path.join(path, "maths_data_filtered.parquet"))
    return ex


def main():
    with tempfile.TemporaryDirectory() as tmp:
        truth = simulate(tmp)
        script = os.path.join(os.path.dirname(__file__), "..", "scripts", "metrics_v0.py")
        subprocess.run([sys.executable, script, tmp, "--outdir", tmp, "--out", os.path.join(tmp, "s.txt")],
                       check=True, stdout=subprocess.DEVNULL)
        s = pd.read_parquet(os.path.join(tmp, "exercise_scores_v0.parquet")).merge(
            truth, on=["exercise_id", "activity_id"])
    rc = lambda a, b: s[a].rank().corr(s[b].rank())
    r_e, r_s = rc("enabler", "enab"), rc("stimulating", "stim")
    print(f"enabler vs true effect: {r_e:.3f}   stimulating vs true effect: {r_s:.3f}")
    assert r_e > .6 and r_s > .6, "metrics failed to recover the simulated effects"
    print("OK")


if __name__ == "__main__":
    main()
