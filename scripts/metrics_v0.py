"""Step 0, v0: score each exercise occurrence with "Enabler" and "Stimulating", then test the scores.

Usage:
    uv run python scripts/metrics_v0.py ../MIAAM-v2/data --out metrics_v0_summary.txt

Unit of analysis: one zpdes interaction i = (student u, exercise e, activity a, objective o, time t).

ENABLER (does doing e help the student on this skill afterwards?)
    pre_acc(i)  = accuracy on the K interactions on the same skill (objective by default) before i
    post_acc(i) = accuracy on the K interactions on the same skill after i
    gain(i)     = post_acc - pre_acc
    Raw gain is confounded (regression to the mean, the student's general trend, ZPDES
    choosing what comes next). So we compare students in the SAME situation:
    stratum = (activity, pre_acc, how many skill interactions so far [log2 bucket])
    enabler(i)  = gain(i) - mean gain in its stratum

STIMULATING (does e keep the student working?)
    stay(i)       = 1 if the student does another exercise in the same session after i
    remaining(i)  = log(1 + number of exercises left in the session after i)
    stratum = (activity, position in session [log2 bucket], recent in-session accuracy)
    stimulating(i) = stay(i) - mean stay in its stratum   (remaining_resid reported too)

Exercise-level score = mean of the residuals over the exercise's occurrences.

TESTS
  T1 "Is the exercise served at random within an activity?" If ZPDES picks the activity and
     the exercise is then drawn at random, exercise comparisons within an activity are
     close to a natural experiment. We check whether pre_acc differs by exercise more than
     chance (permutation).
  T2 Split-half reliability: split STUDENTS into two disjoint halves, score exercises on each,
     correlate. A metric that captures a property of the exercise must replicate (r > 0),
     while a shuffled control gives r ~ 0.
  T3 Similar histories: in half B, group interactions by stratum (similar history) and by the
     decile of the exercise's score learned on half A. If the metric is real, students with
     the same history who got a higher-scored exercise do better afterwards.
  T4 Is the metric just difficulty? Correlate scores with exercise success rate, and with
     each other.

Outputs (in --outdir, default outputs/): exercise_scores_v0.parquet (+ interactions if
--save-interactions). Only aggregates are printed.
"""
import argparse
import hashlib
import io
import os
import sys

import numpy as np
import pandas as pd
import pyarrow.parquet as pq

OUT = io.StringIO()


def p(*a):
    s = " ".join(str(x) for x in a)
    OUT.write(s + "\n")
    print(s, flush=True)


def h(t):
    p(f"\n{'=' * 80}\n{t}\n{'=' * 80}")


def spearman(a, b):
    a, b = pd.Series(a).rank(), pd.Series(b).rank()
    return float(np.corrcoef(a, b)[0, 1]) if len(a) > 2 else float("nan")


def log2_bucket(x):
    return np.floor(np.log2(np.asarray(x, dtype=float) + 1)).astype(int)


# ----------------------------------------------------------------------------- load
def load(root, fname):
    cols = ["user_id", "exercise_id", "activity_id", "objective_id", "playlist_or_module_id",
            "created_at", "data_correct", "work_mode", "attempt_index", "session_id", "data_duration"]
    avail = set(pq.ParquetFile(os.path.join(root, fname)).schema_arrow.names)
    df = pq.read_table(os.path.join(root, fname), columns=[c for c in cols if c in avail]).to_pandas()
    for c in ["user_id", "exercise_id", "activity_id", "objective_id", "playlist_or_module_id",
              "work_mode", "session_id"]:
        if c in df:
            df[c] = df[c].astype("category")
    df["correct"] = df.data_correct.astype(np.int8)
    return df.sort_values(["user_id", "created_at"], kind="stable").reset_index(drop=True)


def add_sessions(df, gap_min):
    """Use session_id; rows with null session get a gap-based session."""
    same_user = df.user_id.eq(df.user_id.shift())
    gap = (df.created_at - df.created_at.shift()).dt.total_seconds()
    null = df.session_id.isna()
    sid = df.session_id.astype(str).where(~null, "<none>")
    new = ~same_user | (sid != sid.shift()) | (null & (gap > gap_min * 60))
    df["sess"] = new.cumsum().astype(np.int64)
    g = df.groupby("sess", sort=False)
    df["pos_in_sess"] = g.cumcount()
    df["sess_len"] = g.correct.transform("size")
    df["remaining"] = df.sess_len - df.pos_in_sess - 1
    df["stay"] = (df.remaining > 0).astype(np.int8)
    # in-session accuracy over the previous (up to) 5 interactions
    cs = g.correct.cumsum() - df.correct
    cs_lag = cs.groupby(df.sess).shift(5).fillna(0)
    n = np.minimum(df.pos_in_sess, 5)
    df["sess_recent_acc"] = np.where(n > 0, (cs - cs_lag) / n.replace(0, 1), np.nan)
    return df


def add_skill_windows(df, skill, K):
    """pre/post accuracy over K interactions on the same skill (vectorised with cumsums)."""
    order = df.sort_values(["user_id", skill, "created_at"], kind="stable").index
    d = df.loc[order, ["user_id", skill, "correct"]]
    g = d.groupby(["user_id", skill], observed=True, sort=False)
    k = g.cumcount()
    n = g.correct.transform("size")
    cs = g.correct.cumsum()                       # inclusive
    cs_g = cs.groupby([d.user_id, d[skill]], observed=True, sort=False)
    prev = cs - d.correct                         # sum of all before
    prev_lag = cs_g.shift(K + 1).fillna(0)        # sum up to K+1 before (inclusive)
    pre_sum = prev - prev_lag
    pre_n = np.minimum(k, K)
    post_end = cs_g.shift(-K)                     # inclusive sum up to K after
    post_n = np.minimum(n - k - 1, K)
    post_end = post_end.fillna(cs.groupby([d.user_id, d[skill]], observed=True, sort=False).transform("last"))
    post_sum = post_end - cs
    df.loc[order, "k_skill"] = k.values
    df.loc[order, "pre_n"] = pre_n.values
    df.loc[order, "post_n"] = post_n.values
    df.loc[order, "pre_acc"] = np.where(pre_n > 0, pre_sum / np.maximum(pre_n, 1), np.nan)
    df.loc[order, "post_acc"] = np.where(post_n > 0, post_sum / np.maximum(post_n, 1), np.nan)
    return df


def residualise(df, value, strata, min_stratum):
    g = df.groupby(strata, observed=True)[value]
    m = g.transform("mean")
    c = g.transform("size")
    return (df[value] - m).where(c >= min_stratum)


def user_half(u):
    return int(hashlib.md5(str(u).encode()).hexdigest(), 16) % 2


def exercise_scores(d, col, min_n):
    g = d.groupby(["activity_id", "exercise_id"], observed=True)[col].agg(["mean", "size"])
    return g[g["size"] >= min_n]


# ----------------------------------------------------------------------------- tests
def t1_randomisation(d, rng):
    h("T1. Is the exercise random within an activity? (pre_acc by exercise vs permutation)")
    x = d.dropna(subset=["pre_acc"])[["activity_id", "exercise_id", "pre_acc", "k_skill"]].copy()

    def between_var(frame, col):
        gm = frame.groupby(["activity_id", "exercise_id"], observed=True)[col].agg(["mean", "size"])
        am = frame.groupby("activity_id", observed=True)[col].mean()
        gm = gm.join(am.rename("am"), on="activity_id")
        return float(np.average((gm["mean"] - gm["am"]) ** 2, weights=gm["size"]))

    for col in ["pre_acc", "k_skill"]:
        obs = between_var(x, col)
        perm = []
        for _ in range(5):
            y = x.copy()
            y["exercise_id"] = y.groupby("activity_id", observed=True).exercise_id.transform(
                lambda s: s.sample(frac=1, random_state=int(rng.integers(1e9))).values)
            perm.append(between_var(y, col))
        p(f"  {col}: between-exercise variance within activity = {obs:.5f}; "
          f"shuffled = {np.mean(perm):.5f} (ratio {obs / np.mean(perm):.2f}; ~1 means as good as random)")


def t2_reliability(d, col, min_n, rng):
    a = exercise_scores(d[d.half == 0], col, min_n)
    b = exercise_scores(d[d.half == 1], col, min_n)
    j = a.join(b, lsuffix="_a", rsuffix="_b", how="inner")
    if len(j) < 10:
        p(f"  {col}: too few exercises ({len(j)})")
        return
    def within(jj):
        jj = jj.reset_index()
        jj = jj.groupby("activity_id").filter(lambda s: len(s) >= 3)
        return spearman(jj.mean_a - jj.groupby("activity_id").mean_a.transform("mean"),
                        jj.mean_b - jj.groupby("activity_id").mean_b.transform("mean"))

    r, r_w = spearman(j.mean_a, j.mean_b), within(j)
    # control: shuffle the values across interactions within each activity, which destroys
    # any exercise effect but keeps activity-level differences
    sh = d[["activity_id", "exercise_id", "half", col]].copy()
    sh[col] = sh.groupby("activity_id", observed=True)[col].transform(
        lambda s: s.sample(frac=1, random_state=int(rng.integers(1e9))).values)
    sa, sb = exercise_scores(sh[sh.half == 0], col, min_n), exercise_scores(sh[sh.half == 1], col, min_n)
    sj = sa.join(sb, lsuffix="_a", rsuffix="_b", how="inner")
    p(f"  {col:<16} exercises={len(j):,}  r(all)={r:+.3f}  r(within activity)={r_w:+.3f}  "
      f"| shuffled control: r(all)={spearman(sj.mean_a, sj.mean_b):+.3f} r(within)={within(sj):+.3f}")


def t5_history_dependence(d, col, state, min_n_cell):
    """Does the best exercise DEPEND on the student's history?

    Cell = (activity, exercise, state). Interaction effect = cell mean - exercise mean - state mean
    + activity mean (two-way decomposition within activity). If the interaction effects replicate
    across disjoint student halves, the right exercise depends on where the student is.
    """
    x = d.dropna(subset=[col] + state)[["activity_id", "exercise_id", "half", col] + state].copy()
    x["state"] = x[state].astype(str).agg("|".join, axis=1)

    def inter(frame):
        cell = frame.groupby(["activity_id", "exercise_id", "state"], observed=True)[col].agg(["mean", "size"])
        cell = cell[cell["size"] >= min_n_cell]
        ex = frame.groupby(["activity_id", "exercise_id"], observed=True)[col].mean().rename("ex")
        st = frame.groupby(["activity_id", "state"], observed=True)[col].mean().rename("st")
        ac = frame.groupby("activity_id", observed=True)[col].mean().rename("ac")
        c = cell.reset_index().join(ex, on=["activity_id", "exercise_id"]).join(
            st, on=["activity_id", "state"]).join(ac, on="activity_id")
        c["inter"] = c["mean"] - c.ex - c.st + c.ac
        return c.set_index(["activity_id", "exercise_id", "state"]).inter

    a, b = inter(x[x.half == 0]), inter(x[x.half == 1])
    j = pd.concat([a.rename("a"), b.rename("b")], axis=1, join="inner")
    sh = x.copy()
    sh["state"] = sh.groupby(["activity_id", "exercise_id"], observed=True).state.transform(
        lambda s: s.sample(frac=1, random_state=0).values)
    sa, sb = inter(sh[sh.half == 0]), inter(sh[sh.half == 1])
    sj = pd.concat([sa.rename("a"), sb.rename("b")], axis=1, join="inner")
    if len(j) < 20:
        p(f"  {col} x {state}: too few cells ({len(j)})")
        return
    p(f"  {col:<12} state={'+'.join(state):<22} cells={len(j):,}  r(interaction)={spearman(j.a, j.b):+.3f}"
      f"  | shuffled-state control r={spearman(sj.a, sj.b) if len(sj) > 2 else float('nan'):+.3f}")


def t3_similar_histories(d, col, outcome, min_n):
    a = exercise_scores(d[d.half == 0], col, min_n)["mean"].rename("score_a").reset_index()
    b = d[d.half == 1].merge(a, on=["activity_id", "exercise_id"], how="inner")
    b = b.dropna(subset=[outcome, "score_a"])
    if len(b) < 1000:
        p(f"  {col}: too few rows")
        return
    b["dec"] = pd.qcut(b.score_a.rank(method="first"), 10, labels=False)
    t = b.groupby("dec").agg(n=(outcome, "size"), outcome=(outcome, "mean"))
    p(f"  {col} (decile learned on half A) -> mean {outcome} in half B (same-history residual):")
    p("    " + "  ".join(f"D{int(i) + 1}:{r.outcome:+.4f}" for i, r in t.iterrows()))
    p(f"    top-minus-bottom decile = {t.outcome.iloc[-1] - t.outcome.iloc[0]:+.4f}  (n per decile ~{int(t.n.mean()):,})")


# ----------------------------------------------------------------------------- main
def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("root")
    ap.add_argument("--file", default="maths_data_filtered.parquet")
    ap.add_argument("--skill", default="objective_id", choices=["objective_id", "activity_id"])
    ap.add_argument("--K", type=int, default=5, help="window size for pre/post accuracy")
    ap.add_argument("--min-window", type=int, default=3, help="min interactions in pre and post")
    ap.add_argument("--min-stratum", type=int, default=20)
    ap.add_argument("--min-n", type=int, default=30, help="min occurrences for an exercise score")
    ap.add_argument("--session-gap-min", type=float, default=30)
    ap.add_argument("--outdir", default="outputs")
    ap.add_argument("--out", default="metrics_v0_summary.txt")
    ap.add_argument("--save-interactions", action="store_true")
    ap.add_argument("--sample-users", type=int, default=0, help="debug: only use N users")
    a = ap.parse_args()
    rng = np.random.default_rng(0)
    os.makedirs(a.outdir, exist_ok=True)
    p(f"python {sys.version.split()[0]}, pandas {pd.__version__}; args: {vars(a)}")

    df = load(os.path.expanduser(a.root), a.file)
    if a.sample_users:
        keep = df.user_id.drop_duplicates().sample(a.sample_users, random_state=0)
        df = df[df.user_id.isin(keep)].reset_index(drop=True)
    p(f"loaded {len(df):,} rows, {df.user_id.nunique():,} users")

    df = add_sessions(df, a.session_gap_min)
    df = add_skill_windows(df, a.skill, a.K)
    df["half"] = df.user_id.astype(str).map(user_half).astype(np.int8)
    df["k_bucket"] = log2_bucket(df.k_skill)
    df["pos_bucket"] = log2_bucket(df.pos_in_sess)
    df["pre_bin"] = (df.pre_acc * a.K).round()
    df["recent_bin"] = (df.sess_recent_acc * 4).round().fillna(-1)

    d = df[df.work_mode.astype(str) == "zpdes"].copy()
    p(f"zpdes interactions: {len(d):,}")

    # ---------------- Enabler
    ok = (d.pre_n >= a.min_window) & (d.post_n >= a.min_window)
    d["gain"] = (d.post_acc - d.pre_acc).where(ok)
    d["enabler"] = residualise(d.dropna(subset=["gain"]), "gain",
                               ["activity_id", "pre_bin", "k_bucket"], a.min_stratum).reindex(d.index)
    # ---------------- Stimulating
    d["log_remaining"] = np.log1p(d.remaining)
    strat = ["activity_id", "pos_bucket", "recent_bin"]
    d["stimulating"] = residualise(d, "stay", strat, a.min_stratum)
    d["remaining_resid"] = residualise(d, "log_remaining", strat, a.min_stratum)

    h("DESCRIPTIVES")
    p(f"skill level = {a.skill}, K = {a.K}")
    p(f"interactions with a valid Enabler: {d.enabler.notna().mean():.1%}; "
      f"Stimulating: {d.stimulating.notna().mean():.1%}")
    p("raw gain:", d.gain.describe().round(4).to_dict())
    p("raw gain by pre_bin (regression to the mean check):",
      d.groupby("pre_bin").gain.mean().round(3).to_dict())
    p("stay rate:", round(d.stay.mean(), 4), "| stay by pos_bucket:",
      d.groupby("pos_bucket").stay.mean().round(3).to_dict())
    p("stay by correctness of current exercise:", d.groupby("correct").stay.mean().round(4).to_dict())
    p("enabler resid by correctness of current exercise:",
      d.groupby("correct").enabler.mean().round(4).to_dict())

    t1_randomisation(d, rng)

    h("T2. Split-half reliability of exercise scores (disjoint students)")
    for col in ["enabler", "gain", "stimulating", "remaining_resid", "correct"]:
        t2_reliability(d, col, a.min_n, rng)
    p("  r(within activity) is the key number: it asks whether exercises of the SAME activity")
    p("  differ consistently. The shuffled control should be ~0 there.")
    p("  ('correct' = exercise success rate, shown as a reference for a known exercise property)")

    h("T3. Students with similar histories: does a higher-scored exercise lead to better outcomes?")
    t3_similar_histories(d, "enabler", "enabler", a.min_n)
    t3_similar_histories(d, "stimulating", "stimulating", a.min_n)
    t3_similar_histories(d, "stimulating", "remaining_resid", a.min_n)
    t3_similar_histories(d, "correct", "enabler", a.min_n)

    h("T5. Does the best exercise depend on the student's history? (exercise x state interaction)")
    p("  r > 0 (and control ~0) means some exercises are better for some histories than others,")
    p("  i.e. a per-(history, exercise) score carries information beyond a per-exercise score.")
    t5_history_dependence(d, "enabler", ["pre_bin"], 15)
    t5_history_dependence(d, "enabler", ["k_bucket"], 15)
    t5_history_dependence(d, "stimulating", ["recent_bin"], 15)
    t5_history_dependence(d, "stimulating", ["pos_bucket"], 15)

    h("T4. Exercise-level correlations (all students)")
    cols = ["enabler", "stimulating", "remaining_resid", "correct"]
    ex = d.groupby(["activity_id", "exercise_id"], observed=True).agg(
        n=("correct", "size"), success=("correct", "mean"),
        enabler=("enabler", "mean"), enabler_n=("enabler", "count"),
        stimulating=("stimulating", "mean"), remaining_resid=("remaining_resid", "mean"),
        median_duration_s=("data_duration", lambda s: s.clip(0, 3.6e6).median() / 1000))
    exn = ex[ex.n >= a.min_n]
    p(f"exercises with n >= {a.min_n}: {len(exn):,} of {len(ex):,}")
    p("Spearman correlation matrix:")
    p("  " + exn[["enabler", "stimulating", "remaining_resid", "success", "median_duration_s"]]
      .rank().corr().round(3).to_string().replace("\n", "\n  "))
    p("\nscore spread (exercise level, n >= min-n):")
    p("  " + exn[["enabler", "stimulating", "success"]].describe(percentiles=[.05, .25, .5, .75, .95])
      .round(4).to_string().replace("\n", "\n  "))

    ex.reset_index().to_parquet(os.path.join(a.outdir, "exercise_scores_v0.parquet"))
    p(f"\n[exercise scores written to {a.outdir}/exercise_scores_v0.parquet]")
    if a.save_interactions:
        keep = ["user_id", "sess", "pos_in_sess", "exercise_id", "activity_id", "objective_id", "created_at",
                "correct", "pre_acc", "post_acc", "gain", "enabler", "stay", "remaining", "stimulating",
                "remaining_resid", "k_skill", "half"]
        d[keep].to_parquet(os.path.join(a.outdir, "interaction_scores_v0.parquet"))
        p(f"[interaction scores written to {a.outdir}/interaction_scores_v0.parquet]")
    with open(a.out, "w") as f:
        f.write(OUT.getvalue())
    print(f"\n[summary written to {a.out}]")


if __name__ == "__main__":
    main()
