"""Second-pass exploration of MIAAM, aimed at designing the "good next exercise" metrics.

Usage:
    uv run python scripts/explore_miaam.py ../MIAAM/data --out explore_summary.txt

Prints (and writes to --out) a compact report:
  1. Outline of the JSON files (structure only, truncated values), incl. the knowledge graph.
  2. The dataset README.
  3. Exercise table: hierarchy module > objective > activity > exercise.
  4. For maths_data and maths_data_filtered: volumes, work modes, sessions,
     attempts, timing, how trajectories move through the hierarchy.
  5. Overlap between the two interaction files.
No raw student-level rows are printed, only aggregates.
"""
import argparse
import io
import json
import os
import sys

import numpy as np
import pandas as pd
import pyarrow.parquet as pq

OUT = io.StringIO()


def parse_content(c):
    """Exercise content is JSON in V1 but may be a Python dict repr in V2."""
    import ast
    if isinstance(c, dict):
        return c
    try:
        return json.loads(c)
    except Exception:
        try:
            v = ast.literal_eval(c)
            return v if isinstance(v, dict) else {"<non-dict>": v}
        except Exception:
            return None


def p(*args):
    OUT.write(" ".join(str(a) for a in args) + "\n")


def h(title):
    p(f"\n{'=' * 80}\n{title}\n{'=' * 80}")


def q(s, qs=(0.05, 0.25, 0.5, 0.75, 0.95, 0.99)):
    s = pd.Series(s).dropna()
    if s.empty:
        return "empty"
    vals = ", ".join(f"p{int(x * 100)}={s.quantile(x):.4g}" for x in qs)
    return f"n={len(s):,} mean={s.mean():.4g} min={s.min():.4g} {vals} max={s.max():.4g}"


# ---------------------------------------------------------------- JSON outline
def outline(obj, depth=0, max_depth=6, max_keys=12, key="root"):
    pad = "  " * depth
    if isinstance(obj, dict):
        keys = list(obj.keys())
        uuid_like = len(keys) > 3 and all(len(k) == 36 and k.count("-") == 4 for k in keys[:5])
        p(f"{pad}{key}: dict[{len(keys)}]" + (" (keys are ids)" if uuid_like else ""))
        if depth >= max_depth:
            return
        shown = keys[:2] if uuid_like else keys[:max_keys]
        for k in shown:
            outline(obj[k], depth + 1, max_depth, max_keys, key=k)
        if len(keys) > len(shown):
            p(f"{pad}  ... {len(keys) - len(shown)} more keys"
              + ("" if uuid_like else f": {keys[len(shown):len(shown) + 20]}"))
    elif isinstance(obj, list):
        types = sorted({type(x).__name__ for x in obj})
        p(f"{pad}{key}: list[{len(obj)}] of {types}")
        if obj and depth < max_depth:
            outline(obj[0], depth + 1, max_depth, max_keys, key="[0]")
    else:
        v = repr(obj)
        p(f"{pad}{key}: {type(obj).__name__} = {v[:100]}{'...' if len(v) > 100 else ''}")


def json_section(root):
    h("1. JSON FILES (structure)")
    for name in ["maths_dependencies.json", "descriptions.json",
                 "maths_data.parquet.manifest.json", "maths_data_filtered.parquet.manifest.json"]:
        path = os.path.join(root, name)
        if not os.path.exists(path):
            p(f"\n-- {name}: missing")
            continue
        p(f"\n-- {name}")
        try:
            with open(path) as f:
                outline(json.load(f))
        except Exception as e:
            p(f"  !! {type(e).__name__}: {e}")
    # knowledge graph: try to find edge-like fields
    path = os.path.join(root, "maths_dependencies.json")
    if os.path.exists(path):
        with open(path) as f:
            dep = json.load(f)
        p("\n-- maths_dependencies.json: keys seen at any depth (count)")
        counts = {}

        def walk(o):
            if isinstance(o, dict):
                for k, v in o.items():
                    if not (len(k) == 36 and k.count("-") == 4):
                        counts[k] = counts.get(k, 0) + 1
                    walk(v)
            elif isinstance(o, list):
                for v in o:
                    walk(v)
        walk(dep)
        for k, c in sorted(counts.items(), key=lambda x: -x[1])[:60]:
            p(f"  {k}: {c}")


# ---------------------------------------------------------------- README
def readme_section(root):
    h("2. DATASET README")
    path = os.path.join(root, "README.md")
    if os.path.exists(path):
        with open(path) as f:
            p(f.read()[:15000])


# ---------------------------------------------------------------- exercises
def exercises_section(root):
    h("3. EXERCISE TABLE")
    ex = pd.read_parquet(os.path.join(root, "maths_exercises_table.parquet"))
    p(f"rows={len(ex):,} unique exercise_id={ex.exercise_id.nunique():,}")
    dup = ex[ex.exercise_id.duplicated(keep=False)]
    p(f"duplicated exercise_id rows: {len(dup)}; differing columns among duplicates:",
      [c for c in ex.columns if dup.groupby('exercise_id')[c].nunique().gt(1).any()] if len(dup) else [])
    p("source:", ex.source.value_counts().to_dict())
    p("gameplay_type:", ex.gameplay_type.value_counts().to_dict())
    p("\nper module: objectives / activities / exercises")
    g = ex.groupby(["module_id", "module_name"]).agg(
        objectives=("objective_id", "nunique"), activities=("activity_id", "nunique"),
        exercises=("exercise_id", "nunique"))
    p(g.to_string())
    p("\nexercises per activity:", q(ex.groupby("activity_id").exercise_id.nunique()))
    p("activities per objective:", q(ex.groupby("objective_id").activity_id.nunique()))
    p("objective_id -> objective_name is 1:1?",
      ex.groupby("objective_id").objective_name.nunique().max() == 1)
    p("activity in >1 objective:", int((ex.groupby("activity_id").objective_id.nunique() > 1).sum()))
    p("\nexample objectives (name | intent):")
    for _, r in ex.drop_duplicates("objective_id").head(8).iterrows():
        p(f"  [{r.module_name}] {r.objective_name} | {str(r.objective_pedagogical_intent)[:150]}")
    p("\ncontent JSON keys (first 200 rows):")
    p("  first chars of content:", repr(str(ex.content.iloc[0])[:120]))
    parsed = [parse_content(c) for c in ex.content]
    p(f"  parseable content: {np.mean([x is not None for x in parsed]):.1%}")
    keys = {}
    for x in parsed[:500]:
        for k in (x or {"<unparseable>": 1}):
            keys[k] = keys.get(k, 0) + 1
    p("  keys (first 500 rows):", keys)
    ok = [x for x in parsed if x is not None]
    if ok:
        p("  empty correct_answer share:",
          f"{np.mean([x.get('correct_answer', '') in ('', None, [], {}) for x in ok]):.1%}")
    return ex


# ---------------------------------------------------------------- interactions
COLS = ["user_id", "classroom_id", "playlist_or_module_id", "exercise_id", "created_at",
        "data_correct", "work_mode", "data_duration", "source", "attempt_index", "session_id"]


def load(path):
    avail = set(pq.ParquetFile(path).schema_arrow.names)
    cols = [c for c in COLS + ["objective_id", "activity_id"] if c in avail]  # V2 has these
    t = pq.read_table(path, columns=cols)
    df = t.to_pandas()
    for c in ["user_id", "classroom_id", "playlist_or_module_id", "exercise_id",
              "work_mode", "source", "session_id", "objective_id", "activity_id"]:
        if c not in df:
            continue
        df[c] = df[c].astype("category")
    return df


def interactions_section(root, name, ex):
    h(f"4. INTERACTIONS: {name}")
    df = load(os.path.join(root, name))
    p(f"rows={len(df):,} users={df.user_id.nunique():,} classrooms={df.classroom_id.nunique():,} "
      f"exercises={df.exercise_id.nunique():,} sessions={df.session_id.nunique():,}")
    p(f"date range: {df.created_at.min()} -> {df.created_at.max()}")
    p("created_at per user starts at:", df.groupby("user_id", observed=True).created_at.min().dt.date.astype(str).value_counts().head(5).to_dict())
    p("by month:", df.created_at.dt.to_period("M").value_counts().sort_index().to_dict())
    p("hour-of-day of created_at (check if time is real):",
      df.created_at.dt.hour.value_counts().sort_index().to_dict())
    p("source:", df.source.value_counts().to_dict())
    p("work_mode x source:\n" + pd.crosstab(df.work_mode, df.source).to_string())
    p("work_mode:", df.work_mode.value_counts().to_dict())
    p("correct rate by work_mode:", df.groupby("work_mode", observed=True).data_correct.mean().round(3).to_dict())
    p("playlist_or_module_id:", df.playlist_or_module_id.value_counts().to_dict())
    mod_names = ex.drop_duplicates("module_id").set_index("module_id").module_name.to_dict()
    p("  as module names:", {mod_names.get(k, "?"): v for k, v in df.playlist_or_module_id.value_counts().items()})
    p("overall correct rate:", round(df.data_correct.mean(), 3))
    p("data_duration (ms):", q(df.data_duration))

    known = df.exercise_id.isin(set(ex.exercise_id))
    p(f"interactions whose exercise_id is in exercise table: {known.mean():.2%}")
    exm = ex.drop_duplicates("exercise_id").set_index("exercise_id")
    if "objective_id" in df:  # V2: trust the log's own activity/objective (exercises can be reused)
        df["objective_id"] = df.objective_id.astype(str)
        df["activity_id"] = df.activity_id.astype(str)
        act_mod = ex.drop_duplicates("activity_id").set_index("activity_id").module_id
        df["module_id"] = df.activity_id.map(act_mod)
        p("log activity_id found in exercise table:", f"{df.activity_id.isin(set(ex.activity_id)).mean():.2%}")
        pairs = set(zip(ex.exercise_id, ex.activity_id))
        smp = df.sample(min(len(df), 200_000), random_state=0)
        p("log (exercise, activity) pair found in table:",
          f"{np.mean([(e, a) in pairs for e, a in zip(smp.exercise_id.astype(str), smp.activity_id)]):.2%}")
        p("exercises appearing under >1 activity in the log:",
          int((df.groupby("exercise_id", observed=True).activity_id.nunique() > 1).sum()))
    else:
        df["objective_id"] = df.exercise_id.astype(str).map(exm.objective_id)
        df["activity_id"] = df.exercise_id.astype(str).map(exm.activity_id)
        df["module_id"] = df.exercise_id.astype(str).map(exm.module_id)
    p("playlist_or_module_id == exercise module_id:",
      f"{(df.playlist_or_module_id.astype(str) == df.module_id).mean():.2%}")

    p("\nattempt_index:", df.attempt_index.clip(upper=6).value_counts().sort_index().to_dict(), "(6 = 6+)")
    p("correct rate by attempt_index:",
      df.groupby(df.attempt_index.clip(upper=6)).data_correct.mean().round(3).to_dict())

    p("\nper user: interactions", q(df.groupby("user_id", observed=True).size()))
    p("per user: sessions", q(df.groupby("user_id", observed=True).session_id.nunique()))
    p("per user: active days", q(df.groupby("user_id", observed=True).created_at.apply(lambda s: s.dt.date.nunique())))
    p("per user: objectives touched", q(df.groupby("user_id", observed=True).objective_id.nunique()))
    p("users per classroom", q(df.groupby("classroom_id", observed=True).user_id.nunique()))
    p(f"null session_id rows: {df.session_id.isna().mean():.2%} (work_mode of those: "
      f"{df[df.session_id.isna()].work_mode.value_counts().to_dict()})")
    p("sessions spanning >1 user:", int((df.groupby("session_id", observed=True).user_id.nunique() > 1).sum()))

    df = df.sort_values(["user_id", "created_at"], kind="stable").reset_index(drop=True)
    same_user = df.user_id.eq(df.user_id.shift())
    gap = (df.created_at - df.created_at.shift()).dt.total_seconds().where(same_user)
    p("\ngap to previous interaction of same user (s):", q(gap))
    same_sess = same_user & df.session_id.eq(df.session_id.shift())
    p("gap within same session (s):", q(gap[same_sess]))
    p("gap when session changes (s):", q(gap[same_user & ~same_sess]))
    p("ties (gap==0) share:", f"{(gap == 0).mean():.2%}")

    s = df.dropna(subset=["session_id"]).groupby("session_id", observed=True)
    p("\nsession length (interactions):", q(s.size()))
    p("session span (min):", q((s.created_at.max() - s.created_at.min()).dt.total_seconds() / 60))
    p("work_modes per session:", s.work_mode.nunique().value_counts().to_dict())
    p("objectives per session:", q(s.objective_id.nunique()))
    p("activities per session:", q(s.activity_id.nunique()))

    nxt = same_sess.shift(-1, fill_value=False)
    for lvl in ["exercise_id", "activity_id", "objective_id", "module_id"]:
        stay = (df[lvl].astype(str) == df[lvl].astype(str).shift(-1))[nxt]
        p(f"next interaction in same session stays on same {lvl}: {stay.mean():.2%}")
    rep = (df.exercise_id.astype(str) == df.exercise_id.astype(str).shift(-1))[nxt]
    p("immediate repeat after wrong vs right:",
      df[nxt].assign(rep=rep).groupby("data_correct").rep.mean().round(3).to_dict())

    # how much signal per (user, objective): needed for before/after performance
    uo = df.groupby(["user_id", "objective_id"], observed=True).size()
    p("\ninteractions per (user, objective):", q(uo))
    p("share of (user, objective) with >=10 interactions:", f"{(uo >= 10).mean():.2%}")
    ua = df.groupby(["user_id", "activity_id"], observed=True).size()
    p("interactions per (user, activity):", q(ua))
    # learning curve per objective visit index
    df["k_obj"] = df.groupby(["user_id", "objective_id"], observed=True).cumcount()
    curve = df.groupby(df.k_obj.clip(upper=50) // 5 * 5).data_correct.agg(["mean", "size"])
    p("correct rate vs nth interaction on same objective (bucketed by 5):")
    p("  ", {int(k): (round(float(r["mean"]), 3), int(r["size"])) for k, r in curve.iterrows()})
    return set(df.user_id.astype(str)), df


def overlap_section(u1, u2, d1, d2):
    h("5. maths_data vs maths_data_filtered")
    p(f"users: only_full={len(u1 - u2):,} only_filtered={len(u2 - u1):,} both={len(u1 & u2):,}")
    for c in ["created_at", "work_mode"]:
        p(f"{c}: full {d1[c].min() if c == 'created_at' else d1[c].value_counts().to_dict()} | "
          f"filtered {d2[c].min() if c == 'created_at' else d2[c].value_counts().to_dict()}")
    common = list(u1 & u2)[:200]
    a = d1[d1.user_id.astype(str).isin(common)].groupby(d1.user_id.astype(str)).size()
    b = d2[d2.user_id.astype(str).isin(common)].groupby(d2.user_id.astype(str)).size()
    p("for 200 common users, rows filtered/full:", q((b / a).dropna()))


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("root")
    ap.add_argument("--out", default="explore_summary.txt")
    ap.add_argument("--skip-full", action="store_true", help="only analyse maths_data_filtered")
    a = ap.parse_args()
    root = os.path.expanduser(a.root)
    p(f"python {sys.version.split()[0]}, pandas {pd.__version__}")
    def safe(fn, *args):
        try:
            return fn(*args)
        except Exception as e:
            import traceback
            p(f"\n!! {fn.__name__} failed: {type(e).__name__}: {e}\n" + traceback.format_exc()[-1500:])
            return None

    safe(json_section, root)
    safe(readme_section, root)
    ex = exercises_section(root)  # needed by everything below
    r2 = safe(interactions_section, root, "maths_data_filtered.parquet", ex)
    if not a.skip_full:
        r1 = safe(interactions_section, root, "maths_data.parquet", ex)
        if r1 and r2:
            safe(overlap_section, r1[0], r2[0], r1[1], r2[1])
    text = OUT.getvalue()
    print(text)
    with open(a.out, "w") as f:
        f.write(text)
    print(f"\n[written to {a.out}]")


if __name__ == "__main__":
    main()
