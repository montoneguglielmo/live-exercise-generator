"""How does the adaptive (placement) test appear in MIAAM?

Usage:
    uv run python scripts/adaptive_test.py ../MIAAM-v2/data --out adaptive_test_summary.txt

Questions answered (aggregates only):
  - How many students take it, how many items per test, per module?
  - Is it only at the very beginning, or do students take it again later (re-tests)?
    Re-tests would give an independent before/after measure for the Enabler metric.
  - Which objectives does it cover, and does the test score predict later zpdes success?
"""
import argparse
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


def q(s, qs=(0.05, 0.25, 0.5, 0.75, 0.95)):
    s = pd.Series(s).dropna()
    if s.empty:
        return "empty"
    return f"n={len(s):,} mean={s.mean():.3g} " + " ".join(f"p{int(x * 100)}={s.quantile(x):.3g}" for x in qs) \
        + f" max={s.max():.3g}"


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("root")
    ap.add_argument("--file", default="maths_data_filtered.parquet")
    ap.add_argument("--out", default="adaptive_test_summary.txt")
    a = ap.parse_args()
    path = os.path.join(os.path.expanduser(a.root), a.file)
    cols = ["user_id", "playlist_or_module_id", "objective_id", "activity_id", "exercise_id",
            "created_at", "data_correct", "work_mode", "session_id", "source"]
    avail = set(pq.ParquetFile(path).schema_arrow.names)
    df = pq.read_table(path, columns=[c for c in cols if c in avail]).to_pandas()
    for c in ["user_id", "playlist_or_module_id", "objective_id", "activity_id", "exercise_id",
              "work_mode", "session_id", "source"]:
        if c in df:
            df[c] = df[c].astype("category")
    df = df.sort_values(["user_id", "created_at"], kind="stable").reset_index(drop=True)
    df["wm"] = df.work_mode.astype(str)
    df["is_test"] = df.wm == "adaptive-test"
    p(f"python {sys.version.split()[0]}, pandas {pd.__version__}; file={a.file}")
    p(f"rows={len(df):,}; work_mode values: {df.wm.value_counts().to_dict()}")

    # ---- episodes: maximal runs of consecutive test interactions per user (and module)
    new_user = df.user_id.ne(df.user_id.shift())
    mod = df.playlist_or_module_id.astype(str)
    run_break = new_user | df.is_test.ne(df.is_test.shift()) | mod.ne(mod.shift())
    df["run"] = run_break.cumsum()
    df["k_user"] = df.groupby("user_id", observed=True).cumcount()
    t = df[df.is_test]
    ep = t.groupby("run").agg(user=("user_id", "first"), module=("playlist_or_module_id", "first"),
                              source=("source", "first"), n_items=("is_test", "size"),
                              acc=("data_correct", "mean"), start=("created_at", "min"),
                              end=("created_at", "max"), k_first=("k_user", "min"),
                              objectives=("objective_id", "nunique"))
    ep["user"] = ep.user.astype(str)
    ep["module"] = ep.module.astype(str)
    ep["minutes"] = (ep.end - ep.start).dt.total_seconds() / 60
    ep = ep.sort_values(["user", "start"])
    ep["nth_test_of_user_in_module"] = ep.groupby(["user", "module"]).cumcount() + 1

    users = df.user_id.nunique()
    tusers = t.user_id.nunique()
    p(f"\nstudents: {users:,}; with >=1 test item: {tusers:,} ({tusers / users:.1%})")
    p("test items per student:", q(t.groupby("user_id", observed=True).size()))
    p(f"test episodes (consecutive test items in one module): {len(ep):,}")
    p("items per episode:", q(ep.n_items))
    p("items per episode, most common:", ep.n_items.value_counts().head(12).to_dict())
    p("minutes per episode:", q(ep.minutes))
    p("objectives covered per episode:", q(ep.objectives))
    p("accuracy per episode:", q(ep.acc))
    p("episodes by source:", ep.source.astype(str).value_counts().to_dict())

    p("\n-- WHEN is the test taken?")
    p("index of the student's first interaction in the episode (0 = very first thing they do):",
      q(ep.k_first))
    p(f"episodes starting at the student's first interaction: {(ep.k_first == 0).mean():.1%}")
    first_in_mod = df.groupby(["user_id", "playlist_or_module_id"], observed=True).k_user.transform("min")
    t_first = t.assign(fm=first_in_mod[t.index]).groupby("run").apply(
        lambda g: (g.k_user.min() == g.fm.min()), include_groups=False)
    p(f"episodes starting at the student's first interaction IN THAT MODULE: {t_first.mean():.1%}")
    p("episodes per (student, module):", ep.groupby(["user", "module"]).size().value_counts().sort_index().to_dict())

    p("\n-- RE-TESTS (same student, same module, more than one episode)")
    multi = ep[ep.groupby(["user", "module"]).n_items.transform("size") > 1]
    p(f"(student, module) pairs with a re-test: {multi.groupby(['user', 'module']).ngroups:,}")
    if len(multi):
        first = multi[multi.nth_test_of_user_in_module == 1].set_index(["user", "module"])
        second = multi[multi.nth_test_of_user_in_module == 2].set_index(["user", "module"])
        j = first.join(second, lsuffix="_1", rsuffix="_2", how="inner")
        j["days_between"] = (j.start_2 - j.end_1).dt.total_seconds() / 86400
        j["zpdes_items_between"] = j.k_first_2 - (j.k_first_1 + j.n_items_1)
        p("days between test 1 and test 2:", q(j.days_between))
        p("interactions between test 1 and test 2:", q(j.zpdes_items_between))
        p("accuracy test 1:", q(j.acc_1), "\naccuracy test 2:", q(j.acc_2))
        p("gain (test2 - test1):", q(j.acc_2 - j.acc_1))
        p("gain vs practice in between (Spearman):",
          round(float(j.zpdes_items_between.rank().corr((j.acc_2 - j.acc_1).rank())), 3))
        run_ex = t.groupby("run").exercise_id.apply(lambda x: set(x.astype(str)))
        r1 = first.loc[j.index].reset_index().merge(ep.reset_index()[["run", "user", "module", "nth_test_of_user_in_module"]]
                                                    .query("nth_test_of_user_in_module == 1"), on=["user", "module"]).run
        r2 = second.loc[j.index].reset_index().merge(ep.reset_index()[["run", "user", "module", "nth_test_of_user_in_module"]]
                                                     .query("nth_test_of_user_in_module == 2"), on=["user", "module"]).run
        ov = [len(run_ex[x] & run_ex[y]) / len(run_ex[y]) for x, y in zip(r1, r2)]
        p("share of test-2 exercises already seen in test 1:")
        p("  overlap:", q(ov))

    p("\n-- Does the first test predict later zpdes success? (per student, module)")
    zp = df[df.wm == "zpdes"].groupby(["user_id", "playlist_or_module_id"], observed=True).agg(
        zp_acc=("data_correct", "mean"), zp_n=("data_correct", "size")).reset_index()
    zp["user"] = zp.user_id.astype(str)
    zp["module"] = zp.playlist_or_module_id.astype(str)
    f1 = ep[ep.nth_test_of_user_in_module == 1][["user", "module", "acc", "n_items"]]
    m = f1.merge(zp, on=["user", "module"])
    m = m[m.zp_n >= 20]
    if len(m) > 10:
        p(f"pairs: {len(m):,}; Spearman(test acc, later zpdes acc) = "
          f"{float(m.acc.rank().corr(m.zp_acc.rank())):.3f}; with zpdes volume = "
          f"{float(m.acc.rank().corr(m.zp_n.rank())):.3f}")

    p("\n-- Test items: which objectives/activities (top 15 objectives by items)")
    p(t.objective_id.astype(str).value_counts().head(15).to_string())
    p(f"distinct exercises used in tests: {t.exercise_id.nunique():,}; also used in zpdes: "
      f"{t.exercise_id.astype(str).drop_duplicates().isin(set(df[df.wm == 'zpdes'].exercise_id.astype(str))).mean():.1%}")

    with open(a.out, "w") as f:
        f.write(OUT.getvalue())
    print(f"\n[written to {a.out}]")


if __name__ == "__main__":
    main()
