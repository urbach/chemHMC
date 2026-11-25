"""Plot results of run_schemes.py: energy error vs time step and vs cost, with error bars over initial conditions.

With several seeds (initial conditions) every curve is the geometric mean over the seeds and the band is
the standard error of log10(error), i.e. a "x/÷ factor" band (--band sd shows the spread between seeds instead).
A second row then shows the spread between initial conditions against the time step, and with --ratio REF
the paired ratio RMS(scheme)/RMS(REF) at equal cost, taken seed by seed (much tighter than the bands).

Examples:
  python3 plot_schemes.py --cutoff switch                       # all schemes in the low-density results
  python3 plot_schemes.py --density high --cutoff shift 'n2q1BA#0' 'n4q3BA#0' --ratio 'n2q1BA#0'
  python3 plot_schemes.py --cutoff switch --table                # also print the numbers
At most 8 schemes per plot (one colour each); pass scheme ids to choose them.
"""
import argparse, json
from collections import defaultdict
from pathlib import Path

import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

TESTSYS = Path(__file__).resolve().parent.parent
KT = 0.5961 * 4.184e-4               # k_B * 300 K in amu A^2/fs^2
COLORS = ["#2a78d6", "#eb6834", "#1baf7a", "#eda100", "#e87ba4", "#008300", "#6250d6", "#e34948"]
MARKERS = ["o", "s", "^", "D", "v", "P", "X", "<"]
INK, MUTED, GRID, REF = "#1f1f1e", "#6b6a63", "#e4e3dd", "#8c8b84"


def slope(x, y, lo, hi):
    """log-log slope over the points with lo <= y <= hi (asymptotic range, above round-off)."""
    m = (y >= lo) & (y <= hi)
    if m.sum() < 3:          # too few points in the window: use the three smallest steps above lo
        idx = np.where(y >= lo)[0][np.argsort(x[y >= lo])][:3]
        m = np.zeros_like(m); m[idx] = True
    return np.polyfit(np.log(x[m]), np.log(y[m]), 1)[0] if m.sum() >= 3 else np.nan


def log_stats(values):
    """geometric mean, standard error and standard deviation of log10(values)."""
    l = np.log10(values)
    sd = l.std(ddof=1) if len(l) > 1 else 0.0
    return 10 ** l.mean(), sd / np.sqrt(len(l)), sd


def style(ax):
    ax.grid(True, which="major", color=GRID, lw=0.8); ax.set_axisbelow(True)
    for sp in ("top", "right"): ax.spines[sp].set_visible(False)


def main():
    p = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("schemes", nargs="*", help="scheme ids to plot (default: all in the results, max 8)")
    p.add_argument("--density", choices=["low", "high"], default="low")
    p.add_argument("--cutoff", default="switch")
    p.add_argument("--rc", type=float, default=12.0)
    p.add_argument("--T", type=float, default=100.0)
    p.add_argument("--results", type=Path)
    p.add_argument("--out", type=Path, help="figure file (default next to the results)")
    p.add_argument("--band", choices=["se", "sd"], default="se",
                   help="band: standard error of the mean (se, default) or spread between seeds (sd)")
    p.add_argument("--ratio", metavar="REF", help="also plot the paired ratio RMS(scheme)/RMS(REF) at equal cost")
    p.add_argument("--fit", type=float, nargs=2, default=(1e-13, 1e-8), metavar=("YMIN", "YMAX"),
                   help="fit local slopes to points with YMIN <= max|dE| <= YMAX (default 1e-13 1e-8)")
    p.add_argument("--diverged", type=float, default=1e-1, help="|dE| above this counts as diverged (default 0.1)")
    p.add_argument("--table", action="store_true", help="print the numbers for every scheme and step count")
    args = p.parse_args()
    args.results = args.results or TESTSYS / "Data" / f"{args.density}_dens" / "Scheme_tests" / "results.json"

    rows = [r for r in json.loads(args.results.read_text()).values()
            if r["density"] == args.density and r["cutoff"] == args.cutoff and r["rc"] == args.rc and r["T"] == args.T]
    ids = args.schemes or sorted({r["scheme"] for r in rows}, key=lambda s: (int(s[1:s.index("q")]), s))
    if not rows or not ids:
        raise SystemExit("no matching results")
    if len(ids) > len(COLORS):
        raise SystemExit(f"{len(ids)} schemes match; pick at most {len(COLORS)} by passing their ids")

    # (scheme, n) -> {seed: run}
    runs = defaultdict(dict)
    for r in rows:
        runs[(r["scheme"], r["n"])][r.get("seed", 123)] = r
    ok = lambda r: r["status"] == "ok" and r["maxabs"] <= args.diverged
    nseeds = max(len(v) for v in runs.values())
    T = args.T

    plt.rcParams.update({"font.size": 11, "axes.edgecolor": MUTED, "axes.labelcolor": INK,
                         "xtick.color": MUTED, "ytick.color": MUTED, "text.color": INK})
    two_rows = nseeds > 1 or args.ratio
    fig, axes = plt.subplots(2 if two_rows else 1, 2, figsize=(14, 11 if two_rows else 6.6), squeeze=False)
    ax1, ax2 = axes[0]
    for ax in axes.flat: style(ax)

    stats = {}
    for i, sid in enumerate(ids):
        ns = sorted(n for (s, n) in runs if s == sid)
        if not ns:
            print(f"no results for {sid}"); continue
        q = next(iter(runs[(sid, ns[0])].values()))["q"]
        order = next(iter(runs[(sid, ns[0])].values()))["order"]
        st = stats[sid] = dict(q=q, n=[], gm=[], se=[], sd=[], rms=[], rse=[], rsd=[], bad=[], nbad={})
        for n in ns:
            rs = list(runs[(sid, n)].values())
            nbad = sum(not ok(r) for r in rs)
            st["nbad"][n] = (nbad, len(rs))
            if nbad:
                st["bad"].append(n); continue
            g, se, sd = log_stats([r["maxabs"] for r in rs])
            gr, rse, rsd = log_stats([r["rms"] for r in rs])
            for k, v in (("n", n), ("gm", g), ("se", se), ("sd", sd), ("rms", gr), ("rse", rse), ("rsd", rsd)):
                st[k].append(v)
        for k in ("n", "gm", "se", "sd", "rms", "rse", "rsd"):
            st[k] = np.array(st[k])
        h, cost = T / st["n"], q * st["n"] / T
        w = st["se"] if args.band == "se" else st["sd"]
        wr = st["rse"] if args.band == "se" else st["rsd"]
        sty = dict(color=COLORS[i], marker=MARKERS[i], lw=2, ms=4, mec="white", mew=0.5)
        label = f"{sid} (order {order}, q={q}): local slope {slope(h, st['gm'], *args.fit):.1f}"
        ax1.loglog(h, st["gm"], label=label, **sty)
        ax1.fill_between(h, st["gm"] / 10 ** w, st["gm"] * 10 ** w, color=COLORS[i], alpha=0.2, lw=0)
        ax2.loglog(cost, st["rms"], **sty)
        ax2.fill_between(cost, st["rms"] / 10 ** wr, st["rms"] * 10 ** wr, color=COLORS[i], alpha=0.2, lw=0)
        ymark = args.diverged * 0.7 / 1.5 ** i
        if st["bad"]:
            ax1.plot([T / n for n in st["bad"]], [ymark] * len(st["bad"]), ls="none", marker="x", ms=7, mew=1.8,
                     color=COLORS[i], clip_on=False)
            ax2.plot([q * n / T for n in st["bad"]], [ymark] * len(st["bad"]), ls="none", marker="x", ms=7, mew=1.8,
                     color=COLORS[i], clip_on=False)
        if two_rows and nseeds > 1:
            axes[1, 0].semilogx(h, st["sd"], **sty)

    for ax in (ax1, ax2):
        x0, x1 = ax.get_xlim()
        ax.plot([x0, x1], [KT, KT], color=REF, lw=1, ls=":", zorder=0); ax.set_xlim(x0, x1)
    ax1.text(ax1.get_xlim()[0] * 1.3, KT * 1.6, "1 kT (whole box)", color=MUTED, fontsize=9)
    ylo = min(ax1.get_ylim()[0], ax2.get_ylim()[0])
    for ax in (ax1, ax2): ax.set_ylim(ylo, args.diverged)
    ax1.set_xlabel("time step $h$ [fs]"); ax2.set_xlabel("cost $q/h$ [force evaluations per fs]")
    ax1.set_ylabel(r"max$_{t\leq T}\,|\Delta E(t)|$ [amu Å$^2$/fs$^2$]")
    ax2.set_ylabel(r"RMS of $\Delta E$ over $[0, T]$ [amu Å$^2$/fs$^2$]")
    top1 = ax1.secondary_xaxis("top", functions=(lambda x: T / np.maximum(x, 1e-300), lambda n: T / np.maximum(n, 1e-300)))
    top1.set_xlabel("number of steps $T/h$", color=MUTED, fontsize=10)
    top2 = ax2.secondary_xaxis("top", functions=(lambda c: c * T, lambda f: f / T))
    top2.set_xlabel(f"force evaluations to reach T = {T:g} fs", color=MUTED, fontsize=10)

    if two_rows:
        a3, a4 = axes[1]
        if nseeds > 1:
            a3.set_xlim(ax1.get_xlim()); a3.set_ylim(bottom=0)
            a3.set_xlabel("time step $h$ [fs]")
            a3.set_ylabel("spread between initial conditions\nSD of log$_{10}$ max|ΔE|  [decades]")
            a3.text(0.98, 0.03, f"error of the mean = SD/√{nseeds}", transform=a3.transAxes, ha="right", va="bottom",
                    color=MUTED, fontsize=9)
        else:
            a3.set_visible(False)
        if args.ratio and args.ratio in stats:
            rq = stats[args.ratio]["q"]
            for i, sid in enumerate(ids):
                if sid == args.ratio or sid not in stats: continue
                q = stats[sid]["q"]; xs, gm, se = [], [], []
                for n in sorted(n for (s, n) in runs if s == sid):
                    if (q * n) % rq or (args.ratio, q * n // rq) not in runs: continue
                    A, B = runs[(sid, n)], runs[(args.ratio, q * n // rq)]
                    seeds = [sd for sd in A if sd in B and ok(A[sd]) and ok(B[sd])]
                    if len(seeds) < max(2, len(A) // 2): continue
                    g, e, _ = log_stats([A[sd]["rms"] / B[sd]["rms"] for sd in seeds])
                    xs.append(q * n / T); gm.append(g); se.append(e)
                xs, gm, se = map(np.array, (xs, gm, se))
                if len(xs):
                    a4.errorbar(xs, gm, yerr=[gm - gm / 10 ** se, gm * 10 ** se - gm], color=COLORS[i],
                                marker=MARKERS[i], ms=5, mec="white", mew=0.5, lw=1.5, capsize=3, label=sid)
            a4.set_xscale("log"); a4.set_yscale("log")
            a4.axhline(1, color=REF, lw=1, ls=":")
            a4.set_xlabel("cost $q/h$ [force evaluations per fs]")
            a4.set_ylabel(f"RMS / RMS of {args.ratio} at equal cost\n(paired by seed, ± error of the mean)")
            a4.text(0.02, 0.03, f"below 1: better than {args.ratio}", transform=a4.transAxes, color=MUTED, fontsize=9)
        else:
            a4.set_visible(False)

    band = "± error of the mean" if args.band == "se" else "± spread between seeds"
    fig.legend(*ax1.get_legend_handles_labels(), loc="upper center", ncol=2, frameon=False, fontsize=9.5,
               bbox_to_anchor=(0.5, 0.965 if two_rows else 0.95))
    fig.suptitle(f"{args.density} density, cutoff_scheme {args.cutoff}, $r_c$ = {args.rc:g} Å, T = {T:g} fs, "
                 f"{nseeds} initial condition{'s' if nseeds > 1 else ''}"
                 + (f" (geometric mean, band {band})" if nseeds > 1 else "") + "; × = diverged", fontsize=12, y=0.995)
    nleg = (len(ids) + 1) // 2
    fig.tight_layout(rect=(0, 0, 1, 1 - (0.022 if two_rows else 0.035) * (1 + nleg)))
    out = args.out or args.results.parent / f"schemes_{args.density}_{args.cutoff}_rc{args.rc:g}_T{T:g}.png"
    fig.savefig(out, dpi=120); fig.savefig(Path(out).with_suffix(".pdf"))
    print(f"saved {out}")

    if args.table:
        for sid, st in stats.items():
            print(f"\n{sid}   max|dE|: geometric mean, x/÷ error of the mean, spread (SD of log10); diverged seeds")
            print(f"{'n':>6s} {'h':>8s} {'max|dE|':>10s} {'x/÷ SE':>8s} {'SD dec':>7s} {'diverged':>9s}")
            good = dict(zip(st["n"], zip(st["gm"], st["se"], st["sd"])))
            for n, (nb, m) in sorted(st["nbad"].items()):
                if n in good:
                    g, se, sd = good[n]
                    print(f"{n:6d} {T / n:8.3g} {g:10.2e} {10 ** se:8.2f} {sd:7.2f} {nb:>4d}/{m}")
                else:
                    print(f"{n:6d} {T / n:8.3g} {'':10s} {'':8s} {'':7s} {nb:>4d}/{m}")


if __name__ == "__main__":
    main()
