"""Energy-error sweeps for splitting schemes on the argon test systems.

Every run integrates for the same physical time T with n = T/dt steps, so runs with different dt
(and different schemes) are directly comparable. The initial positions come from argon.xyz and the
momenta are drawn at 300 K with the random seed; --seeds runs several initial conditions, which
plot_schemes.py turns into error bars.
Results are added to a JSON file; runs that are already in it are skipped, so a sweep can be
stopped and restarted, or extended with more schemes or step counts.

Examples:
  # order check with the smooth cutoff (clean slopes)
  python3 run_schemes.py 'n2q1BA#0' n4:top2 n6:top3 --cutoff switch
  # production potential, dense system
  python3 run_schemes.py 'n2q1BA#0' n4:top2 n6:top3 --cutoff shift --density high
  # 16 initial conditions (momenta seeds) for error bars
  python3 run_schemes.py 'n2q1BA#0' n4q3BA:top1 --seeds 1-16 --density high --cutoff shift
  # see how much work a sweep is before running it
  python3 run_schemes.py n6q7:top4 --dry-run

Schemes: ids like n6q7AB#0, or selections like n6q7AB:top3, n6q7:top3 (both orderings), n6:top5.
"""
import argparse, json, os, re, subprocess, sys, tempfile, time
from concurrent.futures import ThreadPoolExecutor, as_completed
from pathlib import Path

import numpy as np
import schemes as S

HERE = Path(__file__).resolve().parent
TESTSYS = HERE.parent                                  # code/argon_testsystem
EXE = TESTSYS.parent / "builds" / "openMP" / "main" / "main"
SYSTEMS = {"low": dict(L=100.0, max_neighbors=450), "high": dict(L=32.0, max_neighbors=700)}
DEFAULT_STEPS = "1-20,25,40,50,80,100,200,400,500,1000"
PAT = re.compile(r"trajectory (\d+): deltaE = (\S+)")


def parse_steps(text):
    out = set()
    for part in text.split(","):
        lo, _, hi = part.partition("-")
        out.update(range(int(lo), int(hi) + 1) if hi else [int(lo)])
    return sorted(out)


def lj_block(cutoff, rc, switch_on):
    s = f"LJ:\n  cutoff: {rc}\n  algorithm: verlet_list\n  cutoff_scheme: {cutoff}\n"
    return s + (f"  switch_on: {switch_on}\n" if cutoff == "switch" else "")


def input_file(scheme, args, n, seed, out_file):
    inp = TESTSYS / "Input_files" / f"{args.density}_dens"
    sysp = SYSTEMS[args.density]
    return f"""parameter_file: {inp}/parameters.inp
start_configuration_file: {inp}/argon.xyz
output_file: {out_file}
seed: {seed}
Ntrajectories: {n}
thermalization_steps: 0
save_every: 1000000000
print_info_every: 1
simulation_type: MD

geometry:
  Lx: {sysp['L']}
  Ly: {sysp['L']}
  Lz: {sysp['L']}

particles:
  MaxNeighbors: {sysp['max_neighbors']}
  temperature: 300.0

{lj_block(args.cutoff, args.rc, args.switch_on)}
{scheme.yaml(dt=args.T / n, steps=1)}"""


def key(scheme, args, n, seed):
    return f"{scheme.id}|{args.density}|{args.cutoff}|rc{args.rc:g}|T{args.T:g}|n{n}|seed{seed}"


def run_one(scheme, args, n, seed, workdir):
    tag = re.sub(r"[^A-Za-z0-9]", "_", f"{scheme.id}_{n}_{seed}")
    yf, of = Path(workdir) / f"{tag}.yaml", Path(workdir) / f"{tag}.out"
    yf.write_text(input_file(scheme, args, n, seed, of))
    t0 = time.time()
    try:
        p = subprocess.run([str(args.exe), "-i", str(yf)], capture_output=True, text=True, timeout=args.timeout,
                           env=dict(os.environ, OMP_NUM_THREADS=str(args.threads), OMP_PROC_BIND="false"))
        stdout, code, why = p.stdout, p.returncode, (p.stdout + p.stderr)[-400:]
    except subprocess.TimeoutExpired:
        stdout, code, why = "", -1, "timeout"
    for f in (yf, of):
        f.unlink(missing_ok=True)
    de = np.array([float(m[2]) for m in PAT.finditer(stdout)])
    r = dict(scheme=scheme.id, order=scheme.order, q=scheme.q, ordering=scheme.ordering, density=args.density,
             cutoff=args.cutoff, rc=args.rc, T=args.T, n=n, seed=seed, dt=args.T / n, force_evals=scheme.q * n,
             seconds=round(time.time() - t0, 3))
    if code == 0 and len(de) == n and np.all(np.isfinite(de)):
        return dict(r, status="ok", maxabs=float(np.abs(de).max()), rms=float(np.sqrt(np.mean(de ** 2))),
                    final=float(de[-1]))
    if len(de) and not np.all(np.isfinite(de)):
        return dict(r, status="diverged", why="non-finite energy")
    return dict(r, status="failed", why=why)


def save(results, path):
    tmp = path.with_suffix(".tmp")
    tmp.write_text(json.dumps(results, indent=1))
    os.replace(tmp, path)


def main():
    p = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("schemes", nargs="+", help="scheme ids or selections")
    p.add_argument("--density", choices=["low", "high"], default="low")
    p.add_argument("--cutoff", choices=["shift", "force_shift", "switch", "none"], default="switch",
                   help="LJ cutoff_scheme: switch for order checks, shift for production (default: switch)")
    p.add_argument("--rc", type=float, default=12.0, help="LJ cutoff radius in A (default 12)")
    p.add_argument("--switch-on", type=float, help="start of the switch (default rc - 2)")
    p.add_argument("--T", type=float, default=100.0, help="simulated time in fs (default 100)")
    p.add_argument("--steps", default=DEFAULT_STEPS, help=f"numbers of steps n = T/dt (default {DEFAULT_STEPS})")
    p.add_argument("--seeds", default="123", help="random seeds for the initial momenta, e.g. 1-16 (default 123)")
    p.add_argument("--workers", type=int, default=4, help="runs in parallel (default 4)")
    p.add_argument("--threads", type=int, default=1, help="OpenMP threads per run (default 1)")
    p.add_argument("--timeout", type=float, default=3600, help="seconds per run (default 3600)")
    p.add_argument("--results", type=Path, help="JSON file (default Data/<density>_dens/Scheme_tests/results.json)")
    p.add_argument("--exe", type=Path, default=EXE)
    p.add_argument("--precision", default="long_double", choices=["long_double", "double"])
    p.add_argument("--rerun", action="store_true", help="run again even if the result exists")
    p.add_argument("--dry-run", action="store_true", help="only list what would run")
    args = p.parse_args()
    args.switch_on = args.switch_on if args.switch_on is not None else args.rc - 2.0
    args.results = args.results or TESTSYS / "Data" / f"{args.density}_dens" / "Scheme_tests" / "results.json"

    chosen = {}
    for spec in args.schemes:
        for s in S.select(spec, args.precision):
            chosen[s.id] = s
    results = json.loads(args.results.read_text()) if args.results.exists() else {}
    steps, seeds = parse_steps(args.steps), parse_steps(args.seeds)
    jobs = [(s, n, sd) for s in chosen.values() for n in steps for sd in seeds
            if args.rerun or key(s, args, n, sd) not in results]
    jobs.sort(key=lambda j: -j[0].q * j[1])            # longest first keeps the workers busy

    evals = sum(s.q * n for s, n, _ in jobs)
    print(f"{len(chosen)} schemes: {', '.join(chosen)}")
    print(f"{len(jobs)} runs to do ({len(chosen) * len(steps) * len(seeds) - len(jobs)} already in {args.results}), {len(seeds)} seeds, "
          f"{evals} force evaluations, {args.density} density, cutoff_scheme {args.cutoff}, rc {args.rc:g} A, T {args.T:g} fs")
    if args.dry_run or not jobs:
        return
    if not args.exe.exists():
        sys.exit(f"executable not found: {args.exe}")
    args.results.parent.mkdir(parents=True, exist_ok=True)

    t0, done, bad = time.time(), 0, 0
    with tempfile.TemporaryDirectory(prefix="scheme_tests_") as work:
        ex = ThreadPoolExecutor(args.workers)
        futures = {ex.submit(run_one, s, args, n, sd, work): (s, n, sd) for s, n, sd in jobs}
        try:
            for f in as_completed(futures):
                s, n, sd = futures[f]
                r = f.result()
                results[key(s, args, n, sd)] = r
                done += 1
                if r["status"] != "ok":
                    bad += 1
                    print(f"  {s.id} n={n} seed={sd}: {r['status']} ({r['why'][-120:].strip()})", flush=True)
                if done % 10 == 0 or done == len(jobs):
                    save(results, args.results)
                    print(f"[{done}/{len(jobs)}] {time.time() - t0:.0f} s", flush=True)
        except KeyboardInterrupt:
            ex.shutdown(wait=False, cancel_futures=True)
            save(results, args.results)
            sys.exit(f"\ninterrupted: {done} finished runs saved to {args.results}; run the same command to continue")
        ex.shutdown()
    print(f"done: {done - bad} ok, {bad} diverged or failed -> {args.results}")


if __name__ == "__main__":
    main()
