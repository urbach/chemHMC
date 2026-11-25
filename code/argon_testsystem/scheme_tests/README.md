# Testing splitting schemes on the argon systems

Three scripts. Run them from this folder after building `code/builds/openMP`.

| script | does |
|---|---|
| `schemes.py` | reads `../Data/Schemes`, lists and ranks schemes, prints the `integrator:` block for an input file |
| `run_schemes.py` | runs energy-error sweeps at a fixed simulated time T and stores the results in JSON |
| `plot_schemes.py` | plots max\|ΔE\| against the time step, and the RMS of ΔE against the cost |

## Scheme names

`n<order>q<cycles><AB|BA>#<row>`, for example `n6q7AB#0`. The row is the row in the files.
`q` is the number of force evaluations per step.
In the files, A is the kinetic part (drift, `a`) and B is the potential part (kick, `b`).
AB schemes run with `GENERAL_AB` (drift first) and BA schemes with `GENERAL_BA` (kick first).
Velocity Verlet is `n2q1BA#0`.

The coefficients are read from `long_double` and rounded to double (`--precision double` uses the other folder).

## Workflow

```bash
cd code/argon_testsystem/scheme_tests

# 1. what is there, and the most efficient schemes (efficiency at the scheme's own order, from the files)
python3 schemes.py list
python3 schemes.py list --order 6 --top 10
python3 schemes.py list --order 6 --q 9 --ordering BA

# 2. order check: smooth cutoff, so every scheme shows its true order
python3 run_schemes.py 'n2q1BA#0' n4:top3 n6:top4 --cutoff switch --dry-run   # how much work
python3 run_schemes.py 'n2q1BA#0' n4:top3 n6:top4 --cutoff switch
python3 plot_schemes.py --cutoff switch --table

# 3. production potential, dense system
python3 run_schemes.py 'n2q1BA#0' n4:top3 n6:top4 --cutoff shift --density high
python3 plot_schemes.py --cutoff shift --density high

# 4. one scheme in your own input file
python3 schemes.py yaml 'n6q7AB#0' --dt 2.0
```

Selections: an id (`n6q7AB#0`), or the most efficient schemes of a family:
`n6q7AB:top3`, `n6q7:top3` (both orderings), `n6:top5` (all q), `n6BA:top5`.
Put ids with `#` in quotes.

## run_schemes.py

- Every run uses the same initial configuration and integrates to `--T` (default 100 fs) with
  n = T/dt steps, n from `--steps` (default `1-20,25,40,50,80,100,200,400,500,1000`).
- `--density low|high`, `--cutoff switch|shift|force_shift|none` (default `switch`), `--rc` (default 12 Å).
- Results go to `../Data/<density>_dens/Scheme_tests/results.json` (or `--results`).
  Each run is stored with all its settings, so one file can hold several cutoffs, cutoff radii and T.
  Runs already in the file are skipped: you can stop a sweep with Ctrl-C (it saves every 10 runs)
  and restart it, or add schemes and step counts later. `--rerun` forces new runs.
- `--workers` runs in parallel (default 4), each with `--threads` OpenMP threads (default 1).
- Cost: the low-density box needs about 0.2 ms per force evaluation, the high-density box about 4 ms.
  `--dry-run` prints the number of force evaluations of a sweep.
- Runs that stop or produce non-finite energies are stored as `failed` or `diverged`.

## plot_schemes.py

- Choose results with `--density`, `--cutoff`, `--rc`, `--T`; list scheme ids to plot a subset (max 8).
- The local slope in the legend is fitted to the points with 1e-13 ≤ max\|ΔE\| ≤ 1e-8 (`--fit`).
  Schemes with very small leading error constants can show a larger local slope than their order
  in that range, because the next-order terms still dominate.
- Below about 1e-15 the error is round-off.
- Runs with \|ΔE\| > 0.1 are drawn as × at the top (`--diverged`); the dotted line is 1 kT for the whole box.
- `--table` prints max\|ΔE\| for every scheme and step count.
