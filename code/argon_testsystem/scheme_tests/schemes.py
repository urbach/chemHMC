"""Read the splitting schemes in argon_testsystem/Data/Schemes and turn them into chemHMC input.

Convention (the same as the GENERAL_AB / GENERAL_BA integrators):
  A = kinetic part   -> drift (position update), coefficients a
  B = potential part -> kick (momentum update),  coefficients b
  AB: drift first, len(a) = q+1, len(b) = q   -> integrator GENERAL_AB
  BA: kick first,  len(a) = q,   len(b) = q+1 -> integrator GENERAL_BA
One step costs q force evaluations. The files store the first half of the symmetric a and b.

A scheme is named n<order>q<cycles><AB|BA>#<row>, e.g. n6q7AB#0, where <row> is the row in the files.
Leapfrog/velocity Verlet is n2q1BA#0.

Usage:
  python3 schemes.py list                                  # what is available
  python3 schemes.py list --order 6 --q 7 --top 5          # best schemes by efficiency
  python3 schemes.py yaml n6q7AB#0 --dt 2.0 --steps 1      # integrator block for an input file
"""
import argparse
import re
from dataclasses import dataclass
from pathlib import Path

import numpy as np

DATA = Path(__file__).resolve().parent.parent / "Data" / "Schemes"
ID_RE = re.compile(r"^n(\d+)q(\d+)(AB|BA)#(\d+)$")


@dataclass
class Scheme:
    id: str
    order: int
    q: int
    ordering: str
    row: int
    a: list          # drift coefficients
    b: list          # kick coefficients
    error: float     # Error^2 of the leading (order+1) terms, from the files
    efficiency: float

    @property
    def integrator(self):
        return f"GENERAL_{self.ordering}"

    def yaml(self, dt=1.0, steps=1):
        fmt = lambda v: "[" + ", ".join(repr(x) for x in v) + "]"
        return (f"integrator:\n  name: {self.integrator}\n  dt: {dt!r}\n  steps: {steps}\n"
                f"  cycles: {self.q}\n  a: {fmt(self.a)}\n  b: {fmt(self.b)}\n")


def _dir(order, q, ordering, precision):
    return DATA / precision / f"order_n{order}" / f"cycles_q{q}" / ordering


def _expand(half, n):
    """First ceil(n/2) coefficients of a symmetric sequence -> all n (rounded to double)."""
    h = [float(x) for x in half]
    if len(h) != (n + 1) // 2:
        raise ValueError(f"expected {(n + 1) // 2} stored coefficients, found {len(h)}")
    return h + h[:n - len(h)][::-1]


def families(precision="long_double"):
    """All (order, q, ordering, number of schemes) that have data."""
    out = []
    for f in sorted((DATA / precision).glob("order_n*/cycles_q*/*/standard_parameters_a_q*.npy")):
        ordering, qd, od = f.parent.name, f.parent.parent.name, f.parent.parent.parent.name
        out.append((int(od[7:]), int(qd[8:]), ordering, len(np.load(f, mmap_mode="r"))))
    return sorted(out)


def load_family(order, q, ordering, precision="long_double"):
    d = _dir(order, q, ordering, precision)
    A = np.load(d / f"standard_parameters_a_q{q}.npy")
    B = np.load(d / f"standard_parameters_b_q{q}.npy")
    col = order // 2 - 1     # columns are the error terms of order 2, 4, 6, 8
    err = np.load(d / f"errors_per_order_q{q}.npy")[:, col]
    eff = np.load(d / f"efficiencies_per_order_q{q}.npy")[:, col]
    na, nb = (q + 1, q) if ordering == "AB" else (q, q + 1)
    return [Scheme(f"n{order}q{q}{ordering}#{i}", order, q, ordering, i, _expand(A[i], na), _expand(B[i], nb),
                   float(err[i]), float(eff[i])) for i in range(len(A))]


def get(scheme_id, precision="long_double"):
    m = ID_RE.match(scheme_id)
    if not m:
        raise ValueError(f"scheme id must look like n6q7AB#0, got '{scheme_id}'")
    order, q, ordering, row = int(m[1]), int(m[2]), m[3], int(m[4])
    fam = load_family(order, q, ordering, precision)
    if row >= len(fam):
        raise ValueError(f"{scheme_id}: n{order}q{q}{ordering} has only {len(fam)} schemes")
    return fam[row]


def best(order=None, q=None, ordering=None, top=5, precision="long_double"):
    """Most efficient schemes (efficiency at their own order) among the matching families."""
    out = []
    for o, qq, ab, _ in families(precision):
        if (order is None or o == order) and (q is None or qq == q) and (ordering is None or ab == ordering):
            out += load_family(o, qq, ab, precision)
    out.sort(key=lambda s: -s.efficiency)
    return out[:top] if top else out


def select(spec, precision="long_double"):
    """'n6q7AB#0' -> that scheme; 'n6q7AB:top3', 'n6q7:top3', 'n6:top5', 'n6BA:top2' -> best schemes."""
    if "#" in spec:
        return [get(spec, precision)]
    m = re.match(r"^n(\d+)(?:q(\d+))?(AB|BA)?:top(\d+)$", spec)
    if not m:
        raise ValueError(f"cannot read scheme selection '{spec}' (examples: n6q7AB#0, n6q7:top3, n6:top5)")
    return best(int(m[1]), int(m[2]) if m[2] else None, m[3], int(m[4]), precision)


def main():
    p = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    sub = p.add_subparsers(dest="cmd", required=True)
    pl = sub.add_parser("list", help="list families, or the best schemes when --order/--q/--ordering is given")
    pl.add_argument("--order", type=int); pl.add_argument("--q", type=int)
    pl.add_argument("--ordering", choices=["AB", "BA"]); pl.add_argument("--top", type=int, default=10)
    py = sub.add_parser("yaml", help="print the integrator block for one scheme")
    py.add_argument("scheme"); py.add_argument("--dt", type=float, default=1.0); py.add_argument("--steps", type=int, default=1)
    for s in (pl, py):
        s.add_argument("--precision", default="long_double", choices=["long_double", "double"])
    args = p.parse_args()

    if args.cmd == "yaml":
        print(get(args.scheme, args.precision).yaml(args.dt, args.steps), end="")
    elif args.order is None and args.q is None and args.ordering is None:
        print(f"{'order':>5s} {'q':>3s} {'ordering':>8s} {'schemes':>8s}")
        for o, q, ab, n in families(args.precision):
            print(f"{o:5d} {q:3d} {ab:>8s} {n:8d}")
    else:
        print(f"{'scheme':16s} {'integrator':11s} {'Error^2':>10s} {'efficiency':>11s}")
        for s in best(args.order, args.q, args.ordering, args.top, args.precision):
            print(f"{s.id:16s} {s.integrator:11s} {s.error:10.3e} {s.efficiency:11.4g}")


if __name__ == "__main__":
    main()
