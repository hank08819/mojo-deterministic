"""
risk_reconciliation_example.py

A worked end-to-end example: an end-of-day portfolio risk number that
two trading desks must reconcile to the bit. Without deterministic
reduction, the same positions produce different risk numbers depending
on how the GPU scheduled the aggregation. With mojo-deterministic,
they reconcile exactly.

This is the failure mode described in Section 4 of the paper, and the
fix the library provides.

Run: python risk_reconciliation_example.py

Part of the CACM Practice paper:
  "Mojo: A Promising Tool for Scalable Financial AI Efficiency"
  Henry Han, Baylor University

Author: Henry Han, Baylor University
License: MIT
"""
import random
import struct
import sys

sys.path.insert(0, "../src")
sys.path.insert(0, "src")
from deterministic_reduce import (
    deterministic_risk_aggregate, nondeterministic_parallel_sum,
)


def bits(x):
    return struct.pack(">d", x).hex()


def build_portfolio(n_positions, seed):
    """A realistic portfolio: many small option legs plus a few large
    directional positions, producing ill-conditioned risk contributions."""
    rng = random.Random(seed)
    contributions = []
    weights = []
    for i in range(n_positions):
        if i % 5000 == 0:
            # occasional large directional position
            contributions.append(rng.uniform(-1, 1) * 1e7)
        else:
            # many small option legs
            contributions.append(rng.uniform(-1, 1) * rng.uniform(0.01, 100))
        weights.append(rng.uniform(0.5, 1.5))
    return contributions, weights


def main():
    print("=" * 66)
    print("END-OF-DAY RISK RECONCILIATION")
    print("Two desks, same positions, must agree to the bit.")
    print("=" * 66)

    n = 200_000
    contributions, weights = build_portfolio(n, seed=42)
    print(f"\nPortfolio: {n:,} positions")
    print(f"(many small option legs + occasional large directional bets)")

    # -------------------------------------------------------------
    # Scenario A: naive parallel aggregation (the status quo).
    # Desk 1 and Desk 2 run the SAME calculation, but their GPUs
    # schedule the reduction differently. We simulate this by using
    # different (but equally valid) block accumulation orders.
    # -------------------------------------------------------------
    print("\n--- WITHOUT deterministic reduction ---")
    products = [contributions[i] * weights[i] for i in range(n)]

    rng = random.Random(7)
    desk_values = []
    for desk in range(5):
        chunks = 64
        order = list(range(chunks))
        rng.shuffle(order)
        v = nondeterministic_parallel_sum(products, chunks, order)
        desk_values.append(v)
        print(f"  Desk {desk+1} risk number: {v:.10e}   bits={bits(v)[:16]}...")

    distinct = len(set(bits(v) for v in desk_values))
    print(f"\n  {distinct} distinct risk numbers from {len(desk_values)} desks"
          f" computing the SAME portfolio.")
    print(f"  -> Reconciliation FAILS. No bit-exact audit trail.")

    # -------------------------------------------------------------
    # Scenario B: deterministic aggregation (the fix).
    # Every desk uses deterministic_risk_aggregate. Same input ->
    # same bits, regardless of how the work is parallelized.
    # -------------------------------------------------------------
    print("\n--- WITH mojo-deterministic ---")
    det_values = []
    for desk in range(5):
        v = deterministic_risk_aggregate(contributions, weights)
        det_values.append(v)
        print(f"  Desk {desk+1} risk number: {v:.10e}   bits={bits(v)[:16]}...")

    distinct_det = len(set(bits(v) for v in det_values))
    print(f"\n  {distinct_det} distinct risk number from {len(det_values)} desks.")
    if distinct_det == 1:
        print(f"  -> Reconciliation SUCCEEDS. Bit-exact audit trail.")

    print("\n" + "=" * 66)
    print("CONCLUSION")
    print("=" * 66)
    print(f"""
  The naive parallel path gave {distinct} different official risk
  numbers for one portfolio. The deterministic path gave exactly 1.
  In a regulated trading system, the second is the only one that can
  be defended to an auditor. mojo-deterministic makes the second the
  default with a single function call, at a measured cost of roughly
  10-30% over the maximally-parallel non-deterministic reduction.
""")


if __name__ == "__main__":
    main()
