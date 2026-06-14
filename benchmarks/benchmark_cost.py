"""
benchmark_cost.py

Measures the performance cost of determinism: how much slower are the
deterministic reductions versus a naive sum, and versus each other.
The paper claims the cost is "10-30%" for a tree reduction over the
maximally-parallel non-deterministic version; this benchmark lets you
measure it on your own machine.

Run: python benchmark_cost.py

Part of the CACM Practice paper:
  "Mojo: A Promising Tool for Scalable Financial AI Efficiency"
  Henry Han, Baylor University

Author: Henry Han, Baylor University
License: MIT
"""
import random
import time
import sys

sys.path.insert(0, "../src")
sys.path.insert(0, "src")
from deterministic_reduce import (
    naive_sum, tree_sum, kahan_sum, pairwise_kahan_sum,
)


def timeit(fn, data, repeats=5):
    best = None
    for _ in range(repeats):
        t0 = time.perf_counter_ns()
        fn(data)
        t1 = time.perf_counter_ns()
        e = t1 - t0
        if best is None or e < best:
            best = e
    return best


def main():
    print("=" * 60)
    print("Cost of determinism (best of 5 runs)")
    print("=" * 60)

    rng = random.Random(0)
    for n in [100_000, 1_000_000]:
        data = [rng.uniform(-1, 1) * rng.uniform(0.01, 100)
                for _ in range(n)]
        print(f"\nn = {n:,}")

        t_naive = timeit(naive_sum, data)
        t_tree = timeit(tree_sum, data)
        t_kahan = timeit(kahan_sum, data)
        t_pk = timeit(pairwise_kahan_sum, data)

        def ms(ns):
            return ns / 1e6

        def rel(t):
            return t / t_naive

        print(f"  naive_sum          {ms(t_naive):8.2f} ms   1.00x  "
              f"(baseline, NOT parallel-safe)")
        print(f"  tree_sum           {ms(t_tree):8.2f} ms   {rel(t_tree):.2f}x  "
              f"(deterministic)")
        print(f"  kahan_sum          {ms(t_kahan):8.2f} ms   {rel(t_kahan):.2f}x  "
              f"(deterministic, O(eps) error)")
        print(f"  pairwise_kahan_sum {ms(t_pk):8.2f} ms   {rel(t_pk):.2f}x  "
              f"(deterministic, best accuracy)")

    print("\nNote: this is a pure-Python reference timing. The Mojo")
    print("implementation (src/deterministic_reduce.mojo) runs the same")
    print("algorithms at native speed; the RELATIVE cost of determinism")
    print("is the figure of interest, and it carries over to Mojo.")


if __name__ == "__main__":
    main()
