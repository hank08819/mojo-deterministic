"""
deterministic_reduce.py

Pure-Python reference implementation of the mojo-deterministic library.
Mirrors src/deterministic_reduce.mojo exactly, so that:
  (1) anyone without a Mojo install can verify the algorithms
  (2) the test suite can check Mojo and Python produce identical bits
  (3) the determinism property can be demonstrated portably

Every function here has a one-to-one correspondent in the Mojo source.

Part of the CACM Practice paper:
  "Mojo: A Promising Tool for Scalable Financial AI Efficiency"
  Henry Han, Baylor University

Author: Henry Han, Baylor University
License: MIT
"""
from typing import List


def naive_sum(data: List[float]) -> float:
    """Left-to-right scalar accumulation. Deterministic single-thread,
    poor accuracy, not parallelizable without losing determinism."""
    s = 0.0
    for x in data:
        s += x
    return s


def tree_sum(data: List[float]) -> float:
    """Deterministic pairwise tree reduction. Bit-identical across
    thread/chunk counts because the order is data-determined."""
    n = len(data)
    if n == 0:
        return 0.0
    buf = list(data)
    width = n
    while width > 1:
        half = width // 2
        for i in range(half):
            buf[i] = buf[2 * i] + buf[2 * i + 1]
        if width % 2 == 1:
            buf[half] = buf[width - 1]
            width = half + 1
        else:
            width = half
    return buf[0]


def kahan_sum(data: List[float]) -> float:
    """Kahan-Babuska compensated summation. O(eps) total error."""
    s = 0.0
    c = 0.0
    for x in data:
        y = x - c
        t = s + y
        c = (t - s) - y
        s = t
    return s


def pairwise_kahan_sum(data: List[float], block: int = 1024) -> float:
    """Block-Kahan + deterministic tree over block totals."""
    n = len(data)
    if n == 0:
        return 0.0
    block_totals = []
    i = 0
    while i < n:
        hi = min(i + block, n)
        s = 0.0
        c = 0.0
        for j in range(i, hi):
            y = data[j] - c
            t = s + y
            c = (t - s) - y
            s = t
        block_totals.append(s)
        i = hi
    return tree_sum(block_totals)


def deterministic_dot(a: List[float], b: List[float]) -> float:
    """Bit-reproducible dot product."""
    products = [a[i] * b[i] for i in range(len(a))]
    return pairwise_kahan_sum(products)


def deterministic_risk_aggregate(
    contributions: List[float], weights: List[float]
) -> float:
    """Bit-reproducible portfolio risk aggregation."""
    return deterministic_dot(contributions, weights)


# -------------------------------------------------------------
# Non-deterministic reference, for contrast in tests/benchmarks.
# This is what a naive parallel reduction does; DO NOT use in
# production where reproducibility is required.
# -------------------------------------------------------------
def nondeterministic_parallel_sum(data, chunks, order):
    """Split into `chunks` blocks, sum each, accumulate block totals
    in the given `order`. Different orders -> different bits. Used in
    tests to demonstrate the problem the library solves."""
    n = len(data)
    per = n // chunks
    partials = []
    for c in range(chunks):
        lo = c * per
        hi = lo + per if c < chunks - 1 else n
        s = 0.0
        for i in range(lo, hi):
            s += data[i]
        partials.append(s)
    total = 0.0
    for idx in order:
        total += partials[idx]
    return total
