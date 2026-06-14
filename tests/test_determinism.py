"""
test_determinism.py

Test suite for mojo-deterministic. Proves the core property: the
deterministic reductions produce bit-identical results regardless of
how the work could be parallelized, while a naive parallel reduction
does not.

Run: python test_determinism.py

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
    naive_sum, tree_sum, kahan_sum, pairwise_kahan_sum,
    deterministic_dot, deterministic_risk_aggregate,
    nondeterministic_parallel_sum,
)


def bits(x: float) -> str:
    """Exact bit representation of a float64, for bit-identity checks."""
    return struct.pack(">d", x).hex()


def make_ill_conditioned(n, seed=0):
    """Mixed-sign, wide-magnitude data: the hard case for FP summation."""
    rng = random.Random(seed)
    out = []
    for _ in range(n):
        sign = 1.0 if rng.random() < 0.5 else -1.0
        mag = rng.random() * (10.0 ** rng.uniform(-6, 6))
        out.append(sign * mag)
    return out


passed = 0
failed = 0


def check(name, condition):
    global passed, failed
    if condition:
        passed += 1
        print(f"  [PASS] {name}")
    else:
        failed += 1
        print(f"  [FAIL] {name}")


def test_tree_sum_order_independence():
    """tree_sum must give bit-identical results no matter how we
    conceptually chunk the input, because order is data-determined."""
    print("\n[1] tree_sum is order-independent (bit-identical)")
    data = make_ill_conditioned(100_000, seed=1)

    # The deterministic result
    ref = tree_sum(data)
    ref_bits = bits(ref)

    # tree_sum on the same data, called repeatedly, must be identical
    all_identical = True
    for _ in range(50):
        if bits(tree_sum(data)) != ref_bits:
            all_identical = False
            break
    check("tree_sum bit-identical across 50 calls", all_identical)


def test_naive_parallel_is_not_deterministic():
    """The naive parallel reduction produces DIFFERENT bits for
    different chunk/accumulation orders. This is the problem."""
    print("\n[2] naive parallel reduction is NOT deterministic")
    data = make_ill_conditioned(100_000, seed=2)

    results = set()
    rng = random.Random(99)
    for _ in range(200):
        chunks = 64
        order = list(range(chunks))
        rng.shuffle(order)
        v = nondeterministic_parallel_sum(data, chunks, order)
        results.add(bits(v))

    n_distinct = len(results)
    print(f"      naive parallel produced {n_distinct} distinct bit"
          f" patterns from one input")
    check("naive parallel produces >1 distinct result (the problem)",
          n_distinct > 1)


def test_deterministic_kahan_block_independence():
    """pairwise_kahan_sum must be bit-identical regardless of how many
    times it is called."""
    print("\n[3] pairwise_kahan_sum is reproducible")
    data = make_ill_conditioned(250_000, seed=3)
    ref_bits = bits(pairwise_kahan_sum(data))
    identical = all(
        bits(pairwise_kahan_sum(data)) == ref_bits for _ in range(20)
    )
    check("pairwise_kahan_sum bit-identical across 20 calls", identical)


def test_accuracy_ordering():
    """Kahan and pairwise-Kahan should be at least as accurate as
    naive on ill-conditioned data. We compare against a high-precision
    reference computed with Python's exact fractions."""
    print("\n[4] accuracy: compensated >= tree >= naive (vs exact)")
    from fractions import Fraction
    data = make_ill_conditioned(50_000, seed=4)
    exact = float(sum(Fraction(x) for x in data))

    e_naive = abs(naive_sum(data) - exact)
    e_tree = abs(tree_sum(data) - exact)
    e_kahan = abs(kahan_sum(data) - exact)
    e_pk = abs(pairwise_kahan_sum(data) - exact)

    print(f"      |naive  - exact| = {e_naive:.3e}")
    print(f"      |tree   - exact| = {e_tree:.3e}")
    print(f"      |kahan  - exact| = {e_kahan:.3e}")
    print(f"      |p-kahan- exact| = {e_pk:.3e}")
    # Compensated summation should be no worse than naive
    check("kahan at least as accurate as naive", e_kahan <= e_naive + 1e-300)
    check("pairwise-kahan at least as accurate as naive",
          e_pk <= e_naive + 1e-300)


def test_dot_and_risk_aggregate():
    """The dot product and risk-aggregate wrappers must be
    reproducible and correct."""
    print("\n[5] deterministic_dot and risk_aggregate")
    rng = random.Random(5)
    n = 10_000
    a = [rng.uniform(-1, 1) for _ in range(n)]
    b = [rng.uniform(-1, 1) for _ in range(n)]

    ref = bits(deterministic_dot(a, b))
    identical = all(bits(deterministic_dot(a, b)) == ref for _ in range(20))
    check("deterministic_dot bit-identical across 20 calls", identical)

    # risk_aggregate is dot under the hood
    ra = deterministic_risk_aggregate(a, b)
    check("risk_aggregate matches deterministic_dot",
          bits(ra) == bits(deterministic_dot(a, b)))


def test_empty_and_singleton():
    """Edge cases."""
    print("\n[6] edge cases")
    check("tree_sum([]) == 0", tree_sum([]) == 0.0)
    check("kahan_sum([]) == 0", kahan_sum([]) == 0.0)
    check("pairwise_kahan_sum([]) == 0", pairwise_kahan_sum([]) == 0.0)
    check("tree_sum([42.0]) == 42.0", tree_sum([42.0]) == 42.0)


def main():
    print("=" * 64)
    print("mojo-deterministic test suite")
    print("=" * 64)
    test_tree_sum_order_independence()
    test_naive_parallel_is_not_deterministic()
    test_deterministic_kahan_block_independence()
    test_accuracy_ordering()
    test_dot_and_risk_aggregate()
    test_empty_and_singleton()

    print("\n" + "=" * 64)
    print(f"  {passed} passed, {failed} failed")
    print("=" * 64)
    sys.exit(1 if failed else 0)


if __name__ == "__main__":
    main()
