# =============================================================
# deterministic_reduce.mojo
#
# mojo-deterministic: bit-reproducible reduction kernels for
# financial AI.
#
# The problem this library solves:
#   IEEE 754 floating-point addition is not associative, so
#   (a + b) + c != a + (b + c) in general. When a sum is computed
#   by a parallel reduction whose accumulation order depends on
#   thread or kernel scheduling (as on GPUs, and on multicore
#   reductions generally), the same input can produce different
#   bit patterns on different runs or different hardware. In a
#   regulated trading system this breaks bit-exact audit
#   reproducibility.
#
# The fix this library provides:
#   reduction kernels whose accumulation order is determined
#   ENTIRELY by the data layout, not by the runtime schedule.
#   Given the same input, they produce the same bits regardless of
#   thread count, chunk count, or device, on a given floating-point
#   architecture. Three strategies are provided, trading speed for
#   accuracy:
#     - tree_sum:       pairwise tree reduction, O(log n) depth,
#                       deterministic, better rounding than naive.
#     - kahan_sum:      compensated summation, O(eps) total error,
#                       deterministic, sequential.
#     - pairwise_kahan: tree reduction with Kahan accumulation in
#                       each node; deterministic, best accuracy.
#
# License: MIT
# Author:  Henry Han, Baylor University
# Part of the CACM Practice paper:
#   "Mojo: A Promising Tool for Scalable Financial AI Efficiency"
# =============================================================

from collections import List


# -------------------------------------------------------------
# Naive sequential sum (deterministic, but poor accuracy and slow).
# Provided as a reference / baseline only.
# -------------------------------------------------------------
def naive_sum(data: List[Float64]) -> Float64:
    """Left-to-right scalar accumulation. Deterministic on a single
    thread, but O(n*eps) worst-case error and not parallelizable
    without losing determinism."""
    var s: Float64 = 0.0
    for i in range(len(data)):
        s += data[i]
    return s


# -------------------------------------------------------------
# Deterministic pairwise tree reduction.
#
# The topology is fixed by index structure: element 2i and 2i+1
# always combine, regardless of how many threads execute the layer.
# This is the key property: the reduction order does not depend on
# the runtime schedule, so the result is bit-identical across thread
# counts and devices (on the same FP architecture).
# -------------------------------------------------------------
def tree_sum(data: List[Float64]) -> Float64:
    """Deterministic pairwise tree reduction.

    Bit-identical across thread counts and chunkings because the
    accumulation order is data-determined. Also has O(log n) error
    growth versus O(n) for naive summation (Higham, Accuracy and
    Stability of Numerical Algorithms, ch. 4)."""
    var n = len(data)
    if n == 0:
        return 0.0
    # Work on a mutable copy
    var buf = List[Float64]()
    for i in range(n):
        buf.append(data[i])

    var width = n
    while width > 1:
        var half = width // 2
        for i in range(half):
            buf[i] = buf[2 * i] + buf[2 * i + 1]
        if width % 2 == 1:
            # Carry the odd tail element forward unchanged
            buf[half] = buf[width - 1]
            width = half + 1
        else:
            width = half
    return buf[0]


# -------------------------------------------------------------
# Kahan compensated summation.
#
# Deterministic and sequential, with O(eps) total error independent
# of n. Use when accuracy matters more than parallelism.
# -------------------------------------------------------------
def kahan_sum(data: List[Float64]) -> Float64:
    """Kahan-Babuska compensated summation. Deterministic, O(eps)
    total rounding error regardless of n."""
    var s: Float64 = 0.0
    var c: Float64 = 0.0   # running compensation for lost low-order bits
    for i in range(len(data)):
        var y = data[i] - c
        var t = s + y
        c = (t - s) - y
        s = t
    return s


# -------------------------------------------------------------
# Pairwise tree reduction with Kahan accumulation in each node.
#
# Combines the order-independence of the tree with the accuracy of
# compensated summation. Deterministic, best accuracy of the three.
# Implemented as block-wise Kahan over fixed-size blocks, then a
# tree over the block totals.
# -------------------------------------------------------------
def pairwise_kahan_sum(data: List[Float64], block: Int = 1024) -> Float64:
    """Deterministic block-Kahan + tree reduction.

    The vector is split into fixed-size blocks; each block is summed
    with Kahan compensation; the block totals are combined with a
    deterministic tree. Block size is a fixed parameter, not a
    runtime schedule choice, so the result is reproducible."""
    var n = len(data)
    if n == 0:
        return 0.0

    var block_totals = List[Float64]()
    var i = 0
    while i < n:
        var hi = i + block
        if hi > n:
            hi = n
        # Kahan sum over this block
        var s: Float64 = 0.0
        var c: Float64 = 0.0
        for j in range(i, hi):
            var y = data[j] - c
            var t = s + y
            c = (t - s) - y
            s = t
        block_totals.append(s)
        i = hi

    # Deterministic tree over the block totals
    return tree_sum(block_totals)


# -------------------------------------------------------------
# Dot product with deterministic reduction.
#
# The element-wise products are formed in index order (deterministic
# by construction), then summed with the chosen deterministic
# reduction. This is the building block for deterministic matmul,
# covariance, and risk-aggregation kernels.
# -------------------------------------------------------------
def deterministic_dot(a: List[Float64], b: List[Float64]) -> Float64:
    """Bit-reproducible dot product. Products formed in index order,
    summed with the pairwise-Kahan reduction."""
    var n = len(a)
    var products = List[Float64]()
    for i in range(n):
        products.append(a[i] * b[i])
    return pairwise_kahan_sum(products)


# -------------------------------------------------------------
# Deterministic weighted sum (risk aggregation primitive).
#
# Aggregates contributions[i] * weights[i] reproducibly. This is the
# exact pattern that produced 8 distinct values out of 200 parallel
# schedules in the paper's case study; here it produces one.
# -------------------------------------------------------------
def deterministic_risk_aggregate(
    contributions: List[Float64], weights: List[Float64]
) -> Float64:
    """Bit-reproducible portfolio risk aggregation:
    sum_i contributions[i] * weights[i], computed in a fixed order."""
    return deterministic_dot(contributions, weights)
