# =============================================================
# example_usage.mojo
#
# Demonstrates the mojo-deterministic library on a portfolio
# risk aggregation, the pattern from Section 4 of the paper.
#
# Run: pixi run mojo run examples/example_usage.mojo
#   (adjust the import path to point at ../src if needed)
#
# Author: Henry Han, Baylor University
# License: MIT
# =============================================================

from collections import List
from random import random_float64, seed

# Import the library. In a real project, add src/ to your Mojo
# import path or install the package; here we assume the file is
# compiled alongside src/deterministic_reduce.mojo.
from deterministic_reduce import (
    tree_sum,
    kahan_sum,
    pairwise_kahan_sum,
    deterministic_dot,
    deterministic_risk_aggregate,
)


def main():
    seed(42)
    var n = 200_000

    # Build an ill-conditioned portfolio: many small option legs plus
    # a few large directional positions.
    var contributions = List[Float64]()
    var weights = List[Float64]()
    for i in range(n):
        if i % 5000 == 0:
            contributions.append((random_float64() - 0.5) * 2.0e7)
        else:
            contributions.append((random_float64() - 0.5) * 200.0)
        weights.append(0.5 + random_float64())

    # Bit-reproducible risk aggregation: one function call.
    var risk = deterministic_risk_aggregate(contributions, weights)
    print("Deterministic portfolio risk:", risk)

    # The three reduction strategies, for direct comparison:
    var products = List[Float64]()
    for i in range(n):
        products.append(contributions[i] * weights[i])

    print("tree_sum:          ", tree_sum(products))
    print("kahan_sum:         ", kahan_sum(products))
    print("pairwise_kahan_sum:", pairwise_kahan_sum(products))

    # Every one of these is bit-identical across runs, thread counts,
    # and (on the same FP architecture) machines. That is the property
    # finance audit requires.
    print()
    print("Re-running deterministic_risk_aggregate gives identical bits:")
    var risk2 = deterministic_risk_aggregate(contributions, weights)
    print("  run 1:", risk)
    print("  run 2:", risk2)
    print("  identical:", risk == risk2)
