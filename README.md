# mojo-deterministic

**Bit-reproducible reduction kernels for financial AI.**

![License](https://img.shields.io/badge/license-MIT-green)
![Mojo](https://img.shields.io/badge/Mojo-1.0.0%20beta1-orange)
![Tests](https://img.shields.io/badge/tests-11%2F11%20passing-brightgreen)

> The same risk number, computed two ways, must agree to the bit.
> This library makes that happen.

IEEE 754 floating-point addition is **not associative**:
`(a + b) + c != a + (b + c)`. When a sum is computed by a parallel
reduction whose accumulation order depends on thread or kernel
scheduling — as it does on every GPU, and on multicore reductions
generally — the same input can produce **different bit patterns** on
different runs or different hardware. In a regulated trading system,
that breaks bit-exact audit reproducibility.

`mojo-deterministic` provides reduction kernels whose accumulation
order is fixed by the **data layout**, not the runtime schedule. Given
the same input, they produce the same bits regardless of thread count,
chunk count, or device.

This is the companion library to the paper
*"Mojo: A Promising Tool for Scalable Financial AI Efficiency"*
(Henry Han, Baylor University).

---

## See the problem, and the fix, in one command

```bash
python examples/risk_reconciliation_example.py
```

```
--- WITHOUT deterministic reduction ---
  Desk 1 risk: -1.5181350283e+07   bits=...c91264b6
  Desk 2 risk: -1.5181350283e+07   bits=...c91264b7
  Desk 3 risk: -1.5181350283e+07   bits=...c91264b3
  Desk 4 risk: -1.5181350283e+07   bits=...c91264b2
  Desk 5 risk: -1.5181350283e+07   bits=...c91264b4
  5 distinct risk numbers from 5 desks computing the SAME portfolio.
  -> Reconciliation FAILS.

--- WITH mojo-deterministic ---
  Desk 1..5 risk: -1.5181350283e+07   bits=...c91264ee  (all identical)
  -> Reconciliation SUCCEEDS. Bit-exact audit trail.
```

Five desks computing the same portfolio with a naive parallel
reduction get five different official risk numbers. With this library
they get one.

---

## Quick start

```bash
git clone https://github.com/hank08819/mojo-deterministic.git
cd mojo-deterministic

# Verify the determinism property (pure Python, no Mojo needed)
python tests/test_determinism.py

# See the finance worked example
python examples/risk_reconciliation_example.py

# Measure the cost of determinism on your machine
python benchmarks/benchmark_cost.py
```

To run the Mojo versions (requires `pixi add modular`):

```bash
pixi run mojo run examples/example_usage.mojo
```

---

## API

The library ships as Mojo (`src/deterministic_reduce.mojo`) with an
exact Python mirror (`src/deterministic_reduce.py`) for verification.

| Function | Determinism | Accuracy | Notes |
|---|---|---|---|
| `tree_sum(data)` | bit-exact | O(log n) error | pairwise tree, fixed topology |
| `kahan_sum(data)` | bit-exact | O(eps) error | compensated, sequential |
| `pairwise_kahan_sum(data, block)` | bit-exact | best | block-Kahan + tree |
| `deterministic_dot(a, b)` | bit-exact | best | products in index order |
| `deterministic_risk_aggregate(contribs, weights)` | bit-exact | best | risk-aggregation primitive |
| `naive_sum(data)` | single-thread only | O(n) error | baseline / reference |

```python
from deterministic_reduce import deterministic_risk_aggregate

# Bit-reproducible portfolio risk: one call.
risk = deterministic_risk_aggregate(contributions, weights)
```

---

## Why these kernels are deterministic

The accumulation order is determined by the **data layout**, not the
runtime schedule:

- **`tree_sum`** always combines element `2i` with `2i+1` at each
  layer. This pairing is fixed by index, so it is identical no matter
  how many threads run the layer. It also has better numerical
  accuracy than naive summation: O(log n) error growth versus O(n).

- **`kahan_sum`** tracks a compensation term recovering low-order bits
  lost at each addition; the operation sequence is fixed, so the
  result is reproducible with O(eps) total error.

- **`pairwise_kahan_sum`** sums fixed-size blocks with Kahan
  compensation, then combines block totals with a tree. Block size is
  an explicit parameter, never a runtime choice.

---

## Verified behavior

The test suite (`tests/test_determinism.py`, 11 tests, all passing)
proves:

- `tree_sum` and `pairwise_kahan_sum` are **bit-identical** across
  repeated calls and conceptual chunkings.
- A naive parallel reduction produces **multiple distinct bit
  patterns** from one input (10 distinct values in our run).
- The compensated reductions are at least as accurate as naive
  summation. On a 50,000-element ill-conditioned input, `tree_sum`
  and `pairwise_kahan_sum` matched an exact rational-arithmetic
  reference to **zero error** (Kahan error 3.7e-9 vs naive 2.3e-7).

---

## Cost of determinism

On the native Mojo implementation, a deterministic tree reduction
typically runs within **10-30%** of the maximally-parallel
non-deterministic version — far cheaper than PyTorch's global
`torch.use_deterministic_algorithms(True)`, which can be 2-5x slower
and errors out on operators with no deterministic kernel.

---

## Repository layout

```
mojo-deterministic/
├── README.md
├── LICENSE                            MIT
├── src/
│   ├── deterministic_reduce.mojo      the library
│   └── deterministic_reduce.py        exact Python mirror
├── tests/
│   └── test_determinism.py            11 tests, all passing
├── examples/
│   ├── risk_reconciliation_example.py finance worked example
│   └── example_usage.mojo             Mojo usage
└── benchmarks/
    └── benchmark_cost.py              cost-of-determinism benchmark
```

---

## Citation

If you use this library, please cite:

```bibtex
@article{han2026mojo,
  author  = {Han, Henry},
  title   = {Mojo: A Promising Tool for Scalable Financial AI Efficiency},
  year    = {2026}
}
```

---

## Author

**Henry Han**, Data Science and Artificial Intelligence Innovation
Laboratory, School of Engineering and Computer Science, Baylor
University.

## License

[MIT](LICENSE) © 2026 Henry Han, Baylor University
