# Beyond Backprop

A math-first crash course in alternative neural network architectures, all benchmarked on MNIST.

## Architectures

| Architecture | Phase | Training | Gradients? | Expected Accuracy |
|---|---|---|---|---|
| Extreme Learning Machine | 1 | Analytic least-squares | No | 96–97% |
| Transformer | 1 | Backprop + Adam | Yes | 98.5–99% |
| Restricted Boltzmann Machine | 1 | Contrastive divergence | Approximate | ~98% (DBN) |
| Echo State Network | 2 | Linear regression on reservoir | No | 90–95% |
| Self-Organizing Map | 2 | Competitive neighborhood update | No | 85–92% |
| Spiking Neural Network | 3 | BPTT + surrogate gradients | Surrogate | 98–99% |
| Neural ODE | 3 | Adjoint method | Yes | 98–99% |

## Setup

```sh
uv sync
```

## Project Structure

```
docs/       # Typst source documents (math writeups)
src/        # Python implementations
test/       # Tests
data/       # Datasets (not committed)
```

## Building Docs

```sh
typst compile docs/intro.typ
# or watch for changes:
typst watch docs/phase1_crash_course.typ
```
