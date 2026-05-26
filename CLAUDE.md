# CLAUDE.md

This file provides guidance to Claude Code (claude.ai/code) when working with code in this repository.

## What This Repo Is

"Beyond Backprop" — a math-first crash course in alternative neural network architectures, all benchmarked on MNIST. Every document is written in [Typst](https://typst.app/) (`.typ`), a modern typesetting language with LaTeX-like math. There is no executable code.

## Building the Documents

```sh
# Compile a single document to PDF
typst compile docs/intro.typ

# Watch and recompile on save
typst watch docs/phase1_crash_course.typ

# Compile all phases at once
for f in docs/*.typ; do typst compile "$f"; done
```

Typst must be installed (`brew install typst` on macOS).

## Document Structure

| File | Contents |
|------|----------|
| `docs/intro.typ` | Course overview, prerequisites, notation reference, full architecture table |
| `docs/phase1_crash_course.typ` | ELM · Transformer · RBM |
| `docs/phase2_crash_course.typ` | Echo State Network · Self-Organizing Map |
| `docs/phase3_crash_course.typ` | Spiking Neural Networks · Neural ODE |

Each module follows a fixed structure: Core Idea → Architecture → Math → Applied to MNIST → What to Observe → Gotchas → Summary table.

## Custom Callout Blocks

All four files define the same set of reusable blocks at the top:

```typst
#note[...]      // grey — factual clarification or "why X not Y"
#gotcha[...]    // yellow — common mistakes or subtle pitfalls
#insight[...]   // green — conceptual connections across architectures
```

`phase3_crash_course.typ` additionally defines:

```typst
#definition("Title", [...])  // blue — formal mathematical definitions
```

Use these consistently when adding new content. Do not introduce new colored-block types without adding the definition at the top of the file.

## Math Conventions

Notation is defined in `docs/intro.typ` (Notation Reference section) and must stay consistent across all phases:

- `$bold(X) in RR^(N times d)$` — data matrix
- `$sigma(dot)$` — sigmoid or generic activation
- `$bold(A)^dagger$` — Moore–Penrose pseudoinverse
- `$rho(bold(A))$` — spectral radius
- `$cal(L)$` — loss function

Typst math uses `$...$` (inline) and `$ ... $` (block). It is **not** LaTeX — use `bold(X)`, `frac(a, b)`, `sum_(i=1)^N`, not `\mathbf`, `\frac`, `\sum`.

## Python Code Style

Apply these rules without being asked:

- **Python 3.14**: use modern syntax throughout (no `from __future__ import annotations`)
- **Type hints**: always, on every function signature and class attribute
  - Union types: `X | Y`, never `Union[X, Y]`
  - Optional: `X | None`, never `Optional[X]`
  - Built-in generics: `list[int]`, `tuple[str, ...]`, not `List`, `Tuple`
  - Use `typing.Literal` for constrained string params; prefer `StrEnum` when the same values appear in multiple places (e.g. data splits, modes)
- **Docstrings**: NumPy format on ALL functions (private and public)
- **Paths**: always `pathlib.Path`; accept `Path | str` at public boundaries, coerce inside
- **Immutable module-level mappings**: use `types.MappingProxyType`
- **No bare `except`**; catch specific exceptions

## Workflow

Before starting implementation of an approved plan:
1. Create a GitHub issue summarising the plan
2. Create a feature branch named after the issue number (e.g. `issue-7-mnist-loader`)
3. Implement on that branch; open a PR when done

## Architecture Coverage

Seven architectures across three phases, all applied to MNIST (784-dim input, 10 classes):

| Architecture | Phase | Training | Gradients? | Expected accuracy |
|---|---|---|---|---|
| Extreme Learning Machine | 1 | Analytic least-squares | No | 96–97% |
| Transformer | 1 | Backprop + Adam | Yes | 98.5–99% |
| Restricted Boltzmann Machine | 1 | Contrastive divergence | Approximate | ~98% (DBN) |
| Echo State Network | 2 | Linear regression on reservoir | No | 90–95% |
| Self-Organizing Map | 2 | Competitive neighborhood update | No | 85–92% |
| Spiking Neural Network | 3 | BPTT + surrogate gradients | Surrogate | 98–99% |
| Neural ODE | 3 | Adjoint method | Yes | 98–99% |
