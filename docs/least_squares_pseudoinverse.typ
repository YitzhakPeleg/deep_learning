#set document(title: "Least Squares and the Moore–Penrose Pseudoinverse", author: "")
#set page(margin: (x: 2.5cm, y: 2.5cm))
#set text(font: "New Computer Modern", size: 11pt)
#set heading(numbering: "1.1")
#set math.equation(numbering: "(1)")
#show math.equation.where(block: true): it => pad(y: 0.5em, it)

#let note(body) = block(
  fill: luma(235),
  inset: 10pt,
  radius: 4pt,
  width: 100%,
  body
)

#let gotcha(body) = block(
  fill: rgb("fff3cd"),
  inset: 10pt,
  radius: 4pt,
  width: 100%,
  [⚠ *Gotcha:* ] + body
)

#let insight(body) = block(
  fill: rgb("d4edda"),
  inset: 10pt,
  radius: 4pt,
  width: 100%,
  [💡 *Insight:* ] + body
)

#let definition(title, body) = block(
  fill: rgb("cce5ff"),
  inset: 10pt,
  radius: 4pt,
  width: 100%,
  [*Definition (#title):* ] + body
)

// ─── TITLE ───────────────────────────────────────────────────────────────────

#align(center)[
  #text(size: 22pt, weight: "bold")[Beyond Backprop — Mathematical Foundations]
  #v(0.3em)
  #text(size: 15pt)[Least Squares and the Moore–Penrose Pseudoinverse]
  #v(0.3em)
  #text(size: 10pt, fill: luma(100))[The analytic engine behind the Extreme Learning Machine]
]

#v(1em)
#line(length: 100%)
#v(0.5em)

#outline(indent: 1em, depth: 2)

#v(0.5em)
#line(length: 100%)

// ─── INTRO ───────────────────────────────────────────────────────────────────

= Why This Document Exists

The Extreme Learning Machine (ELM) trains its output weights by solving a linear system — no gradient descent, no iteration. The phrase "solved analytically via the pseudoinverse" appears in one line in most tutorials and is immediately accepted without explanation. This document makes that line honest.

We will derive everything from first principles: what it means to solve an overdetermined linear system, why the normal equations arise, what the Moore–Penrose pseudoinverse actually is (and how to construct it from scratch via singular value decomposition), and why ridge regularization is not optional in practice. Every step is justified.

The development proceeds in four parts:

+ *The least squares problem* — what we are minimizing and why it is the right objective.
+ *The normal equations* — two derivations (calculus and geometry) leading to the same formula.
+ *The Moore–Penrose pseudoinverse* — formal definition, construction via SVD, and the four conditions that make it unique.
+ *Regularization and numerics* — why the theoretical solution breaks down in practice and how ridge regression fixes it.

// ═══════════════════════════════════════════════════════════════════════════════
= The Problem: Solving an Overdetermined Linear System
// ═══════════════════════════════════════════════════════════════════════════════

== Setup

We want to find a matrix $bold(beta) in RR^(L times C)$ satisfying

$ bold(H) bold(beta) = bold(Y) $

where $bold(H) in RR^(N times L)$ is given (the hidden-layer output matrix in the ELM, but the analysis is completely general) and $bold(Y) in RR^(N times C)$ is the target (the one-hot label matrix).

This is a system of $N times C$ equations in $L times C$ unknowns. The case of interest is $N > L$: many more data points than parameters. The system is *overdetermined* — there are more equations than unknowns — and in general *no exact solution exists*.

#note[
  For MNIST with $N = 60{,}000$ training samples and $L = 1000$ hidden neurons, we have 60,000 equations per output class and only 1,000 unknowns per class. The 60,000 equations are almost never simultaneously satisfiable. We need a principled way to find the "best" $bold(beta)$ when perfection is off the table.
]

Because each output class (each column of $bold(beta)$) decouples — the $c$-th column of $bold(Y)$ only involves the $c$-th column of $bold(beta)$ — it suffices to analyze the single-output case: find $bold(beta) in RR^L$ minimizing $||bold(H) bold(beta) - bold(y)||_2^2$ for a column vector $bold(y) in RR^N$. The multi-output result follows by applying the same formula column-wise, which is exactly what the matrix pseudoinverse does.

== The Least Squares Objective

The *ordinary least squares* (OLS) problem is:

$ bold(beta)^* = arg min_(bold(beta) in RR^L) cal(L)(bold(beta)), quad cal(L)(bold(beta)) = ||bold(H) bold(beta) - bold(y)||_2^2 $

Expanding the squared norm (using $||bold(v)||^2 = bold(v)^top bold(v)$):

$ cal(L)(bold(beta)) = (bold(H) bold(beta) - bold(y))^top (bold(H) bold(beta) - bold(y)) $

$ = bold(beta)^top bold(H)^top bold(H) bold(beta) - 2 bold(y)^top bold(H) bold(beta) + bold(y)^top bold(y) $

This is a *quadratic form* in $bold(beta)$. The matrix $bold(H)^top bold(H) in RR^(L times L)$ is symmetric and positive semi-definite (PSD), so the function $cal(L)$ is convex, and any critical point is a global minimum.

#insight[
  The loss $cal(L)(bold(beta))$ is a paraboloid (or a flat-bottomed channel if $bold(H)$ is rank-deficient). A convex paraboloid has exactly one minimum. A flat-bottomed channel has infinitely many — they form an affine subspace. The pseudoinverse picks the unique one with minimum norm from that subspace.
]

// ═══════════════════════════════════════════════════════════════════════════════
= The Normal Equations
// ═══════════════════════════════════════════════════════════════════════════════

== Derivation via Calculus

Setting the gradient to zero gives the necessary (and here also sufficient) condition for a minimum.

$ nabla_(bold(beta)) cal(L) = 2 bold(H)^top bold(H) bold(beta) - 2 bold(H)^top bold(y) = bold(0) $

Setting this to zero:

$ bold(H)^top bold(H) bold(beta) = bold(H)^top bold(y) $

This is the *normal equation*. Its solutions are the least squares solutions to $bold(H) bold(beta) = bold(y)$.

#note[
  The gradient formula $nabla_(bold(beta)) (bold(A) bold(beta)) = bold(A)^top$ and $nabla_(bold(beta)) (bold(beta)^top bold(A) bold(beta)) = 2 bold(A) bold(beta)$ (for symmetric $bold(A)$) are standard results from matrix calculus. They follow from writing out the scalar expansion and differentiating term by term. The factor of 2 cancels when setting the gradient to zero, so it does not affect the solution.
]

If $bold(H)^top bold(H)$ is invertible — equivalently, if $bold(H)$ has full column rank ($"rank"(bold(H)) = L$) — the unique solution is:

$ bold(beta)^* = (bold(H)^top bold(H))^(-1) bold(H)^top bold(y) $

The matrix $(bold(H)^top bold(H))^(-1) bold(H)^top$ is called the *left pseudoinverse* of $bold(H)$, and in this full-rank overdetermined case it coincides with the Moore–Penrose pseudoinverse $bold(H)^dagger$.

== Derivation via Geometry

There is a cleaner way to see why the normal equations must hold — one that does not require computing any derivatives.

The columns of $bold(H)$ span a subspace $cal(C)(bold(H)) subset.eq RR^N$ called the *column space* of $bold(H)$. When we compute $bold(H) bold(beta)$ for any $bold(beta)$, we get a vector in $cal(C)(bold(H))$. Since $bold(y) in.not cal(C)(bold(H))$ in the overdetermined case, no $bold(beta)$ makes $bold(H) bold(beta) = bold(y)$ exactly.

The closest point in $cal(C)(bold(H))$ to $bold(y)$ — the point that minimizes $||bold(H) bold(beta) - bold(y)||$ — is the *orthogonal projection* of $bold(y)$ onto $cal(C)(bold(H))$. Call it $hat(bold(y)) = bold(H) bold(beta)^*$.

Orthogonality means the residual $bold(r) = bold(y) - hat(bold(y))$ must be perpendicular to every vector in $cal(C)(bold(H))$:

$ bold(r) perp cal(C)(bold(H)) $

$ (bold(y) - bold(H) bold(beta)^*) perp bold(H) bold(v) quad forall bold(v) in RR^L $

$ bold(v)^top bold(H)^top (bold(y) - bold(H) bold(beta)^*) = 0 quad forall bold(v) in RR^L $

Since this must hold for every $bold(v)$:

$ bold(H)^top (bold(y) - bold(H) bold(beta)^*) = bold(0) $

$ bold(H)^top bold(H) bold(beta)^* = bold(H)^top bold(y) $

This is exactly the normal equation again — derived without touching a derivative. The *geometric insight* is: the optimal $bold(beta)^*$ is the one whose fitted values $bold(H) bold(beta)^*$ are the orthogonal projection of $bold(y)$ onto the column space of $bold(H)$.

#insight[
  This geometric view makes a prediction immediately: the *residual* $bold(y) - bold(H) bold(beta)^*$ is always orthogonal to every column of $bold(H)$. This is a testable, interpretable statement — not just a consequence of algebra. It says that least squares leaves no linear "signal" unexplained in the fitted direction; all remaining error is genuinely perpendicular to the model's reach.
]

== When the Normal Equations Have Infinitely Many Solutions

If $bold(H)$ does not have full column rank — for example when $L > N$ (more neurons than training samples) — then $bold(H)^top bold(H)$ is *singular* and has no inverse. The normal equations still hold, but they have infinitely many solutions: every $bold(beta)^*$ in a particular affine subspace minimizes the squared error equally.

Among all these solutions, one stands out: the one with the *smallest Euclidean norm*, $||bold(beta)^*||_2$. It has a clean geometric meaning — it is the solution closest to the origin, lying in the orthogonal complement of the null space of $bold(H)$. The Moore–Penrose pseudoinverse always returns this minimum-norm least-squares solution, whether or not $bold(H)$ has full rank.

// ═══════════════════════════════════════════════════════════════════════════════
= The Singular Value Decomposition
// ═══════════════════════════════════════════════════════════════════════════════

The pseudoinverse is best understood through the SVD, so we develop it carefully first.

== Statement

Every real matrix $bold(H) in RR^(N times L)$ of rank $r$ can be factored as:

$ bold(H) = bold(U) bold(Sigma) bold(V)^top $

where:

- $bold(U) in RR^(N times N)$ is *orthogonal*: $bold(U)^top bold(U) = bold(U) bold(U)^top = bold(I)_N$. Its columns $bold(u)_1, dots, bold(u)_N$ are the *left singular vectors*.
- $bold(Sigma) in RR^(N times L)$ is *diagonal* (in the rectangular sense): $Sigma_(i i) = sigma_i >= 0$, all off-diagonal entries zero. The values $sigma_1 >= sigma_2 >= dots >= sigma_r > 0 = sigma_(r+1) = dots$ are the *singular values*.
- $bold(V) in RR^(L times L)$ is *orthogonal*: $bold(V)^top bold(V) = bold(V) bold(V)^top = bold(I)_L$. Its columns $bold(v)_1, dots, bold(v)_L$ are the *right singular vectors*.

== Why the SVD Exists

Consider the symmetric PSD matrix $bold(H)^top bold(H) in RR^(L times L)$. By the spectral theorem for symmetric matrices, it has a complete set of orthonormal eigenvectors $bold(v)_1, dots, bold(v)_L$ with non-negative eigenvalues $lambda_1 >= dots >= lambda_L >= 0$:

$ bold(H)^top bold(H) bold(v)_i = lambda_i bold(v)_i $

Define $sigma_i = sqrt(lambda_i)$ (the singular values) and, for each $i$ with $sigma_i > 0$:

$ bold(u)_i = frac(1, sigma_i) bold(H) bold(v)_i $

These $bold(u)_i$ are orthonormal (verifiable from $bold(H)^top bold(H) bold(v)_i = sigma_i^2 bold(v)_i$) and span the column space of $bold(H)$. Extend them to a full orthonormal basis of $RR^N$ by appending vectors spanning the null space of $bold(H)^top$. This constructs $bold(U)$, $bold(Sigma)$, $bold(V)$, and by construction $bold(H) bold(v)_i = sigma_i bold(u)_i$, which is exactly $bold(H) = bold(U) bold(Sigma) bold(V)^top$ in matrix form.

== The Compact SVD

In practice only the $r$ non-zero singular values matter. The *compact (thin) SVD* keeps only the first $r$ columns of $bold(U)$ and $bold(V)$ and the $r times r$ diagonal block of $bold(Sigma)$:

$ bold(H) = bold(U)_r bold(Sigma)_r bold(V)_r^top $

where $bold(U)_r in RR^(N times r)$, $bold(Sigma)_r = "diag"(sigma_1, dots, sigma_r) in RR^(r times r)$, $bold(V)_r in RR^(L times r)$.

#note[
  The full SVD and the compact SVD encode the same matrix. The difference is bookkeeping: the full SVD pads $bold(Sigma)$ with zeros and extends $bold(U)$ and $bold(V)$ with vectors from the null spaces, which carry no information about $bold(H)$. Most numerical libraries return the compact form by default (e.g. `numpy.linalg.svd` with `full_matrices=False`).
]

// ═══════════════════════════════════════════════════════════════════════════════
= The Moore–Penrose Pseudoinverse
// ═══════════════════════════════════════════════════════════════════════════════

== Motivation: Inverting the SVD

If $bold(H) = bold(U) bold(Sigma) bold(V)^top$ and $bold(H)$ were square and invertible, the inverse would be $bold(H)^(-1) = bold(V) bold(Sigma)^(-1) bold(U)^top$ (using orthogonality of $bold(U)$ and $bold(V)$).

For a rectangular or rank-deficient $bold(H)$, we cannot invert $bold(Sigma)$ outright because it is not square and has zero diagonal entries. But we can *invert the non-zero entries and transpose*: define $bold(Sigma)^dagger in RR^(L times N)$ by

$ Sigma^dagger_(i i) = cases(1 / sigma_i & "if" sigma_i > 0, 0 & "if" sigma_i = 0) $

with all off-diagonal entries zero. Then:

#definition("Moore–Penrose Pseudoinverse")[
  The *Moore–Penrose pseudoinverse* of $bold(H) in RR^(N times L)$ is:

  $ bold(H)^dagger = bold(V) bold(Sigma)^dagger bold(U)^top $
]

For the compact SVD ($r$ non-zero singular values):

$ bold(H)^dagger = bold(V)_r bold(Sigma)_r^(-1) bold(U)_r^top $

This is a well-defined matrix of shape $L times N$, regardless of whether $bold(H)$ is tall, wide, square, full-rank, or rank-deficient.

== The Four Penrose Conditions

The pseudoinverse is uniquely characterized — for any matrix $bold(H)$, there exists exactly one matrix $bold(G)$ satisfying all four of the following conditions:

#table(
  columns: (auto, auto, 1fr),
  stroke: 0.5pt,
  inset: 8pt,
  align: left,
  table.header([*No.*], [*Condition*], [*Geometric meaning*]),
  [1], [$bold(H) bold(G) bold(H) = bold(H)$], [$bold(G)$ is a "generalized inverse"],
  [2], [$bold(G) bold(H) bold(G) = bold(G)$], [$bold(H)$ is a "generalized inverse" of $bold(G)$],
  [3], [$(bold(H) bold(G))^top = bold(H) bold(G)$], [$bold(H) bold(G)$ is an orthogonal projector],
  [4], [$(bold(G) bold(H))^top = bold(G) bold(H)$], [$bold(G) bold(H)$ is an orthogonal projector],
)

The matrix $bold(G) = bold(H)^dagger$ constructed via the SVD satisfies all four. Any other generalized inverse violates at least one. This uniqueness is what makes the pseudoinverse *the* canonical generalization of the matrix inverse.

Let us verify conditions 3 and 4 for the SVD construction, since they are the most illuminating.

*Condition 3:* $bold(H) bold(H)^dagger$ projects onto the column space of $bold(H)$.

$ bold(H) bold(H)^dagger = (bold(U) bold(Sigma) bold(V)^top)(bold(V) bold(Sigma)^dagger bold(U)^top) = bold(U) (bold(Sigma) bold(Sigma)^dagger) bold(U)^top $

Now $bold(Sigma) bold(Sigma)^dagger$ is a diagonal matrix with entry $(i,i)$ equal to $sigma_i / sigma_i = 1$ if $sigma_i > 0$, and $0 dot 0 = 0$ otherwise. So $bold(Sigma) bold(Sigma)^dagger = "diag"(1, dots, 1, 0, dots, 0)$ with $r$ ones and $N - r$ zeros. Therefore:

$ bold(H) bold(H)^dagger = bold(U)_r bold(U)_r^top $

This is the orthogonal projector onto $"span"(bold(u)_1, dots, bold(u)_r) = cal(C)(bold(H))$. Symmetric, as required.

*Condition 4:* $bold(H)^dagger bold(H)$ projects onto the row space of $bold(H)$.

$ bold(H)^dagger bold(H) = (bold(V) bold(Sigma)^dagger bold(U)^top)(bold(U) bold(Sigma) bold(V)^top) = bold(V) (bold(Sigma)^dagger bold(Sigma)) bold(V)^top = bold(V)_r bold(V)_r^top $

This is the orthogonal projector onto $"span"(bold(v)_1, dots, bold(v)_r) = cal(C)(bold(H)^top)$, the row space of $bold(H)$. Also symmetric.

#insight[
  The four Penrose conditions make the pseudoinverse the *unique* "best possible inverse." Conditions 1 and 2 say that $bold(H)$ and $bold(H)^dagger$ are mutual generalized inverses (multiplying by one and then the other gets you back to where you started, up to projection). Conditions 3 and 4 say that the two products $bold(H) bold(H)^dagger$ and $bold(H)^dagger bold(H)$ are *orthogonal projectors* — no rotation, no shear, just perpendicular projection. This is what makes the resulting solution the minimum-norm least-squares one.
]

== The Pseudoinverse Gives the Minimum-Norm Least-Squares Solution

We can now prove the key claim precisely.

*Claim.* The vector $bold(beta)^* = bold(H)^dagger bold(y)$ simultaneously:
1. minimizes $||bold(H) bold(beta) - bold(y)||_2^2$ over all $bold(beta) in RR^L$, and
2. among all minimizers, has the smallest $||bold(beta)||_2$.

*Proof (part 1 — least squares).* Write $bold(H) = bold(U) bold(Sigma) bold(V)^top$ and change variables $bold(alpha) = bold(V)^top bold(beta)$ (an orthogonal change of basis, so norms are preserved). Let $bold(c) = bold(U)^top bold(y)$. Then:

$ ||bold(H) bold(beta) - bold(y)||^2 = ||bold(U) bold(Sigma) bold(alpha) - bold(y)||^2 = ||bold(Sigma) bold(alpha) - bold(c)||^2 $

(using $bold(U)^top bold(U) = bold(I)$ and that $bold(U)$ is norm-preserving). Since $bold(Sigma)$ is diagonal, this decouples into $L + (N - L)$ independent scalar terms:

$ = sum_(i=1)^r (sigma_i alpha_i - c_i)^2 + sum_(i=r+1)^N c_i^2 $

The second sum does not depend on $bold(alpha)$ at all — it is the irreducible error. Minimizing the first sum: for each $i <= r$, the optimal $alpha_i^* = c_i / sigma_i$. For $i > r$ (zero singular values), $sigma_i = 0$ so these terms vanish regardless of $alpha_i$; we are free to set them to zero (the minimum-norm choice).

Therefore $bold(alpha)^* = bold(Sigma)^dagger bold(c) = bold(Sigma)^dagger bold(U)^top bold(y)$, and transforming back:

$ bold(beta)^* = bold(V) bold(alpha)^* = bold(V) bold(Sigma)^dagger bold(U)^top bold(y) = bold(H)^dagger bold(y) $

*Proof (part 2 — minimum norm).* Suppose $bold(beta)'$ is another least-squares minimizer. Then $bold(H) bold(beta)' = bold(H) bold(beta)^*$ (same projection onto the column space), so $bold(H)(bold(beta)' - bold(beta)^*) = bold(0)$, meaning $bold(beta)' - bold(beta)^* in cal(N)(bold(H))$ (the null space of $bold(H)$). Since $bold(beta)^* = bold(H)^dagger bold(y) = bold(V)_r bold(Sigma)_r^(-1) bold(U)_r^top bold(y)$ lies in the row space $cal(C)(bold(V)_r) = cal(C)(bold(H)^top)$, which is orthogonal to $cal(N)(bold(H))$:

$ ||bold(beta)'||^2 = ||bold(beta)^* + (bold(beta)' - bold(beta)^*)||^2 = ||bold(beta)^*||^2 + ||bold(beta)' - bold(beta)^*||^2 >= ||bold(beta)^*||^2 $

with equality only when $bold(beta)' = bold(beta)^*$. $square$

== The Three Regimes

The pseudoinverse behaves differently depending on the shape and rank of $bold(H)$:

#table(
  columns: (auto, auto, auto, 1fr),
  stroke: 0.5pt,
  inset: 8pt,
  align: left,
  table.header([*Regime*], [*Condition*], [*$bold(H)^dagger$*], [*Solution character*]),
  [Overdetermined, full rank], [$N > L$, $"rank" = L$], [$(bold(H)^top bold(H))^(-1) bold(H)^top$], [Unique least-squares solution],
  [Underdetermined, full rank], [$N < L$, $"rank" = N$], [$bold(H)^top (bold(H) bold(H)^top)^(-1)$], [Minimum-norm exact solution],
  [Rank-deficient], [$"rank" = r < min(N, L)$], [$bold(V)_r bold(Sigma)_r^(-1) bold(U)_r^top$], [Minimum-norm least-squares solution],
)

#note[
  In the underdetermined full-rank case, $bold(H) bold(beta) = bold(y)$ is satisfied exactly (the system is consistent), and $bold(H)^dagger bold(y) = bold(H)^top (bold(H) bold(H)^top)^(-1) bold(y)$ is the specific solution that lies in the row space — i.e., the one with no null-space component. This is the "smoothest" solution, the one that generalizes best in the absence of other information.
]

// ═══════════════════════════════════════════════════════════════════════════════
= Regularization and Numerical Reality
// ═══════════════════════════════════════════════════════════════════════════════

== The Conditioning Problem

The theoretical formula $bold(beta)^* = bold(H)^dagger bold(y)$ assumes exact arithmetic. In floating point, the bottleneck is the inversion of the $r times r$ matrix $bold(Sigma)_r$ — specifically, division by small singular values.

If $bold(H)$ has a singular value $sigma_r$ that is very small but nonzero (numerical near-rank-deficiency), the corresponding component of the solution is amplified by $1/sigma_r$. This makes $||bold(beta)^*||$ enormous, and tiny perturbations in $bold(y)$ (measurement noise, floating-point rounding) translate to huge swings in $bold(beta)^*$.

The *condition number* of $bold(H)$ is $kappa(bold(H)) = sigma_1 / sigma_r$ — the ratio of the largest to the smallest non-zero singular value. Large condition number means the normal equations are ill-conditioned: the solution is numerically unreliable.

#gotcha[
  With $N = 60{,}000$ and $L = 1000$, the matrix $bold(H)^top bold(H)$ is $1000 times 1000$ and involves the squares of singular values. If $bold(H)$ has condition number $10^4$, then $bold(H)^top bold(H)$ has condition number $10^8$ — far outside the range of stable inversion with double-precision floats (machine epsilon $approx 10^{-16}$, leaving only $10^{-8}$ relative accuracy). The formula looks clean on paper; it is unreliable in code without regularization.
]

== Ridge Regression (Tikhonov Regularization)

The fix is to add a multiple of the identity to the normal equations before inverting:

$ bold(beta)^*_lambda = (bold(H)^top bold(H) + lambda bold(I))^(-1) bold(H)^top bold(y) $

This is *ridge regression* (also called Tikhonov regularization with $lambda bold(I)$). What does it do geometrically?

In the SVD basis, the solution becomes:

$ bold(beta)^*_lambda = bold(V) "diag"(frac(sigma_i, sigma_i^2 + lambda)) bold(U)^top bold(y) $

compared to the unregularized:

$ bold(beta)^* = bold(V) "diag"(frac(1, sigma_i)) bold(U)^top bold(y) $

Each $1/sigma_i$ (potentially huge for small $sigma_i$) is replaced by $sigma_i / (sigma_i^2 + lambda)$, which is bounded above by $1 / (2 sqrt(lambda))$ for all $sigma_i$. The condition number of the regularized system is:

$ kappa(bold(H)^top bold(H) + lambda bold(I)) = frac(sigma_1^2 + lambda, sigma_r^2 + lambda) $

For $lambda >> sigma_r^2$, this approaches $sigma_1^2 / lambda$ — well-conditioned and directly controllable.

#note[
  Ridge regression has a Bayesian interpretation: it is equivalent to placing a zero-mean Gaussian prior $bold(beta) ~ cal(N)(bold(0), lambda^(-1) bold(I))$ on the output weights, then computing the MAP (maximum a posteriori) estimate. Larger $lambda$ = stronger prior toward zero = smaller, smoother weights. This connects the numerical fix to a genuine modeling assumption: we believe large output weights are a priori unlikely.
]

== Bias–Variance Tradeoff

Ridge regularization introduces a bias: $bold(beta)^*_lambda$ is not the exact least-squares minimizer. As $lambda arrow.r 0$, the bias vanishes and we recover the unregularized solution (along with its numerical instability). As $lambda arrow.r infinity$, $bold(beta)^*_lambda arrow.r bold(0)$ — maximum bias, zero variance.

In SVD terms, the effective rank of the solution smoothly decreases as $lambda$ increases: components corresponding to small singular values are shrunk toward zero first. Ridge does not hard-zero any component (unlike truncated SVD), but soft-shrinks all of them proportionally.

For the ELM, $lambda$ is chosen by cross-validation or by a rough heuristic. Values in $[10^{-5}, 10^{-3}]$ work well for MNIST with $L = 1000$ — small enough to not bias the solution significantly, large enough to stabilize the inversion.

// ═══════════════════════════════════════════════════════════════════════════════
= Putting It Together: ELM Training
// ═══════════════════════════════════════════════════════════════════════════════

The full training procedure of the ELM, stated precisely:

+ *Sample* $bold(W)_1 in RR^(784 times L)$ and $bold(b) in RR^L$ from $cal(N)(bold(0), bold(I))$. Freeze them permanently.

+ *Compute* the hidden-layer output matrix:
  $ bold(H) = sigma(bold(X) bold(W)_1 + bold(1) bold(b)^top) in RR^(N times L) $
  where $bold(X) in RR^(N times 784)$ are the normalized training images.

+ *Encode* labels as one-hot: $bold(Y) in {0, 1}^(N times 10)$.

+ *Solve* for output weights using the regularized normal equations:
  $ bold(beta) = (bold(H)^top bold(H) + lambda bold(I))^(-1) bold(H)^top bold(Y) $
  This is the unique matrix that minimizes
  $ ||bold(H) bold(beta) - bold(Y)||_F^2 + lambda ||bold(beta)||_F^2 $
  the Frobenius norm playing the role of the squared norm column-wise.

+ *Predict* on a new image $bold(x)_"new" in RR^(784)$:
  $ hat(bold(y)) = sigma(bold(x)_"new" bold(W)_1 + bold(b)) bold(beta) in RR^(10) $
  The predicted class is $arg max_c hat(y)_c$.

Steps 1 and 2 cost $O(N L d)$ (matrix multiply). Step 4 costs $O(N L^2 + L^3)$ — forming $bold(H)^top bold(H)$ is $O(N L^2)$, inverting it is $O(L^3)$. With $N = 60{,}000$ and $L = 1{,}000$ this is manageable in seconds on a CPU with NumPy's LAPACK back-end.

#insight[
  The randomness in the ELM is entirely in steps 1–2. Once $bold(H)$ is fixed, step 4 is a deterministic computation with a unique answer (given $lambda$). The universal approximation theorem guarantees that for large enough $L$, there exists a random projection such that the resulting $bold(H)$ allows accurate linear separation — the probability over the random draw goes to 1 as $L arrow.r infinity$. In practice $L = 1000$ is more than enough for MNIST.
]

// ═══════════════════════════════════════════════════════════════════════════════
= Summary
// ═══════════════════════════════════════════════════════════════════════════════

#table(
  columns: (auto, 1fr),
  stroke: 0.5pt,
  inset: 8pt,
  align: left,
  table.header([*Concept*], [*Key result*]),
  [Least squares], [Minimizes $||bold(H) bold(beta) - bold(y)||^2$; the residual is perpendicular to the column space of $bold(H)$],
  [Normal equation], [$bold(H)^top bold(H) bold(beta) = bold(H)^top bold(y)$; arises from both calculus and geometry],
  [SVD], [$bold(H) = bold(U) bold(Sigma) bold(V)^top$; diagonalizes the action of $bold(H)$ into independent one-dimensional problems],
  [Pseudoinverse], [$bold(H)^dagger = bold(V) bold(Sigma)^dagger bold(U)^top$; uniquely characterized by the four Penrose conditions],
  [What $bold(H)^dagger bold(y)$ gives], [The minimum-norm vector among all least-squares solutions],
  [Full-rank overdetermined], [$bold(H)^dagger = (bold(H)^top bold(H))^(-1) bold(H)^top$; unique solution],
  [Rank-deficient], [Infinite solutions; $bold(H)^dagger bold(y)$ selects the one in the row space],
  [Ridge regression], [$(bold(H)^top bold(H) + lambda bold(I))^(-1) bold(H)^top bold(y)$; bounds condition number, introduces shrinkage bias],
  [Condition number], [$kappa = sigma_1 / sigma_r$; measures sensitivity of the solution to perturbations in $bold(y)$],
)
