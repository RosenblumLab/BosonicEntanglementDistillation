# Task: Add an opt-in photon-distribution-aware qudit error model

## Repository


Inspect the repository before editing. In particular, find:

- the existing `Qudit` class;
- the function or method that calculates photon-loss probabilities;
- the continuous-variable state construction and its Fock-basis representation;
- the code that generates the discrete-versus-continuous comparison corresponding to Fig. 8;
- the existing tests, plotting conventions, and dependency-management files.

Do not assume filenames or APIs before inspecting them.

## Objective

Add an **opt-in improved discrete error model** that calculates loss-count probabilities from the full photon-number distribution rather than only the mean photon number.

Preserve the current `Qudit` implementation and its existing behavior as much as possible. Add a small subclass named:

```python
class PhotonDistributionQudit(Qudit):
    ...
```

Use `PhotonDistributionQudit` consistently in the implementation, notebook, tests, and documentation. Do not rename the existing `Qudit` class or perform an unrelated refactor merely to accommodate the subclass.

The primary goal is a runnable notebook containing a figure like the first figure in:

```text
notebook/break_even_gamma_ratio_alpha7.ipynb
```

The new figure must retain the existing curves and add another curve whose legend label is exactly:

```text
photon improved discrete
```

## Mathematical specification

Let

```text
gamma_l = photon-loss probability
eta     = 1 - gamma_l
P[n]    = probability that the input mode contains n photons
```

The loss Kraus operator is

```text
L_l = sqrt(gamma_l**l / l!) * eta**(n_hat/2) * a**l.
```

Its exact event probability is

```text
p_l = <L_l^dagger L_l>
    = sum_{n=l}^infinity P[n] * binom(n,l)
      * gamma_l**l * eta**(n-l).
```

Equivalently,

```text
p_l = gamma_l**l / l!
      * <n_hat * (n_hat-1) * ... * (n_hat-l+1)
         * eta**(n_hat-l)>.
```

This formula is exact for the supplied photon-number distribution. It replaces the existing sharply-peaked approximation

```text
p_tilde_l ~= gamma_l**l / l!
             * eta**n_bar * n_bar**l.
```

Do not use the factorial-moment small-`gamma_l` expansion as the implementation. The implementation should evaluate the full binomial average. Factorial moments may be reported as diagnostics.

## Input distribution

The improved class should accept the photon-number distribution explicitly, rather than silently reconstructing it from only `alpha` and `zeta`. Prefer an API along the following lines, adapted to the existing class constructor:

```python
PhotonDistributionQudit(
    ...existing Qudit arguments...,
    photon_number_distribution=P_n,
)
```

Accept a one-dimensional real array satisfying, up to numerical tolerance:

```text
P_n[n] >= 0
sum(P_n) = 1
```

Requirements:

- Validate that the array is one-dimensional, finite, real, and nonnegative within a small numerical tolerance.
- Reject a distribution with zero or non-finite total weight.
- Normalize a distribution whose total differs from one only because of ordinary Fock-space truncation or floating-point error.
- If the truncation deficit is appreciable, warn or raise according to the repository's existing style. Do not hide a materially under-normalized state.
- Avoid mutating the caller's array.
- Provide a small helper that extracts `P_n` from the diagonal of a state vector or density matrix if the codebase already uses QuTiP objects. Keep this helper separate from the core probability calculation.

For a pure state with Fock coefficients `c_n`, use

```text
P_n = abs(c_n)**2.
```

For a density matrix `rho`, use

```text
P_n = real(diag(rho)).
```

Use the state of the relevant cavity **immediately before the loss channel**. If the simulation applies loss to an entangled two-mode state, inspect whether the existing discrete model intentionally treats the two cavities independently. Preserve that assumption for this first opt-in subclass unless joint loss statistics can be added without broad changes. Document the assumption.

## Required behavior

Override only the smallest method responsible for constructing the loss-error distribution. Reuse all existing `Qudit` protocol, decoder, dephasing, and measurement logic.

For each requested loss count `l`, calculate

```python
p_l = sum(
    P_n[n]
    * binom(n, l)
    * gamma_l**l
    * (1 - gamma_l)**(n - l)
    for n in range(l, len(P_n))
)
```

Use a numerically stable implementation appropriate for the repository's parameter range. Prefer vectorized or recurrence-based evaluation over Python loops if NumPy/SciPy are already dependencies. Do not introduce a new heavy dependency solely for this calculation.

Important details:

- The exact distribution is already normalized when the input `P_n` is normalized and all physically possible `l` values are included. Do not apply the old ad hoc normalization before checking truncation.
- In a finite Fock basis, calculate all `l = 0, ..., N-1`, or enough terms to demonstrate that the omitted probability is below a declared tolerance.
- If the discrete qudit represents loss shifts modulo dimension `d`, aggregate equivalent loss counts only at the interface where the physical count is converted to a discrete shift:

  ```text
  q_r = sum_{j >= 0} p_{r + j*d},  r = 0, ..., d-1.
  ```

  Do not discard probabilities for `l >= d` without accounting for them.
- Handle `gamma_l = 0` exactly: `p_0 = 1`, all other probabilities zero.
- Define or reject `gamma_l = 1` deliberately; avoid expressions such as `0**0` or negative exponents.
- Reject `gamma_l` outside `[0, 1]`.

## Backward compatibility

- Do not change the numerical results or public defaults of the existing `Qudit` class.
- Existing scripts should continue to run unchanged.
- The improved model must be selected explicitly.
- Avoid changing unrelated files or formatting unrelated code.
- If a tiny protected hook in `Qudit` is necessary to avoid duplicating substantial code, make the smallest possible backward-compatible change and explain it in the handoff.

## Validation and tests

Add focused tests using the repository's current test framework.

### 1. Fixed Fock state

For `P_n = delta[n, n0]`, verify

```text
p_l = binom(n0,l) * gamma_l**l * (1-gamma_l)**(n0-l)
```

for all `0 <= l <= n0`, and zero above `n0`.

### 2. Coherent state

For a sufficiently well-converged truncated coherent state with mean `n_bar`, verify that the result agrees with the thinned Poisson distribution

```text
p_l = exp(-gamma_l*n_bar)
      * (gamma_l*n_bar)**l / l!.
```

Choose the Fock cutoff so that truncation error is well below the assertion tolerance.

### 3. Normalization and positivity

Verify

```text
all(p_l >= 0)
abs(sum(p_l) - 1) < tolerance.
```

Also test an intentionally insufficient/truncated distribution according to the chosen warning/error policy.

### 4. Edge cases

Test at least:

- `gamma_l = 0`;
- an invalid negative `P_n` entry;
- an invalid loss probability;
- modulo-`d` aggregation, if applicable.

### 5. Regression

Run the existing test suite and demonstrate that the original `Qudit` path is unchanged.

## Required notebook and figure

Inspect `notebook/break_even_gamma_ratio_alpha7.ipynb` and understand how its first figure is produced. Create a new notebook rather than overwriting the existing one, preferably:

```text
notebook/break_even_gamma_ratio_alpha7_photon_improved.ipynb
```

Keep the structure, parameters, axes, scales, styling, and existing curves of the original notebook's first figure. Add the result obtained from `PhotonDistributionQudit` as an additional curve with the legend label exactly `photon improved discrete`.

The figure should therefore compare, without replacing existing curves:

1. the original mean-photon-number discrete model;
2. the `PhotonDistributionQudit` result using the appropriate full `P(n)` for the state being plotted, labeled `photon improved discrete`;
3. the existing continuous-variable results and baselines already present in the first figure.

Do not add extra improved curves unless they are useful for validation and do not make the requested comparison ambiguous. The requested output is one clearly identifiable `photon improved discrete` curve added to the original comparison.

Use the existing Fig. 8 parameters where available, including approximately

```text
alpha = 7
zeta = -0.8
d = m_i = 16
m_c = 8
m_f = 2
gamma_l = gamma_phi along the original sweep
```

Obtain `P(n)` using the repository's existing state-generation convention. Do not independently change the sign or order of its squeezing and displacement operators. Confirm that the generated squeezed state reproduces the reported photon-number standard deviation `Delta n ~= 15.7`; if it does not, stop and report the convention mismatch rather than tuning the state silently.

Execute the new notebook from a clean kernel and verify that all cells complete successfully and the requested curve appears in the first comparison figure. Do not overwrite a manuscript figure by default. If the notebook saves the figure to disk, use a new descriptive filename.

Report quantitative differences, ideally including:

- the improved-versus-original discrete infidelity;
- whether and by how much the improved curve closes the gap to the CV curve;
- low-order normalized factorial moments

  ```text
  g_k = <n*(n-1)*...*(n-k+1)> / n_bar**k
  ```

  for `k = 1, 2, 3, 4` at the squeezed operating point.

As a cross-check, for the state orientation consistent with `alpha = 7`, `zeta = -0.8`, and `Delta n ~= 15.7`, the expected values are approximately

```text
n_bar ~= 49.789
g_1    = 1
g_2    ~= 1.079
g_3    ~= 1.246
g_4    ~= 1.531
```

Treat these as checks, not as hard-coded inputs.

## Scope limitation to document

This change improves the **classical probabilities assigned to different photon-loss counts**. It does not make the discrete action of loss exact for squeezed states.

In particular, the full Kraus action `a**l` can deform a squeezed conditional state rather than merely enact the discrete cyclic shift `Z**(-l)`. Any residual discrepancy between the improved discrete model and the CV simulation may therefore come from:

- conditional-state deformation under loss;
- finite angular overlap and phase-space discretization;
- the representation of loss as a clean cyclic shift;
- dephasing discretization;
- decoder assumptions;
- neglected two-mode photon-number correlations, if marginal distributions are used.

Do not describe the improved model as a rigorous bound on the entire discrete approximation error.

## Deliverables

At completion, provide:

1. the new opt-in `PhotonDistributionQudit` subclass and any minimal helper functions;
2. focused unit tests;
3. a new runnable notebook based on `notebook/break_even_gamma_ratio_alpha7.ipynb` whose first comparison figure contains the additional curve labeled `photon improved discrete`;
4. the executed notebook output, generated numerical summary, and figure;
5. the commands used to run the tests and execute the notebook;
6. a concise list of changed files;
7. an explanation of any unavoidable modification to the base `Qudit` class;
8. any assumptions about Fock truncation, modulo-`d` aggregation, and independent versus joint loss statistics.

Do not commit or push unless explicitly asked.
