# Linear Regression

Implementation of Linear Regression built from scratch with gradient descent, extended to polynomial fitting and regularization (Ridge, Lasso) — with every result validated against `sklearn` rather than assumed.

---

## The Core Idea

Given input `x`, predict output `y` using a straight line:

```
y_hat = θ0 + θ1 * x
```

`θ0` is where the line starts (intercept), `θ1` is how fast it rises or falls (slope). For multiple input features, this extends to:

```
y_hat = θ0 + θ1*x1 + θ2*x2 + ... + θn*xn
```

The model doesn't know `θ0` and `θ1` upfront — it has to learn them from data. That's what gradient descent does below.

---

## How the Model Learns: Cost + Gradient Descent

**Step 1 — measure error.** Mean Squared Error tells us how far off predictions are:

```
J(θ) = (1/2m) * Σ (y_hat_i - y_i)²
```

**Step 2 — reduce the error iteratively.** Gradient descent nudges `θ` in the direction that lowers `J(θ)`:

```
θ := θ - α * (1/m) * Xᵀ · (Xθ - y)
```

Here `α` is the learning rate — too large and the cost oscillates/diverges, too small and training crawls.

**Setup used in this implementation:** `α = 0.01`, `1000` iterations, single feature, 200 samples (`sklearn.datasets.make_regression`).

**Sanity check — did it actually converge correctly?**

| | Intercept | Slope | R² |
|---|---|---|---|
| From scratch (gradient descent) | 2.8535 | 86.9798 | 0.9414 |
| sklearn `LinearRegression` | 2.8570 | 86.9955 | 0.9414 |

Both land in the same place. That match is the real proof the from-scratch loop is correct — not just that it runs without errors.

---

## Going Beyond Straight Lines: Polynomial Regression

A straight line can't fit curved data. The fix isn't a new algorithm — it's feeding the same Linear Regression *more features*: `x`, `x²`, `x³`, etc. It's still linear in the parameters, just not linear in `x` anymore.

```
y = θ0 + θ1*x + θ2*x² + θ3*x³ + ...
```

**Degree controls model complexity — tested at 1, 4, and 15:**

| Degree | Train R² | Test R² |
|---|---|---|
| 1 | 0.9116 | 0.8742 |
| 4 | 0.9118 | 0.8735 |
| 15 | 0.9164 | 0.8871 |

Interesting catch: on this particular dataset, test R² doesn't visibly collapse at degree 15 — the noise level is high enough that even a degree-15 fit doesn't diverge wildly in R² terms. This is a useful lesson on its own: **R² alone can miss overfitting.** The real evidence of overfitting shows up in the next section — the *size* of the coefficients, not the score.

**A cleaner underfit/correct-fit demo** — generated quadratic data directly (`y = 0.8x² + 0.9x + 2 + noise`) and compared:

| Fit | R² |
|---|---|
| Straight line (degree 1) on quadratic data | 0.2539 |
| Degree-2 polynomial on the same data | 0.9862 |

This is underfitting made obvious — wrong model shape, not a training problem.

---

## Controlling Overfitting: Regularization

Once a model is complex enough to overfit (degree-15 polynomial features here), the fix is to penalize large coefficients rather than reduce features by hand.

| Method | Adds to Cost | What Happens to Coefficients |
|---|---|---|
| Ridge (L2) | `λ * Σθ²` | Shrink toward zero, never hit exactly zero |
| Lasso (L1) | `λ * Σ\|θ\|` | Some get pushed to exactly zero |
| ElasticNet | L1 + L2 mix | Both effects combined — *not implemented here yet* |

**Applied to the degree-15 polynomial features** (No Reg vs Ridge α=1.0 vs Lasso α=0.01), plotted coefficient-by-coefficient:

- **No regularization** — coefficients swing to large, unstable values. This is where overfitting actually shows up clearly, unlike the R² table above.
- **Ridge** — every coefficient shrinks, but stays non-zero.
- **Lasso** — a chunk of coefficients get zeroed out entirely, effectively deciding those polynomial terms don't matter.

*(Ridge/Lasso here use `sklearn.linear_model`, not hand-written — only the base gradient descent is from scratch.)*

---

## Visuals

| Plot | What to look for |
|---|---|
| `plots/01_cost_curve.png` | Cost dropping and flattening — confirms gradient descent converged |
| `plots/02_best_fit.png` | Scratch line vs sklearn line — near-perfect overlap |
| `plots/03_poly_comparison.png` | Degree 1 / 4 / 15 fits side by side |
| `plots/04_regularization.png` | No-Reg vs Ridge vs Lasso coefficient magnitudes on degree-15 features |
| `plots/05_curve_fit(degree=2).png` | Degree-2 fit correctly tracing the quadratic curve |
| `plots/06_underfitting.png` | Straight line failing on quadratic data |

---

## Picking the Right :

| If... | Use |
|---|---|
| Relationship between X and y looks straight | Plain Linear Regression |
| Relationship is curved | Polynomial Regression |
| Model overfits but every feature is likely relevant | Ridge |
| Model overfits and some features are probably noise | Lasso |
| Not sure which of the above applies | ElasticNet |

---

## Stack

`numpy` for the scratch gradient descent · `sklearn` for validation, polynomial features, Ridge/Lasso, and train/test splitting · `matplotlib` for all plots.

## What's Next
- Rewrite Ridge and Lasso from scratch instead of relying on `sklearn`
- Add ElasticNet
- Replace fixed `alpha` values with cross-validated selection










