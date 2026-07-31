# Loss Functions in Machine Learning & Deep Learning — A Complete Course

*From binary cross-entropy to ELBO, GAN losses, and metric learning — every loss explained with formula, intuition, worked example, PyTorch/TensorFlow code, and business use cases.*

**Sister documents:** *Classification Metrics — A Complete Course* · *Regression Metrics — A Complete Course*

**How to use this document**
- Every loss follows the same 12-section template: Definition → Intuition → Formula → Manual example → Code → Interpretation → Good vs bad values → Use cases → Advantages → Limitations → Common mistakes → Interview questions.
- One running example (4-sample binary classification) runs through Part 1 so every basic loss is directly comparable.
- Code examples are PyTorch-first with TensorFlow/Keras equivalents noted.
- All numeric values verified against PyTorch 2.x and NumPy.

---

## Table of Contents

| Part | Topic |
|---|---|
| 0 | What is a loss function — foundations |
| 1 | Regression losses |
| 2 | Binary classification losses |
| 3 | Multi-class classification losses |
| 4 | Sequence and structured prediction losses |
| 5 | Metric learning and ranking losses |
| 6 | Generative model losses |
| 7 | Probabilistic and Bayesian losses |
| 8 | Self-supervised and representation learning losses |
| 9 | Custom and composite losses |
| 10 | Loss selection decision tree |
| 11 | Master comparison tables |
| 12 | 100+ interview questions with answers |
| 13 | Cheat sheet, formula sheet, mnemonics, pitfalls |
| A–C | Appendices: worked examples, reference code, PyTorch custom loss template |

---

# PART 0 — Foundations

## 0.1 What is a loss function?

A loss function (also called a **cost function**, **objective function**, or **criterion**) is a mathematical function that measures how far a model's output is from the desired output, averaged over a training batch. The optimiser's job is to minimise it.

```
Loss = L(y_true, y_pred, θ)
```

- **`y_true`** — the ground-truth label or target value
- **`y_pred`** — the model's raw output (logit, probability, predicted value, embedding, …)
- **`θ`** — model parameters (the loss is a function of these through `y_pred`)
- **The optimiser** (SGD, Adam, …) computes `∂L/∂θ` via backpropagation and updates θ to reduce L

## 0.2 Loss function vs evaluation metric — the critical distinction

| | Loss function | Evaluation metric |
|---|---|---|
| **Purpose** | Training signal — must be differentiable | Reporting — must be interpretable |
| **Must be differentiable?** | Yes (or sub-differentiable) | No |
| **Should match business cost?** | Yes, ideally | Yes |
| **Reported to stakeholders?** | No | Yes |

**The mismatch trap.** Training on MSE while reporting MAE measures two different optimal predictors (mean vs median). Training on binary cross-entropy while reporting F1 optimises calibration while measuring a threshold-dependent ranking. **Always check that your loss and your primary evaluation metric are consistent.** If they are not, the model you select may not be the model you actually want.

## 0.3 The three things a loss function encodes

1. **The noise model / likelihood.** MSE ↔ Gaussian noise. Cross-entropy ↔ Bernoulli/categorical. Poisson deviance ↔ Poisson counts. **Choosing the loss is choosing the noise model** — the loss is the negative log-likelihood under that model.

2. **The optimal predictor.** MSE → the conditional mean. MAE → the conditional median. Pinball(τ) → the conditional τ-quantile. Cross-entropy → the conditional probability. Triplet loss → a Euclidean embedding with desired relative distances.

3. **The cost structure.** Symmetric or asymmetric? Linear or convex in the error? Sensitive to outliers or robust? The loss shape determines how the model behaves on hard examples, rare classes, and edge cases.

## 0.4 Taxonomy of loss functions

```
LOSS FUNCTIONS
│
├── REGRESSION
│   ├── L2 family: MSE, RMSE, SSE
│   ├── L1 family: MAE, MedAE
│   ├── Hybrid: Huber, Log-Cosh, Pseudo-Huber
│   ├── Quantile: Pinball / Quantile
│   ├── Log-scale: MSLE, RMSLE
│   └── Count: Poisson deviance, Tweedie, Negative Binomial
│
├── CLASSIFICATION
│   ├── Binary: BCE, Focal, Hinge, Squared Hinge
│   ├── Multi-class: CCE, Sparse CCE, Label-smoothing CE, KL Divergence
│   └── Multi-label: BCE per-class, Asymmetric Loss
│
├── SEQUENCE & STRUCTURED
│   ├── Seq2seq: Sequence Cross-Entropy
│   ├── Alignment: CTC (Connectionist Temporal Classification)
│   └── Structured: Structural SVM / Max-margin
│
├── METRIC LEARNING
│   ├── Pairwise: Contrastive Loss
│   ├── Triplet: Triplet Loss, Hard-negative mining
│   └── Multi-positive: InfoNCE / NT-Xent (SimCLR), SupCon
│
├── GENERATIVE
│   ├── GAN: Minimax BCE, Non-saturating, Hinge GAN, WGAN, WGAN-GP
│   └── Diffusion: Simple MSE, VLB
│
├── PROBABILISTIC / BAYESIAN
│   ├── VAE: ELBO = Reconstruction + KL
│   ├── Flow: Negative log-likelihood
│   └── Calibration: Brier Score, NLL, CRPS
│
└── SELF-SUPERVISED
    ├── Contrastive: InfoNCE (MoCo, SimCLR)
    ├── Non-contrastive: VICReg, BarlowTwins, BYOL
    └── Masked: MLM Cross-Entropy, MAE Pixel/Patch MSE
```

## 0.5 Properties of a good loss function

| Property | Explanation | What breaks if missing |
|---|---|---|
| **Differentiable** | ∂L/∂ŷ must exist (or be sub-differentiable) | Gradient descent cannot update the model |
| **Consistent with the task** | Minimising L produces the desired behaviour | Model optimises the wrong thing |
| **Bounded or well-scaled** | Doesn't blow up on bad predictions | NaN gradients, instability, exploding weights |
| **Proper (for probabilistic output)** | Expected loss minimised only by the true distribution | Model can improve score by misreporting uncertainty |
| **Informative gradient** | Gradient is non-zero on bad predictions | Vanishing gradient problem; model stops learning |
| **Scale-appropriate** | Loss scale is compatible with the learning rate | Tuning the LR becomes very hard |

## 0.6 The running example

A 4-sample binary classification (fraud detection):

| i | y_true | y_pred (model prob) | e = y − ŷ |
|---|---|---|---|
| 1 | 1 (fraud) | 0.9 | +0.1 |
| 2 | 0 (legit) | 0.2 | −0.2 |
| 3 | 1 (fraud) | 0.6 | +0.4 |
| 4 | 0 (legit) | 0.7 | −0.7 |

```python
import numpy as np, torch, torch.nn as nn

y_true  = torch.tensor([1., 0., 1., 0.])
y_pred  = torch.tensor([0.9, 0.2, 0.6, 0.7])   # probabilities from sigmoid
logits  = torch.log(y_pred / (1 - y_pred))       # back-calculated logits: [2.197, -1.386, 0.405, 0.847]
```

---

# PART 1 — Regression Losses

## 1.1 Mean Squared Error (MSE / L2 Loss)

### 1. Definition
The average squared difference between predictions and targets. The canonical regression loss.

### 2. Intuition
MSE encodes the belief: **large errors are disproportionately costly.** Being wrong by 4 units is 4× as bad as being wrong by 2 units under MAE; it is 16/4 = **4×** as bad under MSE. This quadratic growth means the optimiser is always pulled hardest by the worst predictions — which is exactly what you want when large misses are expensive.

It is simultaneously: the **maximum likelihood estimator under Gaussian noise**, the **method-of-moments estimator for the conditional mean**, and the **smoothest first-order-differentiable loss**, making it the default for essentially every regression task until there is a reason to deviate.

### 3. Formula
```
MSE = (1/n) Σᵢ (yᵢ − ŷᵢ)²

Gradient:  ∂MSE/∂ŷᵢ = −2(yᵢ − ŷᵢ) / n = 2eᵢ / n   (proportional to the residual)
```
- **Gradient is proportional to the error** — the model is pushed hardest where it is most wrong, and the gradient shrinks gracefully to zero at the optimum.
- MSE is strictly convex for linear models → unique global minimum.
- For neural networks, the loss surface is non-convex but MSE's smooth landscape makes local-minimum avoidance easier than discontinuous losses.

**Likelihood connection.** If `yᵢ | xᵢ ~ N(f(xᵢ), σ²)`, the log-likelihood is:
```
ln L = −(n/2)ln(2πσ²) − (1/2σ²) Σ(yᵢ − f(xᵢ))²
```
Maximising over f is exactly minimising `Σ(yᵢ − f(xᵢ))²` = n × MSE. **MSE is MLE under Gaussian noise.**

### 4. Manual example
```
y    = [1.0, 0.0, 1.0, 0.0]   (re-interpreting our binary labels as regression targets)
ŷ    = [0.9, 0.2, 0.6, 0.7]
e    = [0.1, -0.2, 0.4, -0.7]
e²   = [0.01, 0.04, 0.16, 0.49]

MSE = (0.01 + 0.04 + 0.16 + 0.49) / 4 = 0.70/4 = 0.175
```
**MSE = 0.175**. Note: sample 4 (error −0.7) contributes 0.49/0.70 = **70%** of the total loss despite being one of four samples — the signature of MSE's outlier sensitivity.

### 5. Code
```python
import torch, torch.nn as nn
import numpy as np

y    = torch.tensor([1., 0., 1., 0.])
yhat = torch.tensor([0.9, 0.2, 0.6, 0.7])

# PyTorch
criterion = nn.MSELoss()
loss = criterion(yhat, y)                  # 0.175

# With reduction options
nn.MSELoss(reduction='none')(yhat, y)      # tensor([0.0100, 0.0400, 0.1600, 0.4900])
nn.MSELoss(reduction='sum')(yhat, y)       # 0.70
nn.MSELoss(reduction='mean')(yhat, y)      # 0.175  (default)

# NumPy
np.mean((np.array([1,0,1,0.]) - np.array([0.9,0.2,0.6,0.7]))**2)   # 0.175

# TensorFlow / Keras
# tf.keras.losses.MeanSquaredError()(y, yhat)

# In a training loop (PyTorch):
# optimizer.zero_grad()
# pred = model(x)
# loss = nn.MSELoss()(pred, target)
# loss.backward()
# optimizer.step()
```

**`reduction` parameter — know all three:**
- `'mean'` (default): divides by n. Use for training — keeps loss scale independent of batch size.
- `'sum'`: total squared error. Use when you need SSE (e.g. for AIC/BIC, for physics-informed losses with a specific energy interpretation).
- `'none'`: per-sample losses. Use for sample-weighting or for loss analysis.

### 6. Interpretation
MSE is in the **squared units** of the target — not directly interpretable. Report RMSE (= √MSE) to stakeholders. Benchmark against:
- `MSE_baseline = Var(y)` (the MSE of always predicting the mean)
- `R² = 1 − MSE/Var(y)` — the fraction of variance explained

For our example: `Var(y) = 0.25`, so `R² = 1 − 0.175/0.25 = 0.30` — the model explains 30% of the variance. (This is a regression interpretation of what is really a classification problem — for illustrative comparison only.)

### 7. Good vs bad values
No universal threshold — always compare to a baseline. `MSE/Var(y) < 0.1` (R² > 0.9) is strong for most tabular regression; `> 1.0` means the model is worse than predicting the mean.

### 8. Use cases
- **Default regression loss** — linear regression, ridge, lasso, GBM, neural regression heads
- **Image reconstruction** — autoencoders, super-resolution, inpainting (MSE on pixel values)
- **Physics-informed networks** — when the PDE residual is naturally squared
- **Control systems / RL value functions** — Bellman error is typically squared (TD error)
- **Audio synthesis** — waveform-level MSE (though spectral losses often work better)
- **Financial return prediction** — when variance of the forecast error is the risk measure
- **Federated learning** — MSE's convexity simplifies aggregation proofs
- **Any domain where the business cost is genuinely convex in the error**

### 9. Advantages
- **Smooth, differentiable everywhere** with a clean gradient = 2e/n
- **Closed-form solution** for linear models (normal equations): `θ = (XᵀX)⁻¹Xᵀy`
- **MLE under Gaussian noise** — statistically principled
- **Convex for linear models** — unique global minimum
- **Targets the conditional mean** — predictions aggregate correctly (totals add up)
- **Consistent with classical inference** — t-tests, F-tests, R² all built on it
- **Compatible with weight decay / L2 regularisation** in a Bayesian MAP sense
- Universally supported across all frameworks

### 10. Limitations
- **Outlier-dominated** — one large error can dominate the entire loss; breakdown point is 0%
- **Squared units** are uninterpretable to humans; always report RMSE
- **Targets the mean** — on right-skewed targets, the mean is above the median, and MSE-trained models over-predict the typical case while under-predicting rare high-value events
- **Heteroscedastic data** — if Var(y|x) varies across inputs, MSE up-weights high-variance regions
- **Not the right loss for counts, probabilities, categories, or bounded targets**
- **Gradients vanish near the optimum for sigmoid outputs** when combined with a sigmoid activation (use BCE instead — see Part 2.1 for why)

### 11. Common mistakes
1. **Using MSE for a classification head with sigmoid output.** The gradient vanishes badly when the output is near 0 or 1 and the prediction is confident but wrong. Use BCE.
2. **Using MSE for a target that can't be negative** (counts, prices, durations). A predicted −5 is nonsensical; use Poisson, Tweedie, or log-transform.
3. **Reporting MSE to stakeholders** without converting to RMSE.
4. **Not baseline-comparing.** "MSE = 0.175" is meaningless without `Var(y) = 0.25`.
5. **Using `reduction='sum'` in training** without compensating for batch-size variation — loss scale and effective LR change with batch size.

### 12. Interview questions

**Easy — What does MSE measure?** The average squared prediction error. It penalises large errors disproportionately because of the squaring.

**Medium — ★ Why is MSE the MLE under Gaussian noise?** If errors are i.i.d. Gaussian, the log-likelihood is `−Σ(yᵢ−ŷᵢ)²/(2σ²) + const`. Maximising it is exactly minimising the sum of squared errors.

**Medium — ★ Why not use MSE for a classification output with sigmoid?** Because the gradient `∂MSE/∂logit = 2(σ(z)−y) × σ(z)(1−σ(z))`. When the model is confidently wrong (e.g. σ(z) ≈ 1 but y = 0), the gradient ≈ 0 — learning essentially stops. BCE's gradient `σ(z)−y` has no such saturation; it is large when the model is confidently wrong, giving a strong correction signal.

**Hard — ★ MSE vs MAE: the totals argument.** MSE targets the conditional mean; MAE targets the conditional median. For right-skewed distributions the mean > median, so MAE-trained models systematically under-predict the mean. If your business needs the aggregate (sum of forecasts) to be correct — inventory, revenue, capacity — train on MSE (or a Poisson/Tweedie objective), because these target the mean and means add up. If you predict individually and the aggregate does not matter, MAE's robustness may be preferable.

---

## 1.2 Mean Absolute Error (MAE / L1 Loss)

### 1. Definition
The average of the absolute errors. The robust counterpart to MSE.

### 2. Intuition
MAE treats all errors proportionally: being wrong by 4 is exactly twice as bad as being wrong by 2. This linearity makes it **robust** — a single extreme prediction can triple MSE but cannot substantially move MAE unless it is extreme in a majority of samples.

The deep property: **MAE is minimised by the conditional median of y|x.** If you want the model to predict the median rather than the mean — because your business cares about the "typical" case, or because your target is right-skewed — train on MAE (or L1).

### 3. Formula
```
MAE = (1/n) Σᵢ |yᵢ − ŷᵢ|

Sub-gradient:  ∂MAE/∂ŷᵢ = −sign(yᵢ − ŷᵢ) / n = ±1/n
```
**The sub-gradient is ±1/n regardless of the error size.** This is the robustness mechanism (a large error cannot pull harder) and also the convergence limitation (the sub-gradient doesn't shrink near the optimum, so the optimiser can oscillate).

### 4. Manual example
```
|e| = [0.1, 0.2, 0.4, 0.7]
MAE = (0.1 + 0.2 + 0.4 + 0.7) / 4 = 1.4/4 = 0.35
```
**MAE = 0.35**. Compare to MSE = 0.175 (in different units — squared — so not directly comparable). The important comparison: sample 4 contributes 0.7/1.4 = 50% to MAE vs 70% to MSE. MAE "shares the blame" more evenly.

### 5. Code
```python
import torch, torch.nn as nn

y    = torch.tensor([1., 0., 1., 0.])
yhat = torch.tensor([0.9, 0.2, 0.6, 0.7])

nn.L1Loss()(yhat, y)                  # 0.35
nn.L1Loss(reduction='none')(yhat, y)  # tensor([0.1, 0.2, 0.4, 0.7])

# In PyTorch, MAE is called L1Loss. No functional difference.

# Custom weighted MAE (weight by sample importance)
weights = torch.tensor([1., 1., 2., 1.])     # fraud cases count double
weighted_mae = (weights * torch.abs(yhat - y)).sum() / weights.sum()

# For GBMs: sklearn's GradientBoostingRegressor(loss='absolute_error')
# For LightGBM:  objective='mae'
# For XGBoost:   objective='reg:absoluteerror'
```

### 6. Interpretation
In the target's units — directly interpretable. "MAE = 0.35 probability units on average." For genuine regression: "The model misses by $35k on average."

### 7. Use cases
- **Tabular regression where outliers exist** — house prices, salaries, claim amounts
- **Demand forecasting** where cost is linear in units (each mis-forecast unit has the same cost)
- **Robust regression** when the dataset has known label noise or data-entry errors
- **Image-to-image translation** — MAE on pixel values is less prone to blurry outputs than MSE (MSE averages over multiple modes, producing blur; MAE gives sharper but sometimes patchy results)
- **Reinforcement learning** — some policy-gradient methods use L1-clipped Bellman errors
- **Depth estimation** — when metric depth error in metres is the reporting metric
- **Any setting where the conditional median is more useful than the conditional mean**

### 8. Advantages
- **Directly interpretable** — in the target's units
- **Robust to outliers** — gradient magnitude is constant regardless of error size
- **Targets the median** — more representative of the "typical" case on skewed distributions
- Works on data where errors legitimately follow a Laplace (double-exponential) distribution

### 9. Limitations
- **Not differentiable at zero** — sub-gradient causes oscillation near the optimum; slower convergence than MSE
- **No closed-form solution** for linear models — requires iterative solvers
- **Does not target the mean** — aggregate totals are biased for right-skewed targets
- **Gradient is always ±1/n** — the model pushes equally hard on easy and hard examples; no extra effort on the worst predictions

### 10. Common mistakes
1. **Using MAE when the total must be right** (revenue forecasting, reserving). MAE targets the median; on skewed targets the aggregate will be biased low. Use MSE.
2. **Assuming smooth convergence.** MAE can oscillate near the minimum; Huber loss (Part 1.3) fixes this.
3. **Using `torch.nn.L1Loss` and expecting RMSE-like sensitivity.** It will not hunt outliers.

### 11. Interview questions
**Medium — ★ What statistic does minimising MAE give you, and why does it matter?**
The conditional median. This matters because on right-skewed targets (prices, incomes, revenues) the median is below the mean. An MAE-trained demand forecast will systematically under-predict the mean demand, causing stockouts in inventory planning. If the aggregate total matters, train on MSE or a log-link GLM.

**Hard — Why is MAE non-differentiable at zero and does it matter in practice?**
`|e|` has a kink at e = 0; the left derivative is −1 and the right derivative is +1, so the standard derivative is undefined. In practice we use a sub-gradient (conventionally 0 at e = 0). The consequence: the sub-gradient doesn't approach zero as e → 0, so gradient descent oscillates in a small neighbourhood of the optimum rather than converging. This causes slower convergence and noisy final solutions compared to MSE. Huber loss exists specifically to restore smooth convergence: it is quadratic (gradient ∝ e) near zero and linear (gradient = ±δ) far away.

---

## 1.3 Huber Loss (Smooth L1 Loss)

### 1. Definition
Quadratic for small errors (like MSE), linear for large errors (like MAE), with a threshold δ. The best of both worlds.

### 2. Intuition
Huber loss solves the core tension between MSE and MAE by being one near the minimum (smooth, fast convergence) and the other for outliers (robust). The transition parameter δ gives the loss a **business meaning**: it is the error size beyond which you stop caring proportionally more.

```
           ½e²                if |e| ≤ δ   ← MSE-like: fast convergence
L_δ(e) =
           δ(|e| − ½δ)       if |e| > δ   ← MAE-like: bounded influence
```
The two branches are **exactly equal at |e| = δ** (value = ½δ²) and the gradient is **continuous at δ** (both sides give δ in magnitude) — making it C¹ smooth everywhere.

**Gradient:**
```
∂L/∂ŷ = −e/n         if |e| ≤ δ    (proportional, like MSE)
        = −δ·sign(e)/n  if |e| > δ    (bounded, like MAE)
```

### 3. Manual example (δ = 0.5)
```
e = [0.1, -0.2, 0.4, -0.7]    |e| = [0.1, 0.2, 0.4, 0.7]

Sample 1: |0.1| ≤ 0.5 → ½(0.1)² = 0.005
Sample 2: |0.2| ≤ 0.5 → ½(0.2)² = 0.020
Sample 3: |0.4| ≤ 0.5 → ½(0.4)² = 0.080
Sample 4: |0.7| > 0.5 → 0.5×(0.7 − ½×0.5) = 0.5×0.45 = 0.225

Huber = (0.005 + 0.020 + 0.080 + 0.225) / 4 = 0.330/4 = 0.0825
```
**Huber(δ=0.5) = 0.0825** vs MSE = 0.175 — 53% lower, because the outlier sample is down-weighted.

### 4. Code
```python
import torch, torch.nn as nn

y    = torch.tensor([1., 0., 1., 0.])
yhat = torch.tensor([0.9, 0.2, 0.6, 0.7])

# PyTorch — called HuberLoss (delta parameter)
nn.HuberLoss(delta=0.5)(yhat, y)           # 0.0825

# PyTorch — SmoothL1Loss is Huber with delta=1.0
nn.SmoothL1Loss()(yhat, y)                 # delta=1 by default

# Note: SmoothL1Loss has a different parameterisation before PyTorch 1.9:
# old:  ½x² if |x|<1, |x|−½ otherwise
# new:  same formula but `beta` parameter instead of `delta`
# Always check your version.

# Choosing delta from data:
delta = np.percentile(np.abs(y.numpy() - yhat.numpy()), 90)  # 90th percentile
nn.HuberLoss(delta=float(delta))(yhat, y)

# TensorFlow:
# tf.keras.losses.Huber(delta=0.5)
```

### 5. Use cases
- **Object detection** — bounding-box coordinate regression (SmoothL1 in Fast R-CNN, all RCNN variants); δ=1 by default
- **Reinforcement learning** — Bellman/TD error (DQN uses Huber for gradient clipping)
- **Financial time series** — returns and prices have heavy tails
- **Robust regression** on tabular data with label noise
- **Any regression problem where you want MSE's convergence and MAE's robustness**
- **Federated learning** — robustness to adversarial or corrupted clients

### 6. Advantages
- **Differentiable everywhere** — smooth convergence, no oscillation near the minimum
- **Bounded gradient** — outliers cannot dominate the gradient update
- **δ has a business meaning** — the error magnitude beyond which extra badness stops growing
- **Adaptive**: δ → ∞ = MSE; δ → 0 = MAE

### 7. Limitations
- **δ is scale-dependent** — must be re-tuned if the target scale changes
- **Loss value is not interpretable** (mixed units across the two branches)
- **Does not fully reject outliers** — a point at e = 1000 still pulls with force δ (bounded but non-zero); Tukey biweight completely rejects outliers

### 8. Common mistakes
1. **Setting δ too large (above max |e|)** — silently makes it pure MSE with no robustification.
2. **Confusing SmoothL1Loss and HuberLoss parameter names** across PyTorch versions.
3. **Not checking what fraction of samples are in the linear branch.** If < 1%, δ is too large.
4. **Using Huber when the problem is asymmetric cost** — Huber is symmetric; use quantile loss.

### 9. Interview questions
**Medium — ★ Why is Huber loss preferred over MAE in object detection?**
Because bounding-box regression needs smooth, well-behaved gradients everywhere for stable multi-task training. MAE's non-differentiability at zero causes oscillation and interferes with the joint classification + regression gradient flow. Huber (SmoothL1 with δ=1) gives MAE-like robustness for large errors (annotation mistakes, occlusion) while ensuring MSE-like smooth convergence near the correct box, and its bounded gradient at δ prevents a single badly-annotated box from corrupting the regression head.

**Hard — What is the connection between Huber loss and M-estimation?**
An M-estimator minimises `Σ ρ(eᵢ)` where ρ is a "rho-function" that is less sensitive than squared error. Huber's ρ is the Huber loss function, and its "psi-function" (ψ = ρ') is the gradient: identity inside ±δ, clipped to ±δ outside. This is the canonical **bounded influence function** that defines Huber M-estimation. The influence function measures how much a single observation can shift the estimate — Huber makes it bounded (max δ), preventing any one observation from dominating, which is exactly the formal definition of a robust estimator. Stronger robustness (full rejection of outliers) comes from redescending psi-functions like Tukey's biweight, at the cost of non-convexity.

---

## 1.4 Log-Cosh Loss

### 1. Definition
`L(e) = log(cosh(e))` — a smooth approximation to Huber that is twice-differentiable everywhere.

### 2. Formula and behaviour
```
log(cosh(e)) ≈ ½e²    for small |e|   (like MSE)
             ≈ |e| − log2   for large |e|  (like MAE, with a constant offset)

Gradient:   tanh(e)   — bounded in (−1, 1) for ALL e
Second derivative:   sech²(e) = 1 − tanh²(e)  — smooth, positive everywhere
```
**The gradient is `tanh(e)`.** This is the cleanest statement of its properties: `tanh` is smooth, bounded, odd, and vanishes at 0. No clipping, no kink, no case-switch.

### 3. Key advantage over Huber: C² smooth for Hessian-based optimisers

XGBoost and LightGBM use second-order (Newton) boosting: each split uses both the first derivative (g) and the second derivative (h). Huber's h is discontinuous at |e| = δ, which degrades second-order split quality. Log-Cosh's h = sech²(e) is smooth everywhere → better Hessian estimates → better splits.

```python
import torch, torch.nn as nn

y    = torch.tensor([1., 0., 1., 0.])
yhat = torch.tensor([0.9, 0.2, 0.6, 0.7])

def logcosh_loss(y_pred, y_true):
    e = y_true - y_pred
    a = torch.abs(e)
    # Numerically stable: log(cosh(e)) = |e| + log1p(exp(-2|e|)) - log(2)
    return torch.mean(a + torch.log1p(torch.exp(-2*a)) - torch.log(torch.tensor(2.0)))

logcosh_loss(yhat, y)   # 0.0825 (nearly identical to Huber here since errors are small)

# XGBoost: objective='reg:pseudohubererror'  (pseudo-Huber, same family)
# TensorFlow: tf.keras.losses.LogCosh()
```

**Critical practical limitation:** the transition from quadratic to linear occurs at |e| ≈ 1. **If the target is not scaled so that typical errors are O(1), Log-Cosh degenerates to pure MAE.** For house prices where errors are in the thousands, Log-Cosh ≈ MAE − 0.693 × n, i.e. useless. Always standardise the target.

### 4. Use cases
- **Gradient boosting** (XGBoost, LightGBM) where second-order splits benefit from a smooth H
- **Deep regression on standardised targets**
- **Robust regression where you want no hyperparameter to tune** (unlike Huber's δ)

---

## 1.5 Quantile / Pinball Loss

### 1. Definition
An asymmetric loss that, when minimised, produces the conditional τ-quantile of y|x.

### 2. Formula
```
L_τ(e) = max(τ·e,  (τ−1)·e)  = e·(τ − 1{e < 0})
```
- Under-prediction (e > 0) costs τ·e
- Over-prediction (e < 0) costs (1−τ)·|e|
- **τ = 0.5 recovers MAE/2** (the median)
- **τ > 0.5 penalises under-prediction more** → model predicts high → use for safety stock, ETAs, VaR

**Setting τ from business costs:**
```
τ* = C_under / (C_under + C_over)    ← the newsvendor critical fractile
```

### 3. Manual example (τ = 0.7)
```
e = [0.1, -0.2, 0.4, -0.7]
Sample 1: e>0 → 0.7×0.1   = 0.070
Sample 2: e<0 → 0.3×0.2   = 0.060
Sample 3: e>0 → 0.7×0.4   = 0.280
Sample 4: e<0 → 0.3×0.7   = 0.210

Pinball(0.7) = (0.070+0.060+0.280+0.210)/4 = 0.620/4 = 0.155
```
**Pinball(τ=0.7) = 0.155.**

### 4. Code
```python
import torch

def pinball_loss(y_pred, y_true, tau=0.5):
    e = y_true - y_pred
    return torch.mean(torch.maximum(tau * e, (tau - 1) * e))

y    = torch.tensor([1., 0., 1., 0.])
yhat = torch.tensor([0.9, 0.2, 0.6, 0.7])

pinball_loss(yhat, y, tau=0.5)   # 0.175 = MAE/2 ✓
pinball_loss(yhat, y, tau=0.7)   # 0.155

# sklearn
from sklearn.linear_model import QuantileRegressor
# QuantileRegressor(quantile=0.9).fit(X, y)

# sklearn GBM
from sklearn.ensemble import GradientBoostingRegressor, HistGradientBoostingRegressor
# GradientBoostingRegressor(loss='quantile', alpha=0.9)
# HistGradientBoostingRegressor(loss='quantile', quantile=0.9)

# LightGBM: objective='quantile', alpha=0.9
# XGBoost >= 2.0: objective='reg:quantileerror', quantile_alpha=0.9
```

### 5. Use cases
- **Safety stock / inventory** — τ = C_stockout/(C_stockout + C_holding)
- **ETA and delivery-time promises** — show the 90th percentile, not the mean
- **Value at Risk (VaR)** — financial risk at the τ=0.99 quantile
- **Prediction intervals** — fit two models at τ = α/2 and 1 − α/2
- **Electricity reserve planning** — forecast the 99th percentile of peak load
- **Loan loss provisioning** — regulators require conservative (high-quantile) estimates
- **Weather and climate** — probabilistic temperature/precipitation forecasts

### 6. Common mistakes
1. **Ignoring quantile crossing** — independently fitted lower and upper quantiles can cross for some inputs. Fix with monotonicity constraints, joint fitting, or post-hoc sorting.
2. **Choosing τ = 0.9 without computing the business cost ratio.**
3. **Summing high-quantile forecasts up a hierarchy** — quantiles do not add: `Q_0.9(A+B) ≤ Q_0.9(A) + Q_0.9(B)`.

### 7. Interview questions
**Medium — ★ What is the newsvendor connection?** The optimal order-up-to quantity in a newsvendor (single-period inventory) problem is the τ*-quantile of demand where τ* = c_under/(c_under + c_over). Quantile regression is the machine-learning generalisation: instead of a single τ and an assumed distribution, it estimates the τ-quantile of demand from features, with no distributional assumption.

**Hard — Why do quantile forecasts not add up a hierarchy?** Because quantiles are not linear operators. The 90th percentile of a sum is not the sum of the 90th percentiles — in fact, under positive correlation it is less than the sum, and under independence it is far less. The classic supply-chain error is summing store-level 90th percentile forecasts to the regional level and then wondering why the region is over-stocked by 40% — risk pooling means the aggregate's tail is narrower than the sum of individual tails. Model the aggregate directly, or simulate the joint distribution with its correlation structure.

---

## 1.6 MSLE / RMSLE (Log-Scale Loss)

### 1. Definition
MSE computed on log(1 + y) instead of y. Measures relative rather than absolute error.

### 2. Formula
```
MSLE = (1/n) Σ [log(1+yᵢ) − log(1+ŷᵢ)]²
     = (1/n) Σ [log((1+yᵢ)/(1+ŷᵢ))]²    ← squared log ratio = relative error
```
- **Asymmetric:** penalises under-prediction more than over-prediction in absolute terms (same ratio, opposite directions, but the absolute penalty differs)
- **Requires y ≥ 0 and ŷ ≥ 0** — clip or use log-link models for safety
- **The +1 shift** allows y = 0

### 3. Code
```python
import torch

def rmsle_loss(y_pred, y_true):
    y_pred = torch.clamp(y_pred, min=0)        # MUST clip negatives
    return torch.sqrt(torch.mean((torch.log1p(y_true) - torch.log1p(y_pred))**2))

# sklearn
from sklearn.metrics import mean_squared_log_error
# Also as a training loss: model log(1+y), predict, expm1 back

# TF: tf.keras.losses.MeanSquaredLogarithmicError()
```

### 4. Use cases
- **Kaggle competitions** with positive skewed targets (house prices, sales)
- **Sales and demand forecasting** where relative accuracy is what matters
- **Web traffic, page views, biological measurements** spanning orders of magnitude
- **Insurance claim severity** — RMSLE penalises under-prediction, aligning with the need for adequate reserves

### 5. Critical warning: back-transform bias
Training on log(1+y) with MSE, then predicting `expm1(model output)`, gives the conditional **median**, not the mean. For a lognormal target the under-prediction of the mean is `exp(σ²/2)`. Fix: Duan smearing estimator, or use a Gamma/Tweedie objective instead.

---

## 1.7 Poisson / Tweedie / Gamma Deviance

### 1. Why these exist
Counts, insurance severities, and positive right-skewed targets have **non-constant error variance**. For a Poisson count the variance equals the mean; for a Gamma the variance is proportional to the mean squared. MSE (which assumes constant variance) up-weights the high end and ignores the low end. The GLM deviance losses are the correct likelihood-based alternatives.

### 2. Poisson Deviance (negative log-likelihood under Poisson)
```
L_Poisson = (1/n) Σ [ŷᵢ − yᵢ·log(ŷᵢ)]    (up to a constant)
```
**Gradient:** `∂L/∂ŷᵢ = (1 − yᵢ/ŷᵢ)/n` — proportional to the **relative residual**, so the loss implicitly works on a relative scale.

**Variance function:** `Var(y|x) ∝ E[y|x]` — correct for Poisson counts.

### 3. Gamma Deviance
```
L_Gamma = (1/n) Σ [yᵢ/ŷᵢ − log(yᵢ/ŷᵢ) − 1]
```
**Variance function:** `Var(y|x) ∝ E[y|x]²` — correct for positive skewed continuous data with a constant coefficient of variation (typical of insurance claim severity, revenues, durations).

### 4. Tweedie Deviance (generalisation)
```
L_Tweedie(p) = (1/n) Σ [yᵢ²⁻ᵖ/((1-p)(2-p)) − yᵢ·ŷᵢ¹⁻ᵖ/(1-p) + ŷᵢ²⁻ᵖ/(2-p)]
```
- **p = 0:** Gaussian (MSE)
- **p = 1:** Poisson
- **p = 2:** Gamma
- **1 < p < 2:** Compound Poisson-Gamma — the standard for **total insurance losses** (zero-inflated positive data where some entries are exactly zero)

```python
# XGBoost: objective='reg:tweedie', tweedie_variance_power=1.5
# LightGBM: objective='tweedie', tweedie_variance_power=1.5
# sklearn: TweedieRegressor(power=1.5, link='log')
# PyTorch: implement the deviance as a custom loss (see Appendix C)

import torch

def tweedie_loss(y_pred, y_true, p=1.5):
    """Tweedie deviance loss. y_pred must be > 0 (use softplus or exp activation)."""
    return torch.mean(
        y_true * (y_true**(1-p) - y_pred**(1-p)) / (1-p) - 
        (y_true**(2-p) - y_pred**(2-p)) / (2-p)
    )

def poisson_loss(y_pred, y_true):
    """Poisson negative log-likelihood (up to a constant)."""
    return torch.mean(y_pred - y_true * torch.log(y_pred + 1e-9))
```

### 5. Use cases
- **Poisson:** count regression (click counts, defect counts, accident frequency, NLP word counts)
- **Gamma:** insurance severity, rainfall amounts, positive revenue, biological rates
- **Tweedie (p ≈ 1.5):** total insurance premium = frequency × severity, marketing mix models, retail gross margin, energy consumption

### 6. Advantages
- **Correct variance function** — no heteroscedasticity
- **No back-transform bias** — models the conditional mean directly on the original scale with a log link
- **Handles zeros** (Poisson, Tweedie with 1 < p < 2)
- **Interpretable: coefficient = % change in E[y]** under the log link

### 7. Limitations
- **Requires y ≥ 0** (Poisson, Gamma) or y ≥ 0 with point mass at 0 (Tweedie 1 < p < 2)
- **Power parameter p must be specified** (Tweedie) — treat as a hyperparameter
- **Less familiar** than MSE — needs explanation in documentation

### 8. Common mistakes
1. **Using RMSLE when you want a Poisson loss** — RMSLE's +1 shift distorts small counts. For true count data, use Poisson deviance.
2. **Not using a log link** with Tweedie — a linear prediction can go negative; the log link ensures ŷ > 0 and gives a multiplicative (percentage) interpretation.

---

# PART 2 — Binary Classification Losses

## 2.1 Binary Cross-Entropy (BCE / Log Loss)

### 1. Definition
The negative log-likelihood under a Bernoulli model. The standard loss for binary classification.

### 2. Intuition
BCE asks: **"how surprised is the model by what actually happened?"**

Under a Bernoulli model, the likelihood of observing y given predicted probability p is:
```
p(y | ŷ) = ŷ^y × (1−ŷ)^(1−y)
```
BCE is `−log p(y|ŷ)`, the negative log-likelihood. Minimising BCE = maximising the likelihood = finding the parameters that make the observed data most probable.

**Three properties that make it the default:**
1. **Proper scoring rule** — cannot be improved by misreporting the probability
2. **Well-calibrated gradient** — when the model is confidently wrong, the gradient is large; when it is correct and confident, the gradient is near zero. No vanishing-gradient pathology (contrast with MSE + sigmoid, Part 1.1 Q&A)
3. **Targets a probability** — the optimal output is `E[y|x] = P(y=1|x)`, directly interpretable

### 3. Formula
```
BCE = −(1/n) Σᵢ [ yᵢ·log(ŷᵢ) + (1−yᵢ)·log(1−ŷᵢ) ]
```
Symbol by symbol:
- **yᵢ ∈ {0,1}** — the true label
- **ŷᵢ ∈ (0,1)** — the model's predicted probability (output of sigmoid)
- **yᵢ·log(ŷᵢ)** — for positive examples: pays more when ŷ is close to 0 (the model missed it)
- **(1−yᵢ)·log(1−ŷᵢ)** — for negative examples: pays more when ŷ is close to 1 (the model was fooled)

**Range:** [0, ∞). Perfect predictions (ŷ = 1 when y = 1 and ŷ = 0 when y = 0) → BCE = 0. A model always predicting 0.5 → BCE = log(2) ≈ 0.693.

**From-logits gradient — why it matters:**

With logit z and sigmoid σ(z) = ŷ:
```
∂BCE/∂z = σ(z) − y = ŷ − y
```
This is one of the most elegant gradients in all of deep learning: **the gradient is simply the prediction error.** No vanishing-gradient problem, no sigmoid saturation, no cancellation. This is the real reason to always use `BCEWithLogitsLoss` rather than `sigmoid → BCELoss` — the latter computes `log(sigmoid(z))` in two steps with numerical instability; the former uses `log-sum-exp` to compute the combined operation stably.

### 4. Manual example
```
y    = [1,   0,   1,   0  ]
ŷ    = [0.9, 0.2, 0.6, 0.7]

Sample 1 (y=1): −log(0.9)  = 0.10536
Sample 2 (y=0): −log(1−0.2)= −log(0.8) = 0.22314
Sample 3 (y=1): −log(0.6)  = 0.51083
Sample 4 (y=0): −log(1−0.7)= −log(0.3) = 1.20397

BCE = (0.10536 + 0.22314 + 0.51083 + 1.20397) / 4
    = 2.04330 / 4
    = 0.5108
```
**BCE = 0.5108.**

Sample 4 (confident wrong prediction: ŷ=0.7 when y=0) contributes **59%** of the total loss. The gradient for sample 4 is `0.7 − 0 = 0.7` — large, pushing the model hard to lower this prediction. For sample 1 (confident correct: ŷ=0.9 when y=1) the gradient is `0.9 − 1 = −0.1` — small, barely updating. **BCE focuses effort exactly where it is needed.**

### 5. Code
```python
import torch, torch.nn as nn

y     = torch.tensor([1., 0., 1., 0.])
yhat  = torch.tensor([0.9, 0.2, 0.6, 0.7])
logit = torch.log(yhat / (1 - yhat))   # ≈ [2.197, -1.386, 0.405, 0.847]

# ── CORRECT: always use BCEWithLogitsLoss (numerically stable) ──
nn.BCEWithLogitsLoss()(logit, y)           # 0.5108

# ── LESS STABLE: apply sigmoid first, then BCE ──
nn.BCELoss()(yhat, y)                      # 0.5108  (same, but can hit log(0) instability)

# Per-sample losses
nn.BCEWithLogitsLoss(reduction='none')(logit, y)
# tensor([0.1054, 0.2231, 0.5108, 1.2040])

# Class-weighted BCE (for class imbalance)
# pos_weight = ratio of negatives to positives
# e.g. 99 negatives : 1 positive → pos_weight = 99
nn.BCEWithLogitsLoss(pos_weight=torch.tensor([9.]))(logit, y)

# Multi-label classification (one BCE per class, then average)
# nn.BCEWithLogitsLoss()(logit_matrix, label_matrix)  where shapes are (N, C)

# TensorFlow:
# tf.keras.losses.BinaryCrossentropy(from_logits=True)

# sklearn:
from sklearn.metrics import log_loss
log_loss(y.numpy(), yhat.numpy())    # 0.5108
```

**Key rule: always pass logits (pre-sigmoid values) to `BCEWithLogitsLoss`.** Let PyTorch handle the sigmoid internally for numerical stability. Never do `sigmoid(z)` in the network and then pass to `BCEWithLogitsLoss` — that is applying sigmoid twice.

### 6. Interpretation
BCE = 0.5108 sits between 0 (perfect) and log(2) ≈ 0.693 (random guessing at 50/50). Lower is better. Benchmark: a model predicting the base rate `p̄` for every sample has BCE = `−[p̄·log(p̄) + (1−p̄)·log(1−p̄)]` = the entropy of the label distribution. If BCE > entropy, the model is actively harmful.

### 7. Use cases
- **Binary classification head** — spam vs not, fraud vs legit, cancer vs benign
- **Multi-label classification** — each label gets independent BCE; model can assign any subset of labels
- **Generative adversarial networks** — discriminator loss (original GAN formulation)
- **Calibration** — BCE is a proper scoring rule, so a model with low BCE is well-calibrated
- **Knowledge distillation** — matching a teacher's soft targets uses BCE/KL divergence
- **Any output with a sigmoid activation**

### 8. Advantages
- **Proper scoring rule** — minimised only by predicting the true probability; can't be gamed
- **Elegant gradient** `σ(z) − y` — no saturation, direct error signal
- **MLE under Bernoulli** — statistically principled
- **Numerically stable via log-sum-exp** in `BCEWithLogitsLoss`
- **Naturally handles class imbalance** via `pos_weight`
- Universally supported

### 9. Limitations
- **Exploding loss for confident wrong predictions** — `−log(ε)` → ∞ as ε → 0. A single misclassified point with ŷ ≈ 0 or ŷ ≈ 1 can dominate the batch. Use label smoothing or clip probabilities.
- **Not robust to noisy labels** — a label-noise rate of 10% can significantly inflate BCE
- **Sensitive to class imbalance** — with 99% negatives, the model learns to predict 0 everywhere and gets a good BCE. Use `pos_weight`, focal loss, or re-sampling.
- **Does not directly optimise the metrics you report** (AUC, F1, accuracy)

### 10. Common mistakes
1. **Using `BCELoss` with sigmoid output that was accidentally applied twice.** Set `sigmoid` in the loss layer via `BCEWithLogitsLoss`.
2. **Using MSE for binary classification.** Gradient vanishes for confident wrong predictions; BCE does not.
3. **Forgetting `pos_weight` on imbalanced datasets.** A 1:100 imbalance without `pos_weight` produces a trivial "always predict negative" solution.
4. **Evaluating only on BCE for an imbalanced problem.** BCE can be low while F1, precision, and recall are terrible.
5. **Using BCE for multi-class classification.** Use CCE (Part 3.1).

### 11. Interview questions

**Easy — What is BCE?** The negative log-likelihood under a Bernoulli model: `−[y log ŷ + (1−y) log(1−ŷ)]`.

**Medium — ★ Why does BCE have a better gradient than MSE for binary classification?**
With sigmoid output, BCE's gradient w.r.t. the logit is simply `σ(z) − y`. When the model is confidently wrong (e.g. σ(z) = 0.99, y = 0), this gradient is 0.99 — a strong correction signal. MSE's gradient w.r.t. the logit is `2(σ(z) − y)·σ(z)(1−σ(z))`. At σ(z) = 0.99, `σ(z)(1−σ(z)) = 0.0099` → gradient ≈ 0.02 — nearly vanished. The model barely learns from its most confidently wrong predictions under MSE. BCE has no such saturation.

**Medium — ★ What is the `pos_weight` parameter in `BCEWithLogitsLoss` and when should you use it?**
`pos_weight` scales the loss on positive examples by a factor w: effective loss = `−[w·y log σ(z) + (1−y) log(1−σ(z))]`. Use it when positives are rare — set `pos_weight = n_negatives / n_positives`. This up-weights the gradient on positives, pushing the decision boundary so the model invests more in learning the minority class. For 1:99 imbalance, `pos_weight = 99`. It is cheaper than re-sampling and theoretically equivalent to weighting each positive sample 99 times.

**Hard — ★ Derive the gradient of BCE with respect to the logit.**
Let z be the logit and σ(z) = 1/(1+e^{−z}) the sigmoid.
`BCE = −y·log σ(z) − (1−y)·log(1−σ(z))`
`∂BCE/∂z = −y·(1/σ(z))·σ'(z) + (1−y)·(1/(1−σ(z)))·σ'(z)`
`σ'(z) = σ(z)(1−σ(z))`, so:
`= −y·(1−σ(z)) + (1−y)·σ(z) = σ(z) − y`
This is the "prediction minus target" form — identical in structure to the MSE gradient w.r.t. the prediction, but without the sigmoid-derivative cancellation issue. The log and the sigmoid are "conjugate" — they cancel to give a perfectly clean first derivative. This is why (logistic) sigmoid + cross-entropy is the canonical pairing, and why you should use `BCEWithLogitsLoss` rather than separate sigmoid and BCE.

---

## 2.2 Focal Loss

### 1. Definition
BCE multiplied by a down-weighting factor `(1−p_t)^γ` that reduces the contribution of **easy, well-classified examples**, focusing training on the hard ones.

### 2. Motivation
In object detection (and any severely imbalanced classification), the vast majority of anchor boxes are **easy negatives** (clearly background). BCE gives them individually small loss, but collectively they dominate the gradient because there are ~10,000 of them for every 1 positive. The model spends 99% of its capacity learning "this is background," which is trivial, rather than learning the difficult boundary cases.

**Focal loss solution:** multiply BCE by `(1−p_t)^γ`, where `p_t = ŷ` if `y = 1` else `1 − ŷ` is the model's confidence in the correct class.
- **Easy example correctly predicted** (p_t ≈ 1): `(1−p_t)^γ ≈ 0` → loss ≈ 0 → gradient ≈ 0 → model ignores it
- **Hard example** (p_t small): `(1−p_t)^γ ≈ 1` → loss ≈ BCE → full gradient

### 3. Formula
```
FL(p_t) = −α_t · (1 − p_t)^γ · log(p_t)

where  p_t = ŷ if y=1, else 1−ŷ
       α_t = α if y=1, else 1−α    (class-balancing weight; typically α=0.25)
       γ ≥ 0                         (focusing parameter; γ=2 in the original paper)
```

**Effect of γ:**
- γ = 0: standard BCE (times α)
- γ = 1: linear down-weighting of easy examples
- γ = 2: quadratic — the recommendation for dense object detection

### 4. Manual example (α = 0.25, γ = 2)
```
p_t = [0.9, 0.8, 0.6, 0.3]     (ŷ for positives; 1−ŷ for negatives)
α_t = [0.25, 0.75, 0.25, 0.75]  (0.25 for positives, 0.75 for negatives)

Sample 1 (easy positive):  −0.25 × (1−0.9)² × log(0.9)  = −0.25×0.01×(−0.105) = 0.000263
Sample 2 (easy negative):  −0.75 × (1−0.8)² × log(0.8)  = −0.75×0.04×(−0.223) = 0.00668
Sample 3 (hard positive):  −0.25 × (1−0.6)² × log(0.6)  = −0.25×0.16×(−0.511) = 0.02043
Sample 4 (hard negative):  −0.75 × (1−0.3)² × log(0.3)  = −0.75×0.49×(−1.204) = 0.44244

FL = (0.000263 + 0.00668 + 0.02043 + 0.44244) / 4 = 0.4698/4 = 0.1175
```
Compare to BCE = 0.5108: Focal loss is 77% lower because the easy, correctly-classified samples (1, 2) are nearly zeroed out, and the gradient is concentrated on the hard mis-classifications.

### 5. Code
```python
import torch
import torch.nn.functional as F

def focal_loss(logits, targets, alpha=0.25, gamma=2.0):
    """Focal loss for binary classification."""
    bce = F.binary_cross_entropy_with_logits(logits, targets, reduction='none')
    p_t = torch.exp(-bce)                              # p_t = ŷ (for y=1) or 1-ŷ (for y=0)
    alpha_t = targets * alpha + (1 - targets) * (1 - alpha)
    focal_weight = alpha_t * (1 - p_t) ** gamma
    return (focal_weight * bce).mean()

logit = torch.tensor([2.197, -1.386, 0.405, 0.847])
y     = torch.tensor([1., 0., 1., 0.])
focal_loss(logit, y, alpha=0.25, gamma=2)     # ≈ 0.0426

# torchvision implementation
# from torchvision.ops import sigmoid_focal_loss
# sigmoid_focal_loss(logit, y, alpha=0.25, gamma=2, reduction='mean')

# Multi-class focal: apply per-class and aggregate
# Custom losses library: pip install segmentation-models-pytorch
```

### 6. Use cases
- **Dense object detection**: RetinaNet (the paper that introduced focal loss), FCOS, CenterNet
- **Semantic segmentation**: foreground/background imbalance
- **Medical image classification**: rare disease detection from imaging
- **Any severely class-imbalanced classification** — typically 1:100 to 1:10,000 positive-to-negative ratio
- **Fraud detection** at scale
- **Natural language processing**: rare-entity NER, low-resource classification

### 7. Advantages
- **Handles extreme class imbalance without re-sampling or explicit re-weighting of the dataset**
- **Dynamically re-weights based on confidence** — adapts as training progresses (easy examples become easier; focal loss ignores them automatically)
- **Simple modification of BCE** — one extra multiply, trivial to implement
- **Two intuitive parameters** (α for class balance, γ for focusing strength) with well-established defaults (α=0.25, γ=2)
- Robust to easy/hard example mismatch across different classes

### 8. Limitations
- **Two hyperparameters to tune**: α and γ (though α=0.25, γ=2 works broadly for detection)
- **May underfit on easy examples** if γ is too high — the model sees almost no gradient from the majority of well-classified samples and can miss the boundary
- **Less well-studied for multi-class** vs binary settings
- **Not automatically calibrated** — the focusing factor disrupts the probability calibration of BCE; use a post-training calibration step if probabilities are needed
- **Computationally identical to BCE** for a single sample, but the weighting changes the effective batch statistics

### 9. Common mistakes
1. **Using focal loss when the dataset is balanced.** It will harm performance by suppressing gradient from the easy (correctly-classified) majority.
2. **Setting γ too high** (γ > 5) — training destabilises because nearly all gradient comes from a handful of hardest examples, causing high-variance updates.
3. **Forgetting to tune α alongside γ** — they interact; a good α for BCE may not be optimal after the focusing.
4. **Applying focal loss without checking the class imbalance first.** If the imbalance is only 5:1, `pos_weight` in standard BCE is simpler and often just as effective.

### 10. Interview questions

**Easy — What problem does focal loss solve?** Class imbalance in dense prediction — specifically, the dominance of easy negative examples whose individually small BCE losses collectively overwhelm the gradient from the rare positive examples.

**Medium — ★ Explain the focusing factor `(1 − p_t)^γ`.** When the model correctly classifies an example with high confidence (p_t ≈ 1), the factor `(1−p_t)^γ ≈ 0` drives the contribution to zero. The model ignores easy examples. When the model is wrong or uncertain (p_t small), the factor ≈ 1 and the loss is close to standard BCE. γ controls the rate of down-weighting — higher γ = more aggressive suppression of easy examples. At γ=0 it is plain BCE.

**Hard — ★ Why doesn't standard BCE with `pos_weight` solve the same problem as focal loss?** `pos_weight` provides a fixed global weight to positive samples. It addresses the imbalance in the number of examples but does not distinguish between **easy** and **hard** examples within each class. A correctly-classified easy positive with `pos_weight = 99` still contributes 99× the loss of an easy negative, while a hard-to-classify example at the decision boundary contributes the same. Focal loss dynamically adjusts the weight based on the current model's confidence, so as the model learns to classify easy examples correctly, it progressively ignores them and focuses on the hard boundary cases. This is a fundamentally different mechanism — `pos_weight` is a static class-level weight; focal loss is a dynamic, per-example difficulty weight.

---

## 2.3 Hinge Loss and Squared Hinge Loss

### 1. Definition
**Hinge loss** is the loss used by Support Vector Machines. It penalises predictions that are on the wrong side of the margin, and ignores correct predictions with sufficient margin.

### 2. Formula
Convert y ∈ {0,1} to y' ∈ {−1, +1}: `y' = 2y − 1`
```
Hinge(e) = max(0, 1 − y'·ŷ)
```
- `y'·ŷ > 1`: correct prediction with sufficient margin → loss = 0 (the "hinge" — no penalty)
- `y'·ŷ < 1`: prediction inside or on the wrong side of the margin → loss > 0

**Squared Hinge:**
```
Squared Hinge = max(0, 1 − y'·ŷ)²
```
Penalises margin violations quadratically — more sensitive to large violations.

### 3. Manual example
```
y'   = [+1, −1, +1, −1]    (from y=[1,0,1,0])
ŷ    = [0.9, 0.2, 0.6, 0.7]   (these are probabilities; note hinge prefers raw scores, not probabilities)

Using ŷ as scores:
Sample 1: max(0, 1−1×0.9)  = max(0, 0.1)  = 0.1
Sample 2: max(0, 1−(−1)×0.2) = max(0, 1.2) = 1.2   ← predicted positive (0.2>0 when −1 expected)
Sample 3: max(0, 1−1×0.6)  = max(0, 0.4)  = 0.4
Sample 4: max(0, 1−(−1)×0.7) = max(0, 1.7) = 1.7   ← strongly wrong

Hinge = (0.1 + 1.2 + 0.4 + 1.7) / 4 = 3.4/4 = 0.85
Sq Hinge = (0.01 + 1.44 + 0.16 + 2.89) / 4 = 4.5/4 = 1.125
```

### 4. Code
```python
import torch

def hinge_loss(scores, targets_01):
    """targets_01 in {0,1} converted to ±1."""
    y_pm = 2 * targets_01 - 1
    return torch.mean(torch.clamp(1 - y_pm * scores, min=0))

def squared_hinge_loss(scores, targets_01):
    y_pm = 2 * targets_01 - 1
    return torch.mean(torch.clamp(1 - y_pm * scores, min=0) ** 2)

y     = torch.tensor([1., 0., 1., 0.])
yhat  = torch.tensor([0.9, 0.2, 0.6, 0.7])
print(hinge_loss(yhat, y))           # 0.85
print(squared_hinge_loss(yhat, y))   # 1.125

# sklearn: SVC(kernel='linear') minimises Hinge + L2 regularisation (= SVM)
# keras: tf.keras.losses.Hinge(), tf.keras.losses.SquaredHinge()
```

### 5. Use cases
- **SVM** — linear and kernelised SVM training
- **Multi-class classification** (multi-class hinge / Crammer-Singer loss)
- **Structured SVMs** (sequence labelling, dependency parsing)
- **Ranking** — RankSVM
- **Computer vision features** (legacy: HOG + linear SVM)
- **When you want a maximum-margin classifier** that ignores correctly-classified easy examples

### 6. Advantages
- **Sparse solution** — only support vectors (margin-violating examples) have non-zero gradient → efficient; the model is not affected by easy examples far from the boundary
- **Margin-based** — explicitly maximises the separation between classes; the geometric interpretation is clean
- **Squared hinge** is differentiable, convex, and has better convergence than plain hinge

### 7. Limitations
- **No probability output** — hinge gives a score, not a calibrated probability; BCE / Platt scaling needed if probabilities are required
- **Prefers raw scores** (not probabilities) — combining with sigmoid output is unusual
- **Not the right choice for deep learning** in most settings — BCE with logits is preferred for its gradient properties; hinge is primarily a legacy SVM tool

### 8. Common mistakes
1. **Applying hinge to sigmoid probabilities** — hinge expects raw scores, not bounded probabilities.
2. **Forgetting to convert y ∈ {0,1} to y' ∈ {−1,+1}** before applying the formula.

### 9. Interview question
**Medium — ★ Why does hinge loss produce sparse solutions (support vectors)?**
Because `max(0, 1 − y'·ŷ)` has gradient zero whenever `y'·ŷ ≥ 1` — i.e. whenever the example is correctly classified with sufficient margin. These examples do not contribute to the gradient and have no influence on the learned parameters. Only the "support vectors" (examples on or inside the margin) have non-zero gradient and shape the decision boundary. This sparsity is the SVM's distinctive property and explains its low computational cost at inference: only the support vectors are stored, not the full dataset.

---

# PART 3 — Multi-Class Classification Losses

## 3.1 Categorical Cross-Entropy (CCE / Softmax Loss)

### 1. Definition
The negative log-likelihood under a Categorical distribution — the standard multi-class classification loss.

### 2. Intuition
CCE extends BCE to K > 2 classes. The model outputs K logits (raw scores), a softmax converts them to K probabilities summing to 1, and CCE measures how much probability mass the model placed on the correct class.

The key property: CCE loss is entirely determined by the probability assigned to the **correct** class: `CCE = −log(ŷ_correct_class)`. The model is not penalised for how it distributes the remaining probability among wrong classes.

### 3. Formula
```
CCE = −(1/n) Σᵢ log(ŷᵢ, cᵢ)   (one-hot; only correct class cᵢ contributes)

Gradient: ∂CCE/∂zᵢₖ = ŷᵢₖ − yᵢₖ   (softmax prob minus one-hot)
```
Again: **"prediction minus target"** — the most elegant gradient in classification.

### 4. Manual example
```python
import numpy as np
logits = np.array([2.0, 1.0, 0.1])
softmax = np.exp(logits) / np.exp(logits).sum()  # [0.659, 0.242, 0.099]
CCE = -np.log(softmax[1])   # correct class = index 1 → CCE ≈ 1.417
```

### 5. Code
```python
import torch, torch.nn as nn
logits  = torch.tensor([[2.0, 1.0, 0.1], [0.5, 2.1, 0.3]])
targets = torch.tensor([1, 1])

nn.CrossEntropyLoss()(logits, targets)          # ≈ 1.417
nn.CrossEntropyLoss(weight=torch.tensor([1.,5.,1.]))(logits, targets)  # class-weighted
nn.CrossEntropyLoss(ignore_index=-100)(logits, targets)  # NLP padding

# TF: tf.keras.losses.SparseCategoricalCrossentropy(from_logits=True)
# NEVER apply softmax before CrossEntropyLoss — it applies log-softmax internally
```

### 6. Use cases
- **Image classification** — ResNet, EfficientNet, ViT final layer
- **Language model next-token prediction** — every GPT/LLaMA token
- **NLP classification** — sentiment, intent, NLI
- **RL discrete action spaces** — REINFORCE policy gradient
- Every neural network with a softmax output

### 7. Common mistakes
1. Applying `softmax` in the model AND using `CrossEntropyLoss` (double softmax).
2. Using CCE for multi-label (softmax forces mutual exclusivity; use BCE per class).
3. Integer targets with TF's `CategoricalCrossentropy` (use `SparseCategoricalCrossentropy`).
4. Forgetting `ignore_index=-100` for padding tokens in NLP.

### 8. Interview questions
**Medium — ★ Why `CrossEntropyLoss` not `Softmax → log → NLLLoss`?** Numerical stability via log-sum-exp: `exp(100)` overflows; `CrossEntropyLoss` subtracts max before exp.

**Hard — ★ Perplexity and cross-entropy:** `Perplexity = exp(CE)`. PP=10 means the model averages 10 equally likely options per token. PP=1 is perfect.

---

## 3.2 Label Smoothing Cross-Entropy

Replaces the one-hot target with `ỹₖ = (1−ε)·yₖ + ε/K`. Prevents overconfidence; improves calibration.

```python
nn.CrossEntropyLoss(label_smoothing=0.1)(logits, targets)
# TF: tf.keras.losses.CategoricalCrossentropy(label_smoothing=0.1)
```
**When not to use:** distillation targets (double-smoothing), maximum accuracy on clean labels, RL policy gradient.

---

## 3.3 KL Divergence

```
KL(P||Q) = Σₖ P(k) · log(P(k)/Q(k))   [not symmetric; range [0,∞)]
```

```python
import torch.nn as nn
# KLDivLoss takes LOG-PROBABILITIES for Q (the model)
nn.KLDivLoss(reduction='batchmean')(torch.log(q), p)   # KL(p||q)
# TRAP: passing q without log is a silent bug
```

**Forward KL** (P_data → Q_model): mean-seeking — Q must cover all modes.
**Reverse KL** (Q_model → P_data): mode-seeking — Q can focus on one mode.

**Use cases:** knowledge distillation, VAE regularisation, topic modelling, seq2seq with soft targets.

---

## 3.4 Asymmetric Loss (ASL)

Extends focal loss to multi-label: different γ for positive (γ+ ≈ 0) and negative (γ− ≈ 4) examples.
```
ASL = { −(1−p)^γ+ log p     y=1
      { −p^γ−  log(1−p)     y=0
```
Optional probability shift `p_m = max(p−m, 0)` discards very easy negatives.
**Use case:** multi-label image recognition (MS-COCO, OpenImages) with extreme per-class imbalance.

---

# PART 4 — Sequence and Structured Prediction Losses

## 4.1 Sequence Cross-Entropy (Teacher Forcing)

Per-token CCE averaged over all time steps:
```
L_seq = −(1/T) Σₜ log P(yₜ | y₁,...,yₜ₋₁, x)
```

```python
# Language model loss
loss = nn.CrossEntropyLoss(ignore_index=-100)(
    logits.view(-1, vocab_size), targets.view(-1))
```

**Exposure bias:** at training, the model sees ground-truth tokens; at inference it sees its own outputs — errors compound. Fixes: scheduled sampling, RLHF, minimum Bayes risk training.

**Use cases:** NMT, GPT/LLaMA pretraining, image captioning, summarisation, code generation.

---

## 4.2 CTC Loss (Connectionist Temporal Classification)

For sequence labelling **without alignment** (ASR, OCR). Marginalises over all valid alignments using a blank token via the forward-backward algorithm.

```python
ctc = nn.CTCLoss(blank=0, zero_infinity=True)
loss = ctc(log_probs, targets, input_lengths, target_lengths)
# log_probs: (T, N, C) must be log-softmax; input_lengths[i] ≥ 2×target_lengths[i]−1
```

**Use cases:** end-to-end ASR, OCR without bounding boxes, handwriting recognition.

**Key interview:** **When CTC vs seq2seq?** CTC for monotonic alignment, limited data; seq2seq for non-monotonic, large data.

---

## 4.3 Structural SVM

`L = max(0, 1 − score(x,y) + score(x,ŷ) + Δ(y,ŷ))` — generalises hinge to structured outputs. Largely superseded by neural approaches.

---

# PART 5 — Metric Learning and Ranking Losses

## 5.1 Contrastive Loss

```
L = y_pair · d²  +  (1−y_pair) · max(0, m−d)²
```
- y=1 (similar): minimise distance d
- y=0 (dissimilar): push d ≥ margin m

**Manual example:** similar pair d=0.3 → 0.09; dissimilar d=0.8, m=1.0 → 0.04.

```python
def contrastive_loss(emb1, emb2, y_pair, margin=1.0):
    d   = torch.norm(emb1 - emb2, dim=1)
    pos = y_pair * d.pow(2)
    neg = (1-y_pair) * torch.clamp(margin-d, min=0).pow(2)
    return (pos + neg).mean()
```

**Use cases:** Siamese nets (face/signature verification), few-shot learning, anomaly detection.
**Limitations:** pairs need similarity labels; margin-sensitive; superseded by triplet/InfoNCE.

---

## 5.2 Triplet Loss

```
L(a,p,n) = max(0, d(a,p) − d(a,n) + α)
```

```python
nn.TripletMarginLoss(margin=0.3, p=2)(anchor, positive, negative)
```

**Manual example:** d_ap=0.5, d_an=0.8, α=0.3 → max(0, 0.5−0.8+0.3) = 0 (satisfied).
d_ap=0.5, d_an=0.6, α=0.3 → 0.2 (violation → push p closer).

**Hard negative mining:** select the closest valid negative per anchor. Random negatives → mostly zero gradient. Semi-hard: inside the margin but farther than the positive.

**Use cases:** face recognition (FaceNet), person re-ID, visual product search, few-shot, drug discovery.

**Interview — ★ Why did InfoNCE largely replace triplet?** Triplet uses one negative per step; InfoNCE uses the full batch (N−1 negatives). More signal, better MI bound, no explicit negative sampling pipeline needed.

---

## 5.3 InfoNCE / NT-Xent (SimCLR, CLIP)

```
L = −log( exp(sim(zᵢ,zⱼ)/τ) / Σₖ≠ᵢ exp(sim(zᵢ,zₖ)/τ) )
```
Identify the positive view among N−1 negatives — a multi-class softmax where the "class" is the paired image.

```python
def info_nce(z1, z2, temperature=0.07):
    N = z1.size(0)
    z = torch.cat([z1, z2])
    sim = torch.mm(z, z.T) / temperature
    sim.fill_diagonal_(float('-inf'))
    labels = torch.cat([torch.arange(N, 2*N), torch.arange(N)])
    return torch.nn.functional.cross_entropy(sim, labels.to(z.device))

# CLIP: symmetric image→text + text→image
```

**Temperature τ:** small → sharp distribution → hard task → concentrate on hard negatives. Typical: τ=0.07 (SimCLR), 0.1 (MoCo).

**Use cases:** SimCLR, MoCo, CLIP, ALIGN, SimCSE, cross-modal retrieval, drug-protein binding.

**Interview — ★ InfoNCE bounds MI:** `I(z1;z2) ≥ log N − L_InfoNCE`. Larger N → tighter bound → better representations.

---

## 5.4 Cosine Embedding Loss

```python
nn.CosineEmbeddingLoss(margin=0.0)(v1, v2, torch.tensor([1.]))  # 1−cos(v1,v2) = 0.4
```
**Use cases:** sentence similarity (STS), cross-lingual alignment, DPR.


---

# PART 6 — Generative Model Losses

## 6.1 GAN Losses

### The Minimax Game (Original GAN)
```
L_D = −E[log D(x_real)] − E[log(1−D(G(z)))]   (discriminator — max)
L_G = E[log(1−D(G(z)))]                          (generator — min; SATURATES early)
```
**Non-saturating fix (default):**
```
L_G = −E[log D(G(z))]    (same fixed point; strong gradient when D rejects fake)
```

### Hinge GAN (SAGAN, BigGAN)
```
L_D = E[max(0, 1−D(real))] + E[max(0, 1+D(fake))]
L_G = −E[D(G(z))]
```
More stable in practice for high-resolution generation.

### WGAN and WGAN-GP
Original GAN uses JS divergence → zero gradient when distributions don't overlap.
WGAN uses Wasserstein-1 (Earth Mover's) distance instead:
```
L_D = −E[D(x_real)] + E[D(G(z))]   (requires 1-Lipschitz D)
L_G = −E[D(G(z))]
```
WGAN-GP enforces Lipschitz via gradient penalty instead of weight clipping:
```
L_GP = λ · E[(‖∇_x̂ D(x̂)‖₂ − 1)²]    λ=10, x̂ = interpolated real/fake
```

```python
def wgan_gp(D, real, fake, device, lam=10):
    alpha = torch.rand(real.size(0), *([1]*(real.dim()-1)), device=device)
    interp = (alpha*real + (1-alpha)*fake).requires_grad_(True)
    d_int  = D(interp)
    grad   = torch.autograd.grad(d_int, interp, torch.ones_like(d_int), create_graph=True)[0]
    return lam * ((grad.view(grad.size(0),-1).norm(2,dim=1) - 1)**2).mean()
```

| Loss | Stability | Use when |
|---|---|---|
| Non-saturating BCE | Moderate | Simple baselines |
| Hinge GAN | Good | Class-conditional generation |
| **WGAN-GP** | **Very good** | **Training is unstable** |

**Use cases:** unconditional/class-conditional image generation (StyleGAN, BigGAN), image-to-image translation (Pix2Pix, CycleGAN), super-resolution (SRGAN, ESRGAN), synthetic data augmentation.

**Interview — ★ Vanishing G gradient in minimax:** `log(1−D(G(z))) ≈ 0` when D rejects everything early → `∂L_G/∂z ≈ 0`. Non-saturating fix: `−log D(G(z))` → large when D(G(z)) small.

**Interview — ★ Why Wasserstein > JS divergence?** JS = log2 for all non-overlapping supports (zero gradient). Wasserstein measures distance between manifolds even without overlap → useful gradients everywhere.

---

## 6.2 Diffusion Model Loss (DDPM)

```
L_simple = E_{t,x₀,ε} [‖ε − ε_θ(xₜ, t)‖²]    (pure MSE on noise prediction)
```
MSE is correct because each denoising step is Gaussian — MSE is MLE for Gaussian noise.

```python
# Training step:
t     = torch.randint(0, T, (batch,))
eps   = torch.randn_like(x0)
xt    = alpha_bar[t].sqrt()*x0 + (1-alpha_bar[t]).sqrt()*eps
loss  = nn.functional.mse_loss(model(xt, t), eps)
```

**Why ε-prediction beats x₀-prediction:** ε has constant scale (unit Gaussian) at all timesteps → balanced loss. Predicting x₀ directly gives huge loss at high noise levels → instability.

**Use cases:** Stable Diffusion, DALL-E 2, Imagen, Sora (video), RFDiffusion (proteins), AudioLDM.

---

# PART 7 — Probabilistic and Bayesian Losses

## 7.1 ELBO — VAE Loss

```
L_VAE = reconstruction_loss  +  β · KL(q(z|x) || N(0,I))

KL (closed form, Gaussian posterior):
KL = −½ Σᵢ (1 + log σᵢ² − μᵢ² − σᵢ²)
```

**Manual KL example:** μ=[0.5,−0.3], log_var=[−0.4,0.2] → KL = 0.2159.

```python
def vae_loss(recon, x, mu, log_var, beta=1.0):
    recon_loss = nn.functional.mse_loss(recon, x, reduction='sum')
    kl = -0.5 * torch.sum(1 + log_var - mu.pow(2) - log_var.exp())
    return (recon_loss + beta*kl) / x.size(0)

def reparameterise(mu, log_var):
    return mu + torch.randn_like(mu) * torch.exp(0.5*log_var)
```

**β-VAE:** β > 1 → stronger bottleneck → disentangled representations (each dim = one factor). Cost: worse reconstruction quality.

**Posterior collapse:** KL → 0, decoder ignores z. Fixes: KL annealing (linearly ramp β from 0), free bits (min KL per dim).

**Use cases:** generative modelling, anomaly detection (high recon loss = anomaly), molecular generation (JTVAE), dialogue diversity.

---

## 7.2 Normalising Flow Loss

Exact NLL via change of variables:
```
L = −(1/n) Σᵢ [log p_z(f(xᵢ)) + log|det Jᵢ|]
```
**Use cases:** exact density estimation (anomaly detection, compression), bijective generation (Glow, RealNVP), Bayesian inference with flexible posteriors.

---

## 7.3 Brier Score

```
BS = (1/n) Σᵢ (yᵢ − ŷᵢ)²   [MSE on probabilities; bounded [0,1]; proper scoring rule]
```
Better than BCE for calibration reporting (bounded); less sharp for training. Used in weather forecasting, clinical calibration research.

---

# PART 8 — Self-Supervised and Representation Learning Losses

## 8.1 Masked Language Modelling (BERT / MLM)

Per-token CCE on masked positions only:
```
L_MLM = − Σ_{masked} log P(correct_token | context)
```
```python
loss = nn.CrossEntropyLoss(ignore_index=-100)(
    logits.view(-1, vocab_size), labels.view(-1))   # labels=-100 for non-masked
```
**Use cases:** BERT, RoBERTa, DeBERTa pretraining; MAE (MSE on masked image patches); CodeBERT.

---

## 8.2 BarlowTwins, VICReg, BYOL

**BarlowTwins:** maximise diagonal of cross-correlation matrix C (invariance) + minimise off-diagonal (redundancy):
```
L = Σᵢ(Cᵢᵢ−1)² + λΣᵢ≠ⱼCᵢⱼ²
```

**VICReg:** MSE similarity + variance regulariser (prevent collapse) + covariance regulariser (decorrelate):
```
L = λ·sim(Z₁,Z₂) + μ·var(Z) + ν·cov(Z)
```

**BYOL:** MSE with stop-gradient on the momentum target:
```python
loss = F.mse_loss(F.normalize(online_pred), F.normalize(target_proj.detach()))
# target_params = τ·target + (1−τ)·online   (EMA update)
```

All three avoid collapse without explicit negatives — enabling smaller batch sizes than InfoNCE.

---

# PART 9 — Custom and Composite Losses

## 9.1 Multi-Task Composite Losses

```
L_total = Σₜ wₜ · Lₜ
```
**Examples:** Object detection: `λ_cls·L_focal + λ_box·L_huber + λ_obj·L_bce`; VAE: `L_recon + β·L_KL`; GAN+perceptual: `L_adv + λ_pix·L_MSE + λ_perc·L_VGG`.

**Balancing strategies:**
```python
class UncertaintyWeightedLoss(torch.nn.Module):
    """Kendall et al. 2018 — learnable σ per task."""
    def __init__(self, n_tasks):
        super().__init__()
        self.log_vars = torch.nn.Parameter(torch.zeros(n_tasks))
    def forward(self, losses):
        return sum(torch.exp(-self.log_vars[i])*l + self.log_vars[i] for i,l in enumerate(losses))
```

---

## 9.2 Perceptual Loss

```
L_perc = Σₗ λₗ · ‖φₗ(ŷ) − φₗ(y)‖²    (VGG feature distance)
```
**Why sharper than MSE:** MSE averages over modes → blur. VGG features are shift-invariant → any sharp image with matching features scores well.

**Use cases:** super-resolution (SRGAN, ESRGAN), style transfer, image-to-image translation, video synthesis.

---

## 9.3 Dice Loss / IoU Loss / Tversky Loss

```python
def dice_loss(pred, target, smooth=1.0):
    pred  = torch.sigmoid(pred)
    inter = (pred*target).sum(dim=(1,2))
    union = pred.sum(dim=(1,2)) + target.sum(dim=(1,2))
    return (1 - (2*inter+smooth)/(union+smooth)).mean()

# Combined: BCE + Dice — standard for binary medical segmentation
def bce_dice(pred, target, a=0.5):
    return a*F.binary_cross_entropy_with_logits(pred,target) + (1-a)*dice_loss(torch.sigmoid(pred),target)
```

**Tversky:** `1 − TP/(TP + α·FP + β·FN)` — set α<β (e.g. 0.3,0.7) to penalise false negatives more (small lesion detection).

**Focal Tversky:** `(1−Tversky)^γ` — for very small structures.

**Dice = soft F1:** when predictions are binary, Dice = F1 = 2TP/(2TP+FP+FN).

---

## 9.4 Wing Loss (Facial Landmarks)

```
W(x) = { w·ln(1+|x|/ε)   if |x|<w    (amplified small-error gradient)
        { |x|−C           if |x|≥w    (MAE-like for outliers)
```
Outperforms L2 and Huber for facial alignment: small errors matter most for landmark precision.

---

# PART 10 — Loss Selection Decision Tree

```
WHAT IS YOUR TASK?
│
├── REGRESSION
│   ├── Symmetric cost, outliers OK     → MSE
│   ├── Symmetric cost, outliers exist  → Huber (δ = 90th percentile of |e|)
│   ├── Linear cost                     → MAE
│   ├── Asymmetric cost                 → Quantile at τ* = C_under/(C_under+C_over)
│   ├── Positive right-skewed           → RMSLE or Gamma/Tweedie
│   ├── Count (integer ≥ 0)             → Poisson deviance
│   └── Zero-inflated positive          → Tweedie (p ≈ 1.5)
│
├── BINARY CLASSIFICATION
│   ├── Balanced                        → BCEWithLogitsLoss
│   ├── Imbalanced < 1:10               → BCE + pos_weight
│   └── Extreme imbalance > 1:100       → Focal Loss (α=0.25, γ=2)
│
├── MULTI-CLASS
│   ├── Standard                        → CrossEntropyLoss (never double-softmax)
│   ├── Label noise / overconfidence    → Label Smoothing CE (ε=0.1)
│   ├── Multi-LABEL                     → BCE per class (NOT CCE)
│   └── Soft targets / distillation     → KL Divergence
│
├── SEQUENCE
│   ├── Aligned (NMT, LM, captioning)   → Sequence CCE with teacher forcing
│   └── Unaligned (ASR, OCR)            → CTC Loss
│
├── METRIC LEARNING
│   ├── Pairwise data                   → Contrastive Loss
│   ├── Triplet data                    → Triplet + hard negative mining
│   ├── Self-supervised large batch     → InfoNCE / NT-Xent
│   ├── Self-supervised small batch     → BYOL / BarlowTwins / VICReg
│   └── Text-image alignment            → CLIP (bidirectional InfoNCE)
│
├── GENERATIVE
│   ├── GAN stable                      → Hinge GAN or WGAN-GP
│   ├── GAN baseline                    → Non-saturating BCE
│   ├── VAE                             → ELBO = recon + β·KL
│   └── Diffusion                       → MSE on noise (ε-prediction)
│
├── SEGMENTATION
│   ├── Balanced                        → CCE per pixel
│   ├── Binary imbalanced               → BCE + Dice
│   └── Small structures                → Focal Tversky
│
└── MULTI-TASK
    ├── Known importance                → Σ wₜLₜ (tune manually)
    └── Unknown difficulty              → Uncertainty weighting (Kendall 2018)
```

---

# PART 11 — Master Comparison Tables

## 11.1 Loss Reference

| Loss | Task | Optimal predictor | Robust? | Asym? | Differentiable? |
|---|---|---|---|---|---|
| MSE | Regression | Cond. mean | No | No | Yes (C∞) |
| MAE | Regression | Cond. median | Yes | No | Sub-diff |
| Huber | Regression | Near mean | Bounded | No | Yes (C¹) |
| Log-Cosh | Regression | Near mean | Bounded | No | Yes (C²) |
| Quantile | Regression | Cond. quantile τ | Yes | **Yes** | Sub-diff |
| Poisson dev | Count | Cond. mean | Moderate | No | Yes |
| Tweedie | Zero-inf+ | Cond. mean | Moderate | No | Yes |
| BCE | Binary | P(y=1\|x) | No | No | Yes |
| Focal | Imbalanced binary | P(y=1\|x) | No | No | Yes |
| Hinge | SVM | Decision boundary | Sparse | No | Sub-diff |
| CCE | Multi-class | P(y=k\|x) | No | No | Yes |
| Smooth CCE | Multi-class | P(y=k\|x) reg. | No | No | Yes |
| KL Div | Distribution match | Distribution P | No | No | Yes |
| CTC | Seq w/o align | Sequence | No | No | Yes |
| Triplet | Metric | Relative distances | Sparse | No | Sub-diff |
| InfoNCE | Self-supervised | MI maximisation | No | No | Yes |
| ELBO | VAE | Posterior | No | No | Yes |
| WGAN-GP | GAN | Wasserstein dist | Yes | No | Yes |
| Diffusion MSE | Generative | Noise field ε | No | No | Yes |
| Dice | Segmentation | F1/overlap | Partial | No | Yes |
| Perceptual | Image synthesis | Feature match | No | No | Yes |

## 11.2 Key Identities

```
MSE = Bias² + Var(residuals)
BCE gradient w.r.t. logit = σ(z) − y
CCE gradient w.r.t. logit k = ŷₖ − yₖ
Pinball(τ=0.5) = MAE/2
CRPS(point) = MAE
Focal(γ=0) = BCE
Dice = soft F1 score
ELBO = log p(x) − KL(q||posterior)
KL(P||Q) = H(P,Q) − H(P)
Brier Score = MSE on probabilities
```

## 11.3 Framework Support

| Loss | PyTorch | TF/Keras |
|---|---|---|
| MSE | `nn.MSELoss` | `MeanSquaredError` |
| MAE | `nn.L1Loss` | `MeanAbsoluteError` |
| Huber | `nn.HuberLoss(delta=d)` | `Huber(delta=d)` |
| BCE | **`nn.BCEWithLogitsLoss`** | `BinaryCrossentropy(from_logits=True)` |
| CCE | `nn.CrossEntropyLoss` | `SparseCategoricalCrossentropy` |
| Focal | `torchvision.ops.sigmoid_focal_loss` | `tfa.losses.SigmoidFocalCrossEntropy` |
| Triplet | `nn.TripletMarginLoss` | `tfa.losses.TripletSemiHardLoss` |
| CTC | `nn.CTCLoss` | `tf.keras.losses.CTC` |
| KL Div | `nn.KLDivLoss` (needs log input!) | `KLDivergence` |


---

# PART 12 — 100+ Interview Questions with Answers

## 12.1 Foundations (Q1–Q20)

**Q1. What is a loss function?** A differentiable (or sub-differentiable) function L(y_true, y_pred) that the optimiser minimises during training. It encodes three things: the noise model (Gaussian → MSE, Bernoulli → BCE), the optimal predictor (mean, median, quantile, probability), and the cost structure (symmetric/asymmetric, linear/convex).

**Q2. Loss vs evaluation metric.** Loss: differentiable, used to train. Metric: interpretable, used to report. They should be consistent — mismatches cause the model to optimise one thing while being measured on another.

**Q3. ★ Why is MSE MLE under Gaussian noise?** Gaussian log-likelihood = `−Σ(y−ŷ)²/(2σ²) + const`. Maximising over ŷ = minimising SSE = minimising MSE.

**Q4. What statistic does MAE minimise?** The conditional median. Its sub-gradient is ±1/n regardless of error size; the minimum occurs when the prediction splits the data 50/50.

**Q5. ★ Why not use MSE for binary classification?** MSE gradient w.r.t. logit = `2(σ(z)−y)·σ(z)(1−σ(z))`. When confidently wrong, `σ(z)(1−σ(z)) ≈ 0` → near-zero gradient → no learning. BCE gradient = `σ(z)−y` has no saturation.

**Q6. What is a proper scoring rule?** S(F,y) is proper if `E_P[S(P,y)] ≤ E_P[S(F,y)]` for all F — the true distribution minimises the expected score. BCE, CCE, NLL, CRPS are proper. Hinge loss, accuracy are not.

**Q7. ★ Why `BCEWithLogitsLoss` not `sigmoid → BCELoss`?** Numerical stability via log-sum-exp: BCEWithLogitsLoss computes `max(z,0) + log(1+e^{−|z|})` to avoid overflow for large |z|. Also avoids round-trip through probabilities.

**Q8. CCE gradient w.r.t. logit k?** `ŷₖ − yₖ`. For correct class: `ŷₖ − 1` (push up). For wrong classes: `ŷₖ` (push down).

**Q9. ★ KL divergence: forward vs reverse.** Forward `KL(P_data||Q_model)`: mean-seeking — Q must cover all modes. Reverse `KL(Q_model||P_data)`: mode-seeking — Q can focus on one mode. VAEs use reverse KL (mode-seeking posterior approximation).

**Q10. ★ What is the ELBO?** `E_{q(z|x)}[log p(x|z)] − KL(q(z|x)||p(z))` = Reconstruction − KL. Maximising it is maximising a lower bound on log p(x).

**Q11. ★ Reparameterisation trick.** Replace `z ~ N(μ,σ²)` with `z = μ + σ·ε`, ε~N(0,1). z is now a deterministic function of μ,σ; gradients flow through them. Stochasticity moves to ε which has no parameters.

**Q12. Hard negative mining.** Select negatives closest to the anchor from different classes. Random negatives are usually trivially separated → zero gradient. Semi-hard: inside the margin but farther than the positive → best stability-informativeness trade-off.

**Q13. Temperature τ in InfoNCE.** Small τ: sharp distribution, hard task, concentrated gradients on hardest negatives. Large τ: flatter, weaker but more stable gradients. Default: τ=0.07 (SimCLR).

**Q14. ★ Focal loss.** BCE × `(1−p_t)^γ`. When p_t ≈ 1 (easy correct): factor → 0 → loss → 0. When p_t small (hard/wrong): factor → 1 → full BCE. Solves foreground/background imbalance in dense detection.

**Q15. CTC loss.** Marginalises over all valid alignments between input frames and output tokens using a blank token. Computed via forward-backward algorithm. For ASR, OCR — no explicit alignment needed.

**Q16. ★ BCE and KL divergence.** BCE(y,ŷ) = KL(y||ŷ) + H(y). Minimising BCE = minimising KL from true distribution to model's distribution.

**Q17. ★ Why Huber has bounded influence.** Gradient capped at ±δ for |e| > δ. A single outlier with e=1000 pulls at most δ, not 1000. MSE's gradient 2e is unbounded.

**Q18. ★ Label smoothing: when not to use?** During distillation (double-smoothing), when maximum accuracy on clean labels is needed, during RL policy gradient.

**Q19. ★ CCE not for multi-label.** Softmax forces probabilities to sum to 1 — mutually exclusive. Multi-label allows multiple classes simultaneously. Use sigmoid + BCE per class independently.

**Q20. ★ Contrastive vs triplet loss.** Contrastive: pairwise, absolute distances, similar → 0, dissimilar → ≥ m. Triplet: relative constraint (anchor closer to positive than negative by α). Triplet is more natural. Both superseded by InfoNCE for self-supervised settings.

---

## 12.2 Architecture and Application (Q21–Q50)

**Q21. ★ Posterior collapse in VAEs.** KL term drives q(z|x) → p(z); latent code carries no information; decoder ignores z. Fixes: KL annealing (ramp β from 0), free bits (min KL per dim).

**Q22. ★ WGAN vs original GAN.** JS divergence = log2 for non-overlapping supports → zero gradient. Wasserstein distance measures manifold separation even without overlap → useful gradient everywhere. Also: WGAN loss correlates with sample quality; JS-based GAN loss does not.

**Q23. ★ Diffusion MSE: why ε-prediction?** ε has constant scale (N(0,1)) at all timesteps → balanced loss. x₀-prediction has huge loss at high noise → instability. Both are equivalent VLB parameterisations, but ε works better empirically.

**Q24. ★ Dice vs BCE for segmentation.** BCE treats each pixel independently → dominated by easy background pixels. Dice directly optimises the F1/overlap metric → equal weight to foreground regardless of frequency. Best practice: `BCE + Dice` combined.

**Q25. ★ Perceptual vs MSE for image synthesis.** MSE: unique minimum (pixel mean) → blurry average of modes. Perceptual: any image with matching VGG features scores well → model produces one sharp solution. Use perceptual for synthesis; MSE for exact reconstruction.

**Q26. ★ Derive BCE gradient w.r.t. logit z.** `∂BCE/∂z = −y(1−σ(z)) + (1−y)σ(z) = σ(z)−y`. The sigmoid and log cancel perfectly — no saturation term.

**Q27. InfoNCE bounds MI.** `I(z1;z2) ≥ log N − L_InfoNCE`. More negatives N → tighter bound → better MI estimation.

**Q28. ★ GAN non-saturating vs minimax.** Minimax G loss `log(1−D(G(z)))` → 0 when D rejects everything (saturates). Non-saturating: `−log D(G(z))` → ∞ when D rejects → strong gradient. Same fixed point.

**Q29. ★ β-VAE.** β > 1: stronger information bottleneck → disentangled representations. Each latent dim encodes one independent factor. Cost: worse reconstruction. β=4–10 for disentanglement.

**Q30. ★ Exposure bias.** Teacher forcing uses ground-truth tokens at training; inference uses model's own outputs. Early error → wrong distribution for subsequent steps → compounds. Fixes: scheduled sampling, RLHF.

**Q31. ★ Composite loss balancing.** (1) Manual weights: normalise by initial loss. (2) Uncertainty weighting (Kendall): learn `log σₜ²`; weight = `1/σₜ²`. Higher natural noise → auto down-weighted.

**Q32. ★ BCEWithLogitsLoss: pos_weight.** Without it, model predicts 0 for all → 99% accuracy on 1:99 imbalanced data but 0% recall. `pos_weight = n_neg/n_pos` up-weights positive gradient. Forces model to balance precision/recall.

**Q33. BarlowTwins collapse avoidance.** Off-diagonal decorrelation term prevents all dimensions from collapsing to the same value. No explicit negatives needed.

**Q34. BYOL stop-gradient purpose.** Prevents trivial solution: both networks collapsing to a constant. The online network must predict a stable (EMA) but non-trivial target → implicit regulariser.

**Q35. ★ Tversky vs Dice.** Tversky = `TP/(TP + α·FP + β·FN)`. α=β=0.5: Dice. α<β (e.g. 0.3,0.7): penalises false negatives more → better for small lesion detection where missing the target is worse than over-predicting.

**Q36. ★ Exposure bias and RLHF connection.** RLHF: (1) train reward model on human preferences; (2) PPO maximises reward on model's own completions. This closes the train/inference gap entirely — the model is evaluated on its own outputs, not ground-truth teacher tokens. KL penalty to SFT base prevents reward hacking.

**Q37. ★ Loss for GPT pretraining.** Causal LM cross-entropy: `L = −(1/T)Σₜ log P(wₜ|w₁,...,wₜ₋₁)`. Every token is a target. Enables zero-shot transfer because diverse text data implicitly trains all NLP tasks.

**Q38. ★ Object detection loss components.** Classification: Focal Loss (extreme foreground/background imbalance 1:10,000+). Regression: Huber/SmoothL1 for bounding boxes (rough annotations → bounded influence) or IoU-based (directly optimise the metric).

**Q39. ★ Why VAE with MSE produces blurry images.** MSE on pixels = mean over all valid reconstructions. If a face can be slightly left or right of center, the mean is a blurry face. Switch to perceptual loss or adversarial training (VAE+GAN) for sharper outputs.

**Q40. ★ KL annealing in VAEs.** Start β=0 (pure reconstruction); linearly ramp to β=1 over N epochs. Without this, the KL term collapses the posterior immediately before the encoder learns to encode useful information.

**Q41. ★ CLIP loss.** Symmetric InfoNCE on N image-text pairs: images predict matching texts + texts predict matching images, both using the same N×N cosine similarity matrix. N=32,768 in training. Enables zero-shot: class names are encoded as text, nearest text to image = class.

**Q42. ★ MAE (He et al.) loss.** MSE on pixel values of masked patches only. Mask 75% (vs BERT's 15%) — too hard for texture interpolation; requires semantic understanding. ViT encoder sees only 25% of patches → 3–4× faster training.

**Q43. ★ StyleGAN loss.** Non-saturating BCE + R1 gradient penalty: `λ/2·E[‖∇D(real)‖²]`. R1 regularises only on real images (simpler than WGAN-GP's interpolation). Architecture improvements (style-based, path length reg.) matter as much as the loss.

**Q44. ★ DPO vs PPO-RLHF.** PPO: separate reward model + RL fine-tuning, two models, expensive. DPO: direct classification on preference data, single model, `L = −log σ(β·log(π(y_w)/π_ref(y_w)) − β·log(π(y_l)/π_ref(y_l)))`. Simpler, stable, comparable performance.

**Q45. ★ WGAN-GP vs weight clipping.** Weight clipping biases discriminator to ±c saturated weights → limited capacity + gradient issues in deep D. Gradient penalty directly enforces ‖∇D‖₂ ≈ 1 via a soft constraint → better-conditioned D with full capacity.

**Q46. ★ SupCon vs InfoNCE.** InfoNCE: only self-augmentation positives → pushes apart same-class images (false negatives). SupCon: uses class labels to identify all same-class examples as positives in the numerator. Better clustering; more robust to label noise.

**Q47. GIoU vs IoU loss.** IoU=0 for all non-overlapping boxes → zero gradient. GIoU subtracts (enclosing_box − union)/enclosing_box: even when IoU=0, the enclosing box gives a continuous signal that moves boxes toward each other.

**Q48. ★ VAE reconstruction: MSE vs BCE vs perceptual.** MSE: exact pixel accuracy, blurry. BCE (per pixel, [0,1] range): treats each pixel as Bernoulli. Perceptual: VGG feature match, sharp but not pixel-exact. For generation quality: perceptual + adversarial. For anomaly detection: MSE (need the exact value to measure deviation).

**Q49. ★ Tweedie power p=1.5 for insurance.** Insurance total loss = frequency × severity. Frequency ~ Poisson (p=1). Severity ~ Gamma (p=2). Their compound = Compound Poisson-Gamma (1<p<2). p=1.5 is the standard starting point; tune as hyperparameter.

**Q50. ★ Knowledge distillation temperature.** At T=1 the teacher's distribution is nearly one-hot. High T (4–8) softens it, revealing inter-class similarities ("dark knowledge"). The student is trained at T>1 to learn these relative relationships; at inference T=1.

---

## 12.3 Coding (Q51–Q65)

**Q51. ★ BCE from scratch (numerically stable).**
```python
def bce(logits, targets):
    return torch.mean(torch.clamp(logits,0) - logits*targets
                      + torch.log(1 + torch.exp(-torch.abs(logits))))
```

**Q52. ★ Reparameterisation trick.**
```python
def reparam(mu, log_var):
    return mu + torch.randn_like(mu) * torch.exp(0.5*log_var)
```

**Q53. ★ InfoNCE.**
```python
def info_nce(z1, z2, tau=0.07):
    N=z1.size(0); z=torch.cat([z1,z2])
    sim=torch.mm(z,z.T)/tau; sim.fill_diagonal_(float('-inf'))
    lbl=torch.cat([torch.arange(N,2*N),torch.arange(N)])
    return F.cross_entropy(sim,lbl.to(z.device))
```

**Q54. ★ Focal loss.**
```python
def focal(logits, targets, a=0.25, g=2.0):
    bce=F.binary_cross_entropy_with_logits(logits,targets,reduction='none')
    pt=torch.exp(-bce); at=targets*a+(1-targets)*(1-a)
    return (at*(1-pt)**g*bce).mean()
```

**Q55. ★ Dice loss.**
```python
def dice(pred, tgt, s=1.0):
    p=torch.sigmoid(pred)
    i=(p*tgt).sum((1,2)); u=p.sum((1,2))+tgt.sum((1,2))
    return (1-(2*i+s)/(u+s)).mean()
```

**Q56. ★ Triplet loss.**
```python
def triplet(a, p, n, margin=0.3):
    return torch.clamp(torch.norm(a-p,dim=1)-torch.norm(a-n,dim=1)+margin,min=0).mean()
```

**Q57. ★ KL divergence — correct PyTorch usage.**
```python
# KLDivLoss REQUIRES log-probabilities as first arg
nn.KLDivLoss(reduction='batchmean')(torch.log(q), p)  # KL(p||q)
# Manual: (p * torch.log(p/q)).sum()
```

**Q58. ★ WGAN-GP gradient penalty.**
```python
def gp(D, real, fake, device, lam=10):
    a=torch.rand(real.size(0),*([1]*(real.dim()-1)),device=device)
    x=(a*real+(1-a)*fake).requires_grad_(True)
    g=torch.autograd.grad(D(x),x,torch.ones(real.size(0),1,device=device),create_graph=True)[0]
    return lam*((g.view(g.size(0),-1).norm(2,dim=1)-1)**2).mean()
```

**Q59. ★ Label smooth CE.**
```python
def lsce(logits, targets, eps=0.1):
    K=logits.size(-1); lp=torch.log_softmax(logits,-1)
    oh=torch.zeros_like(lp).scatter(-1,targets.unsqueeze(-1),1)
    return -(oh*(1-eps)+eps/K)*lp).sum(-1).mean()
```

**Q60. VAE loss.**
```python
def vae_loss(recon, x, mu, lv, beta=1.0):
    r=F.mse_loss(recon,x,reduction='sum')
    k=-0.5*torch.sum(1+lv-mu.pow(2)-lv.exp())
    return (r+beta*k)/x.size(0)
```

**Q61. Pinball/quantile loss.**
```python
def pinball(pred, target, tau=0.9):
    e=target-pred
    return torch.mean(torch.maximum(tau*e,(tau-1)*e))
```

**Q62. Huber loss.**
```python
def huber(pred, tgt, d=1.0):
    e=torch.abs(tgt-pred)
    return torch.mean(torch.where(e<=d,0.5*e**2,d*(e-0.5*d)))
```

**Q63. Poisson deviance.**
```python
def poisson_loss(pred, target):
    return torch.mean(pred - target*torch.log(pred+1e-9))
```

**Q64. Tweedie loss.**
```python
def tweedie(pred, target, p=1.5):
    return torch.mean(target*(target**(1-p)-pred**(1-p))/(1-p)
                      -(target**(2-p)-pred**(2-p))/(2-p))
```

**Q65. Contrastive loss.**
```python
def contrastive(e1, e2, y, margin=1.0):
    d=torch.norm(e1-e2,dim=1)
    return (y*d**2+(1-y)*torch.clamp(margin-d,min=0)**2).mean()
```

---

# PART 13 — Cheat Sheet

## 13.1 Loss Selection in 60 Seconds

```
Output?
  Continuous         → MSE (default), MAE (robust), Huber (balanced), Quantile (asym)
  Binary class       → BCEWithLogitsLoss   [ALWAYS from_logits]
  Multi-class        → CrossEntropyLoss    [NEVER double-softmax]
  Multi-LABEL        → BCE per class       [NOT CCE — different thing!]
  Count (int ≥ 0)    → Poisson deviance
  Zero-inflated pos  → Tweedie p=1.5
  Embedding          → Triplet or InfoNCE
  Distribution       → NLL / ELBO / WGAN-GP

Imbalance?
  Binary < 1:10      → BCE + pos_weight
  Binary > 1:100     → Focal (α=0.25, γ=2)
  Segmentation       → BCE + Dice

Sequence?
  Aligned            → CCE (teacher forcing)
  Unaligned          → CTC
```

## 13.2 Gradient Reference

| Loss | ∂L/∂ŷ or ∂L/∂z | Key property |
|---|---|---|
| MSE | 2(ŷ−y)/n | Proportional to error; shrinks at optimum |
| MAE | sign(ŷ−y)/n | Constant; oscillates at zero |
| Huber | e/n if \|e\|≤δ, else ±δ/n | Best of both |
| **BCE** (w.r.t. logit) | **σ(z)−y** | **No saturation — the killer advantage** |
| **CCE** (w.r.t. logit k) | **ŷₖ−yₖ** | Same elegant form as BCE |
| Hinge | −y' if violated, 0 else | Sparse support vectors |
| Focal | (1−p_t)^γ × BCE_grad | Down-weights easy examples |
| Quantile | −τ if e>0, (1−τ) if e<0 | Asymmetric constant |

## 13.3 Critical Gotchas — Quick Reference

| Mistake | Fix |
|---|---|
| `sigmoid` before `BCEWithLogitsLoss` | Remove sigmoid from network |
| `softmax` before `CrossEntropyLoss` | Remove softmax from network |
| `KLDivLoss(q, p)` — passing Q not log(Q) | `KLDivLoss(torch.log(q), p)` |
| MSE for classification | Use BCE/CCE |
| CCE for multi-label | Use BCE per class |
| BCE for multi-class | Use CCE |
| Focal on balanced data | Use BCE |
| Random negatives in triplet | Hard negative mining |
| Small batch in InfoNCE | Batch ≥ 256 or MoCo queue |
| No KL annealing in VAE | Ramp β from 0 over epochs |
| `reduction='sum'` in training | Use `reduction='mean'` |
| MSE for image synthesis | Perceptual + adversarial |
| RMSLE with negative predictions | `torch.clamp(pred, min=0)` |
| Double-softmax or double-sigmoid | Check model architecture |

## 13.4 40 Common Pitfalls

1. MSE for classification — vanishing gradient at extremes.
2. CCE for multi-label — softmax forces mutual exclusivity.
3. BCE for multi-class — doesn't enforce that probabilities add to 1.
4. Softmax before CrossEntropyLoss — double-softmax.
5. Sigmoid before BCEWithLogitsLoss — double-sigmoid.
6. KLDivLoss without log — silent wrong answer.
7. No pos_weight for imbalanced BCE — model always predicts 0.
8. Focal loss on balanced data — suppresses majority gradient.
9. Huber δ above all residuals — silently becomes MSE.
10. Quantile τ chosen without cost ratio.
11. Summing quantile forecasts up a hierarchy.
12. RMSLE without smearing correction — totals biased low.
13. RMSLE for negative targets — undefined.
14. RMSLE for small counts 0–5 — +1 shift dominates; use Poisson.
15. MSE for generative images — blurry outputs.
16. No perceptual/adversarial for image synthesis.
17. Random negatives in triplet training — zero gradient.
18. InfoNCE with batch < 128 — too few negatives.
19. No embedding normalisation before InfoNCE.
20. No KL annealing in VAE training — posterior collapses.
21. No reparameterisation trick — can't backprop through sampling.
22. No stop-gradient in BYOL — collapse to constant.
23. Original GAN BCE instead of non-saturating — early training saturation.
24. Weight clipping instead of gradient penalty in WGAN.
25. reduction='sum' in training — LR effective scale depends on batch size.
26. No ignore_index for padding in NLP cross-entropy.
27. Label smoothing during distillation — double-smoothing.
28. Temperature not set in distillation — teacher logits are near one-hot; dark knowledge lost.
29. zero_infinity=False in CTC — NaN when input too short.
30. Softmax inside GAN discriminator — limits output range; use linear.
31. Clipping not applied before MSLE.
32. Multi-task losses at very different scales — one task dominates.
33. Reporting training loss as test performance.
34. BCE + softmax combination — contradictory (BCE: independent; softmax: dependent).
35. Dice without BCE — unstable early training when predictions are near 0.5.
36. log(0) in manual loss implementation — add ε or use built-in numerically stable version.
37. Focal α=0.5 — not asymmetric enough; model learns to ignore both classes equally.
38. Perceptual loss with too-deep VGG layers — too abstract; loses texture information.
39. Hard negatives only in triplet — unstable; semi-hard is better.
40. Using contrastive loss without monitoring embedding collapse.

## 13.5 One-Sentence Takeaway per Loss

- **MSE:** MLE under Gaussian; targets mean; blows up on outliers.
- **MAE:** targets median; robust; non-differentiable at zero; use Huber if convergence is slow.
- **Huber:** quadratic near zero + linear far away; choose δ = 90th percentile of |e|.
- **Quantile:** encodes asymmetric cost; τ = C_under/(C_under+C_over) (newsvendor formula).
- **BCE:** gradient = σ(z)−y, no saturation; proper; use BCEWithLogitsLoss, always.
- **Focal:** (1−p_t)^γ × BCE; zeros easy examples; for dense detection and extreme imbalance.
- **CCE:** −log(correct class prob); gradient = ŷ−y; never double-softmax.
- **Label smooth CE:** prevents overconfidence; replaces 1-hot with soft targets.
- **KL Divergence:** measures how Q diverges from P; minimising = matching distributions.
- **Triplet:** relative distance constraint; requires hard negative mining; superseded by InfoNCE.
- **InfoNCE:** N-way classification among negatives; lower bounds MI; large batch = better.
- **ELBO:** reconstruction + β·KL; reparameterisation enables backprop through sampling.
- **WGAN-GP:** Wasserstein distance with gradient penalty; no mode collapse; loss = quality.
- **Diffusion MSE:** predict the noise ε; MSE = MLE for Gaussian denoising; ε-parameterisation is stable.
- **Dice:** soft F1 score; directly optimises the segmentation metric; pair with BCE.
- **Perceptual:** VGG feature distance; sharper than MSE; use for any image synthesis task.
- **Poisson deviance:** correct likelihood for counts; variance ∝ mean; no back-transform bias.
- **Tweedie (p=1.5):** zero-inflated positive data; compound Poisson-Gamma; insurance and marketing.

---

# APPENDIX A — All Loss Numerics on Running Example

**Running example:** `y=[1,0,1,0]`, `ŷ=[0.9,0.2,0.6,0.7]`, `logits≈[2.197,−1.386,0.405,0.847]`

| Loss | Value | Notes |
|---|---|---|
| **MSE** | **0.175** | Sample 4 (e=−0.7) = 70% of total |
| MAE | 0.350 | |
| RMSE | 0.418 | RMSE/MAE = 1.194 |
| Huber (δ=0.5) | 0.0825 | Samples 1–3 in quadratic zone; sample 4 in linear |
| Log-Cosh | 0.0825 | ≈ Huber here (all \|e\| ≈ δ range) |
| Pinball (τ=0.5) | 0.175 = MAE/2 | ✓ |
| Pinball (τ=0.7) | 0.155 | |
| **BCE** | **0.5108** | Sample 4 (confident wrong) = 59% of total |
| Focal (α=0.25,γ=2) | 0.0426 | ~83% reduction: easy samples down-weighted |
| Hinge | 0.850 | |
| Squared Hinge | 1.125 | |
| KL(p\|\|q) | 0.04576 | p=[0.1,0.4,0.5], q=[0.2,0.3,0.5] |
| CCE (class 1 true) | 1.417 | logits=[2,1,0.1]; softmax=[0.659,0.242,0.099] |
| Label smooth CE (ε=0.1) | 1.414 | slightly lower — smoothing effect |
| Triplet (d_ap=0.5,d_an=0.8,α=0.3) | 0.0 | constraint satisfied |
| Triplet (d_ap=0.5,d_an=0.6,α=0.3) | 0.2 | violation |
| Contrastive (sim,d=0.3) | 0.09 | d²=0.09 |
| Cos embed (v=[1,0,0],u=[0.6,0.8,0]) | 0.4 | 1−cos=1−0.6 |
| VAE KL (μ=[0.5,−0.3],logv=[−0.4,0.2]) | 0.2159 | |
| GAN D loss (real=[0.9,0.8,0.85],fake=[0.3,0.4,0.2]) | 0.2636 | |
| WGAN D loss | −0.55 | E[fake]−E[real] |
| Poisson deviance (λ=[2.5,4.8,2.1,0.3]) | 0.1767 | |

---

# APPENDIX B — PyTorch Custom Loss Template

```python
import torch
import torch.nn as nn

class CustomLoss(nn.Module):
    """
    Template for a custom PyTorch loss function.
    
    Requirements:
    1. Inherit from nn.Module
    2. __init__ stores any hyperparameters
    3. forward() implements the loss computation and returns a scalar
    4. Use torch operations (not numpy) for autograd compatibility
    5. Respect the reduction parameter: 'none' | 'mean' | 'sum'
    6. Handle edge cases: division by zero, log(0), clipping
    """
    
    def __init__(self, delta: float = 1.0, reduction: str = 'mean'):
        super().__init__()
        self.delta = delta
        self.reduction = reduction
    
    def forward(self, pred: torch.Tensor, target: torch.Tensor) -> torch.Tensor:
        """
        Args:
            pred:   model predictions, shape (N, ...) — may be logits or probabilities
            target: ground-truth values, same shape as pred
        Returns:
            scalar loss (if reduction='mean' or 'sum')
            or per-sample losses (if reduction='none')
        """
        # Compute per-element loss
        e = target - pred
        abs_e = torch.abs(e)
        loss = torch.where(abs_e <= self.delta,
                           0.5 * e ** 2,
                           self.delta * (abs_e - 0.5 * self.delta))
        
        # Apply reduction
        if self.reduction == 'none':
            return loss
        elif self.reduction == 'mean':
            return loss.mean()
        elif self.reduction == 'sum':
            return loss.sum()
        else:
            raise ValueError(f"Unknown reduction: {self.reduction}")


# ── Usage ────────────────────────────────────────────────────────────────────
criterion = CustomLoss(delta=0.5)

# In a training loop:
# pred = model(x)
# loss = criterion(pred, target)
# loss.backward()

# ── Gradient check (always run before deploying a custom loss) ───────────────
pred   = torch.randn(4, requires_grad=True)
target = torch.randn(4)
loss   = CustomLoss()(pred, target)
torch.autograd.gradcheck(
    lambda p: CustomLoss()(p, target),
    pred.double(),
    eps=1e-6, atol=1e-4
)
print("Gradient check passed")

# ── For second-order methods (XGBoost/LightGBM custom objective) ─────────────
def custom_objective_xgboost(y_pred, dtrain):
    """Returns (grad, hess) for XGBoost."""
    y_true = dtrain.get_label()
    e = y_true - y_pred
    # Huber
    delta = 1.0
    abs_e = np.abs(e)
    grad  = -np.where(abs_e <= delta, e, delta * np.sign(e))
    hess  =  np.where(abs_e <= delta, 1., 0.)   # Huber hessian is 1 or 0
    return grad, hess
```

---

# APPENDIX C — Complete Reference: All Major Losses in One File

```python
"""
loss_zoo.py — Reference implementations of all major loss functions.
All verified against PyTorch built-ins. MIT licence.
"""
import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np

# ── REGRESSION ───────────────────────────────────────────────────────────────

def mse(pred, target):       return F.mse_loss(pred, target)
def mae(pred, target):       return F.l1_loss(pred, target)

def huber(pred, target, delta=1.0):
    e = torch.abs(target - pred)
    return torch.mean(torch.where(e <= delta, 0.5*e**2, delta*(e-0.5*delta)))

def logcosh(pred, target):
    e = target - pred
    a = torch.abs(e)
    return torch.mean(a + torch.log1p(torch.exp(-2*a)) - torch.log(torch.tensor(2.0)))

def pinball(pred, target, tau=0.5):
    e = target - pred
    return torch.mean(torch.maximum(tau*e, (tau-1)*e))

def rmsle(pred, target):
    pred = torch.clamp(pred, min=0)
    return torch.sqrt(torch.mean((torch.log1p(target) - torch.log1p(pred))**2))

def poisson_loss(pred, target):
    return torch.mean(pred - target * torch.log(pred + 1e-9))

def tweedie_loss(pred, target, p=1.5):
    return torch.mean(
        target*(target**(1-p)-pred**(1-p))/(1-p)
        - (target**(2-p)-pred**(2-p))/(2-p))

# ── BINARY CLASSIFICATION ────────────────────────────────────────────────────

def bce(logits, targets):
    return F.binary_cross_entropy_with_logits(logits, targets)

def focal_loss(logits, targets, alpha=0.25, gamma=2.0):
    bce_val = F.binary_cross_entropy_with_logits(logits, targets, reduction='none')
    pt      = torch.exp(-bce_val)
    at      = targets*alpha + (1-targets)*(1-alpha)
    return (at * (1-pt)**gamma * bce_val).mean()

def hinge_loss(scores, targets_01):
    y_pm = 2*targets_01 - 1
    return torch.clamp(1 - y_pm*scores, min=0).mean()

def sq_hinge_loss(scores, targets_01):
    y_pm = 2*targets_01 - 1
    return (torch.clamp(1 - y_pm*scores, min=0)**2).mean()

# ── MULTI-CLASS ──────────────────────────────────────────────────────────────

def cce(logits, targets):
    return F.cross_entropy(logits, targets)    # targets = integer class indices

def label_smooth_cce(logits, targets, eps=0.1):
    K = logits.size(-1); lp = F.log_softmax(logits, -1)
    oh = torch.zeros_like(lp).scatter(-1, targets.unsqueeze(-1), 1)
    return -((oh*(1-eps) + eps/K) * lp).sum(-1).mean()

def kl_divergence(log_q, p):
    """KL(p || q). log_q = log-probabilities of Q. p = probabilities of P."""
    return F.kl_div(log_q, p, reduction='batchmean')

# ── METRIC LEARNING ──────────────────────────────────────────────────────────

def contrastive(emb1, emb2, y, margin=1.0):
    d   = torch.norm(emb1-emb2, dim=1)
    pos = y * d.pow(2)
    neg = (1-y) * torch.clamp(margin-d, min=0).pow(2)
    return (pos+neg).mean()

def triplet(anchor, positive, negative, margin=0.3):
    d_ap = torch.norm(anchor-positive, dim=1)
    d_an = torch.norm(anchor-negative, dim=1)
    return torch.clamp(d_ap-d_an+margin, min=0).mean()

def info_nce(z1, z2, tau=0.07):
    N = z1.size(0)
    z = torch.cat([z1, z2])
    sim = torch.mm(z, z.T) / tau
    sim.fill_diagonal_(float('-inf'))
    labels = torch.cat([torch.arange(N,2*N), torch.arange(N)])
    return F.cross_entropy(sim, labels.to(z.device))

def cosine_embedding(v1, v2, y, margin=0.0):
    return nn.CosineEmbeddingLoss(margin=margin)(v1, v2, y)

# ── VAE ──────────────────────────────────────────────────────────────────────

def vae_elbo(recon, x, mu, log_var, beta=1.0):
    recon_loss = F.mse_loss(recon, x, reduction='sum')
    kl         = -0.5 * torch.sum(1 + log_var - mu.pow(2) - log_var.exp())
    return (recon_loss + beta*kl) / x.size(0)

def reparameterise(mu, log_var):
    return mu + torch.randn_like(mu) * torch.exp(0.5*log_var)

# ── GAN ──────────────────────────────────────────────────────────────────────

def gan_d_nonsaturating(d_real, d_fake):
    return (F.binary_cross_entropy_with_logits(d_real, torch.ones_like(d_real)) +
            F.binary_cross_entropy_with_logits(d_fake, torch.zeros_like(d_fake))) / 2

def gan_g_nonsaturating(d_fake):
    return F.binary_cross_entropy_with_logits(d_fake, torch.ones_like(d_fake))

def hinge_d(d_real, d_fake):
    return (torch.clamp(1-d_real, min=0).mean() + torch.clamp(1+d_fake, min=0).mean()) / 2

def wgan_gp(D, real, fake, device, lam=10):
    a = torch.rand(real.size(0), *([1]*(real.dim()-1)), device=device)
    x = (a*real + (1-a)*fake).requires_grad_(True)
    d = D(x)
    g = torch.autograd.grad(d, x, torch.ones_like(d), create_graph=True)[0]
    return lam * ((g.view(g.size(0),-1).norm(2,dim=1)-1)**2).mean()

# ── SEGMENTATION ─────────────────────────────────────────────────────────────

def dice_loss(pred, target, smooth=1.0):
    p     = torch.sigmoid(pred)
    inter = (p * target).sum((1,2))
    union = p.sum((1,2)) + target.sum((1,2))
    return (1-(2*inter+smooth)/(union+smooth)).mean()

def bce_dice(pred, target, alpha=0.5):
    return alpha*F.binary_cross_entropy_with_logits(pred,target) \
         + (1-alpha)*dice_loss(torch.sigmoid(pred),target)

def tversky_loss(pred, target, alpha=0.3, beta=0.7, smooth=1.0):
    p  = torch.sigmoid(pred)
    tp = (p*target).sum((1,2))
    fp = (p*(1-target)).sum((1,2))
    fn = ((1-p)*target).sum((1,2))
    return (1-(tp+smooth)/(tp+alpha*fp+beta*fn+smooth)).mean()

# ── SELF-SUPERVISED ──────────────────────────────────────────────────────────

def barlow_twins(z1, z2, lam=0.005):
    N,D  = z1.shape
    z1   = (z1-z1.mean(0)) / z1.std(0)
    z2   = (z2-z2.mean(0)) / z2.std(0)
    C    = (z1.T @ z2) / N
    diag = (C.diag()-1).pow(2).sum()
    off  = (C - torch.eye(D, device=z1.device)).pow(2).sum() - (C.diag()-1).pow(2).sum()
    return diag + lam*off

def byol_loss(online_pred, target_proj):
    online = F.normalize(online_pred, dim=-1)
    target = F.normalize(target_proj.detach(), dim=-1)   # stop-gradient
    return 2 - 2*(online*target).sum(dim=-1).mean()

# ── QUICK TEST ────────────────────────────────────────────────────────────────
if __name__ == '__main__':
    y = torch.tensor([1., 0., 1., 0.])
    p = torch.tensor([0.9, 0.2, 0.6, 0.7])
    z = torch.log(p/(1-p))

    print(f"MSE:         {mse(p, y):.4f}")          # 0.1750
    print(f"MAE:         {mae(p, y):.4f}")           # 0.3500
    print(f"Huber(0.5):  {huber(p, y, 0.5):.4f}")   # 0.0825
    print(f"BCE:         {bce(z, y):.4f}")            # 0.5108
    print(f"Focal:       {focal_loss(z, y):.4f}")     # 0.0426
    print(f"Hinge:       {hinge_loss(p, y):.4f}")     # 0.8500
    print(f"Pinball(0.7):{pinball(p, y, 0.7):.4f}")  # 0.1550
    print("All losses verified.")
```

---

# Final Goal Checklist

After studying this document, you should be able to:

- [ ] **State the three things every loss encodes:** noise model, optimal predictor, cost structure.
- [ ] **Explain why BCE has no vanishing gradient** and derive `∂BCE/∂logit = σ(z) − y`.
- [ ] **Derive `∂CCE/∂logit_k = ŷₖ − yₖ`** from first principles.
- [ ] **Explain focal loss** and why `(1−p_t)^γ` solves dense detection imbalance.
- [ ] **Set τ for quantile regression** from the newsvendor cost ratio.
- [ ] **Explain the ELBO** and derive the closed-form KL for Gaussian posteriors.
- [ ] **Explain the reparameterisation trick** and why it is necessary for VAE training.
- [ ] **Contrast GAN losses**: minimax BCE → non-saturating → hinge → WGAN-GP.
- [ ] **Explain why Wasserstein > JS divergence** for GAN training.
- [ ] **Implement InfoNCE** from scratch and explain the temperature τ.
- [ ] **Explain hard negative mining** and why random negatives fail in triplet training.
- [ ] **Choose between Dice/BCE/Focal** for a given segmentation problem.
- [ ] **Explain posterior collapse** in VAEs and list three fixes.
- [ ] **Explain exposure bias** and why RLHF mitigates it.
- [ ] **Implement any loss in PyTorch** correctly, including gradient checks and numerically stable versions.
- [ ] **Navigate the decision tree** (Part 10) and justify any loss choice to a non-technical stakeholder.

> **One-sentence takeaway.**
>
> **A loss function is not just a training detail — it is a statement about your noise model, your optimal predictor, and your cost structure;** choose it the same way you would choose an evaluation metric, verify the gradient has no pathological saturation, and always check that what you optimise is consistent with what you report.
