# 📘 Complete Linear Regression Workflow — Detailed Notes

> **Purpose**: These notes describe the **correct, practical, industry-level flow** for building a **Linear Regression model for prediction**, including **scaling, train–test split, diagnostics, assumptions, VIF, polynomial features, Ridge/Lasso**, and **what to apply where and why**.

These notes combine **theory + practical rules + small examples**, suitable for:
- Interview preparation
- Project documentation
- Personal reference

---

## 🧠 Mental Model (Read This First)

Linear Regression is **not**:
> “Fit model → check assumptions → done” ❌

Linear Regression **is**:
> “Build a reasonable predictive model → diagnose problems → fix them iteratively” ✅

Assumptions are **diagnostic tools**, not prerequisites.

---

# 1️⃣ Data Understanding & Cleaning

### Objective
Ensure data is usable, meaningful, and leakage-free.

### Steps
- Handle missing values (mean/median/mode/domain-based)
- Fix data types
- Remove columns with:
  - All nulls
  - Single unique value
  - Leakage (future information)

### Example
```text
customer_id ❌ (identifier)
price_next_month ❌ (leakage)
engine_size ✅
```

📌 **No modeling decisions here**

---

# 2️⃣ Train–Test Split (DO THIS EARLY)

```python
X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.2, random_state=42
)
```

### Why early?
- Prevents data leakage
- All learning decisions must use **train only**

📌 After this step:
> ❌ Never use test data to decide encoding, scaling, VIF, polynomial, etc.

---

# 3️⃣ Feature Encoding (Train-Based)

### Encoding by feature type

| Feature Type | Encoding Method |
|-------------|----------------|
| Nominal (no order) | One-hot / Target / Frequency |
| Ordinal (ordered) | Manual ordinal mapping |
| Numerical | None |

### Example
```text
fuel_type → One-hot / Target encode
education_level → {school:1, grad:2, postgrad:3}
age → numerical
```

📌 **Fit encoders on X_train, apply to X_test**

---

# 4️⃣ Feature Scaling (CRITICAL STEP)

```python
scaler = StandardScaler()
X_train_scaled = scaler.fit_transform(X_train)
X_test_scaled = scaler.transform(X_test)
```

### Why scaling?
- Linear Regression coefficients depend on scale
- Required for:
  - Ridge
  - Lasso
  - Polynomial features
  - RFE
  - Stable VIF

### Important Rule
> If the model is trained on **scaled data**, diagnostics must also be done on **scaled data**.

---

# 5️⃣ Baseline Linear Regression Model

```python
model = LinearRegression()
model.fit(X_train_scaled, y_train)
```

### Why baseline?
- Acts as reference
- Helps detect under/overfitting
- Needed before diagnostics

---

# 6️⃣ Performance Check — Train vs Test

```python
r2_train = r2_score(y_train, y_pred_train)
r2_test  = r2_score(y_test, y_pred_test)
```

### Interpretation

| Situation | Meaning | Action |
|---------|--------|--------|
| Low train, low test | Underfitting | Add features / polynomial |
| High train, low test | Overfitting | Regularization |
| Train ≈ Test | Good fit | Proceed |

📌 **Do this BEFORE assumption checking**

---

# 7️⃣ Error Analysis

Check:
- MAE
- RMSE
- MAPE (optional)

### Example Insight
```text
RMSE_train = 5
RMSE_test  = 12  → overfitting
```

---

# 🔍 ASSUMPTION CHECKING (Diagnostics)

> Assumptions are checked **after** a reasonable model exists.

---

# 8️⃣ Linearity & Homoscedasticity

### Plot
**Residuals vs Predicted Values**

```python
plt.scatter(y_pred_train, residuals_train)
plt.axhline(0)
```

### What it checks
- Linearity
- Constant variance (homoscedasticity)

### Patterns & Fixes

| Pattern | Meaning | Treatment |
|-------|--------|----------|
| Random | Good | None |
| Curve | Non-linear | Polynomial |
| Funnel | Heteroscedastic | log(y) |
| Clusters | Missing variable | Feature engineering |

📌 **Check on scaled data**

---

# 9️⃣ Normality of Residuals (LAST)

### Methods
- Histogram
- Q-Q plot
- Shapiro–Wilk test

### Important Truth
> Normality is **NOT required for prediction**.

Required only for:
- Confidence intervals
- Hypothesis testing

### Fixes
- log(y)
- Robust regression

---

# 🔁 VIF — Multicollinearity Handling

### When to check VIF?
- After baseline model
- Before polynomial features
- Before Ridge/Lasso

### Apply VIF on:
✅ Scaled **numerical features only**

❌ Exclude:
- One-hot
- Target encoded
- Frequency encoded
- Ordinal

### Thresholds

| VIF | Interpretation |
|----|---------------|
| < 5 | OK |
| 5–10 | Moderate |
| > 10 | Severe |

### Fixes
| Situation | Solution |
|--------|---------|
| Removable feature | Drop |
| Both important | Ridge |
| Sparse model | Lasso |

📌 Scaling does **not** change VIF values, but improves stability

---

# 🔁 Polynomial Features (Non-Linearity)

### When to apply?
- After seeing curvature in residuals
- After reducing collinearity (VIF)

### Correct Order
```
Encode → Scale → Polynomial → Model
```

### Apply polynomial ONLY to:
✅ Continuous numerical features

### ❌ Do NOT apply to:
| Feature Type | Reason |
|-------------|--------|
| One-hot | Meaningless interactions |
| Target encoded | Breaks interpretation |
| Frequency encoded | Artificial numeric |
| Ordinal | Order ≠ spacing |

### Example
```text
age² ✅
engine_size² ✅
fuel_type_te² ❌
education_level² ❌
```

📌 Polynomial features always increase VIF — recheck after

---

# 🔁 Ridge Regression

### When to use?
- Multicollinearity exists
- Polynomial features added
- Overfitting detected

### Properties
- Shrinks coefficients
- Keeps all features
- Improves stability

---

# 🔁 Lasso Regression

### When to use?
- Feature selection needed
- Sparse model required

### Properties
- Sets some coefficients to zero
- Unstable with correlated features

---

# 🔁 ElasticNet

✔ Mix of Ridge + Lasso
✔ Best for correlated + sparse data

---

# 🔁 RFE (Recursive Feature Elimination)

### When to use?
- After scaling
- Feature count is manageable
- Model-specific selection needed

### Notes
- Depends on estimator
- Sensitive to multicollinearity
- Not used for assumption fixing

---

# 🧠 FINAL MASTER FLOW (Cheat Sheet)

```
1. Data cleaning
2. Train–test split
3. Encoding (train-based)
4. Scaling
5. Baseline Linear Regression
6. Train vs Test performance
7. Error analysis
8. Residual diagnostics (scaled data)
9. VIF (numerical only)
10. Polynomial (scaled numerical only)
11. Recheck VIF & performance
12. Ridge / Lasso if needed
13. Final model
```

---

# 🏆 Golden Rules (Memorize These)

✔ Assumptions → diagnostics, not prerequisites  
✔ Assumptions & VIF → on **scaled data**  
✔ Polynomial → scaled numerical only  
✔ Reduce collinearity **before** polynomial  
✔ Never use test data for decisions  

---

**End of Notes**

