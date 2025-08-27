Choosing the **right statistical test depending on**:

* What type of variable(s) you’re analyzing (**numerical vs categorical**)
* How many groups / comparisons you’re making (univariate, bivariate, multivariate)
* The assumptions you can (or cannot) make (normality, equal variance, sample size, etc.)

Let’s break it down cleanly.

---

## 1. **Univariate analysis**

(Analyzing one variable at a time)

* **Numerical variable**:

  * Check distribution with **Shapiro–Wilk test** (normality)
  * Check variance with **Levene’s test** or **Bartlett’s test**
  * For outlier detection: **Z-score test**, **Grubbs’ test**

* **Categorical variable**:

  * **Chi-square goodness-of-fit test** → does the observed distribution match expected proportions?
  * **Binomial test** (if 2 categories)
  * **One-sample proportion z-test**

---

## 2. **Bivariate analysis**

(Relationship between two variables)

* **Numerical ↔ Numerical**

  * **Pearson correlation** (linear, normal assumption)
  * **Spearman rank correlation** (non-parametric, monotonic)
  * **Regression (simple linear regression)**

* **Categorical ↔ Categorical**

  * **Chi-square test of independence** (expected freq ≥ 5)
  * **Fisher’s exact test** (if small samples)
  * **McNemar’s test** (paired categorical, e.g., before/after)

* **Numerical ↔ Categorical**

  * **t-test (independent samples)** → compare means of two groups
  * **Mann–Whitney U test** (non-parametric alternative to t-test)
  * **One-way ANOVA** → compare means across >2 groups
  * **Kruskal–Wallis test** (non-parametric alternative to ANOVA)

---

## 3. **Multivariate analysis**

(Relationship among 3+ variables)

* **Numerical target with multiple categorical predictors**

  * **Two-way / n-way ANOVA** → main + interaction effects
  * **MANOVA** (Multivariate ANOVA) → multiple dependent variables

* **Categorical target with multiple predictors**

  * **Logistic regression**
  * **Chi-square tests** for pairwise categorical relations

* **Mix of numerical + categorical predictors**

  * **Multiple regression** (if target is numeric)
  * **Logistic regression** (if target is binary categorical)
  * **Cox regression** (if target is survival time/event)

* **Numerical + numerical + categorical interaction**

  * **ANCOVA** (Analysis of Covariance) → combines regression + ANOVA (numerical + categorical predictors together).

---

## Cheat-sheet style summary

| Situation         | Parametric Test            | Non-parametric Test  |
| ----------------- | -------------------------- | -------------------- |
| 1 num var         | Shapiro-Wilk (normality)   | —                    |
| 1 cat var         | Chi-square goodness-of-fit | Binomial test        |
| num ↔ num         | Pearson correlation        | Spearman correlation |
| cat ↔ cat         | Chi-square independence    | Fisher’s exact       |
| num ↔ 2-cat       | Independent t-test         | Mann–Whitney U       |
| num ↔ >2-cat      | One-way ANOVA              | Kruskal–Wallis       |
| num ↔ cat (multi) | Two-way ANOVA / ANCOVA     | Friedman test        |
| multi num targets | MANOVA                     | —                    |

---

👉 Rule of thumb:

* **Parametric tests** (t-test, ANOVA, Pearson) assume normality & equal variances.
* **Non-parametric tests** (Mann-Whitney, Kruskal-Wallis, Spearman) are safer if data is skewed or sample size is small.

