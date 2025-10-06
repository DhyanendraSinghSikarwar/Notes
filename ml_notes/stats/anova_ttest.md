# Assumptions and Alternatives for t-test and ANOVA

When comparing a **numerical variable** across **categorical groups**, we commonly use **t-tests** (for 2 groups) and **ANOVA** (for 3 or more groups). However, these tests rely on certain assumptions. This note summarizes the key assumptions, how to test them, and what alternative tests to use when assumptions are violated.

---

## Key Assumptions

| Assumption                 | Description                                             | Importance                                    |
|----------------------------|---------------------------------------------------------|-----------------------------------------------|
| **Normality**              | Dependent variable is normally distributed within each group | Ensures validity of p-values and test statistics |
| **Homogeneity of Variance** | Variances are equal across groups                       | Prevents inflated Type I error rates           |
| **Independence**            | Observations are independent                            | Avoids bias and invalid inferences              |

---

## Testing Assumptions

| Assumption                | Common Test(s)           | Interpretation                                |
|---------------------------|--------------------------|-----------------------------------------------|
| **Normality**             | Shapiro-Wilk Test        | p > 0.05 → Normality assumed; p < 0.05 → Not normal |
| **Homogeneity of Variance** | Levene’s Test           | p > 0.05 → Equal variances; p < 0.05 → Unequal variances |

---

## Which Test to Use Based on Assumptions

| Scenario                             | 2 Groups                      | 3 or More Groups             |
|-------------------------------------|------------------------------|-----------------------------|
| Normality ✔, Equal Variance ✔       | Standard **t-test**           | Standard **ANOVA**           |
| Normality ✔, Variance ❌             | **Welch’s t-test**            | **Welch’s ANOVA**            |
| Normality ❌ (Any variance)           | **Mann-Whitney U test**       | **Kruskal-Wallis test**      |

---

### Notes:

- **Welch’s tests** are parametric tests that do **not assume equal variances** but still assume normality.
- **Mann-Whitney U** and **Kruskal-Wallis** tests are **non-parametric** and do **not assume normality** or equal variances.
- With large sample sizes, mild deviations from normality may still allow using parametric tests (due to the Central Limit Theorem).
- If homogeneity of variance is violated but normality holds, prefer Welch’s tests over standard t-test/ANOVA.
- If normality is violated, use non-parametric alternatives regardless of variance equality.

---

## Example Python Imports for Assumption Testing and Alternatives

```python
from scipy.stats import shapiro, levene, ttest_ind, mannwhitneyu, f_oneway, kruskal

# Normality test example
stat, p = shapiro(group_data)

# Homogeneity of variance test example
stat, p = levene(group1, group2, group3)

# Standard t-test (equal variances)
t_stat, p = ttest_ind(group1, group2, equal_var=True)

# Welch's t-test (unequal variances)
t_stat, p = ttest_ind(group1, group2, equal_var=False)

# Mann-Whitney U test (non-parametric alternative for 2 groups)
stat, p = mannwhitneyu(group1, group2)

# Standard ANOVA (equal variances)
f_stat, p = f_oneway(group1, group2, group3)

# Kruskal-Wallis test (non-parametric alternative for 3+ groups)
stat, p = kruskal(group1, group2, group3)
