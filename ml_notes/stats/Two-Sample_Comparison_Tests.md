

# 📘 **NOTES: Two-Sample Comparison Tests (Mean Difference Analysis)**

These notes cover:
✔ Why the test is used
✔ What each statistic means
✔ Simple real-life example
✔ Python code to compute everything
✔ Interpretation

---

# ---------------------------------------

# **1. Two-Sample t-Test**

# ---------------------------------------

## ✅ **What is it?**

A test that checks whether **two group means are significantly different**.

Used when:

* You have **Group A vs Group B**
* You suspect their averages differ
* Data is numeric and roughly normal

---

## ✅ **Why do we perform it?**

To answer this question:

> “Is the difference between two group averages real or just due to random chance?”

This avoids making wrong decisions based only on observed numbers.

---

## 🧠 **Example Scenario**

You compare **CGPA of Batch A vs Batch B**:

| Group | Mean  | Std   |
| ----- | ----- | ----- |
| A     | 2.977 | 0.37  |
| B     | 2.853 | 0.304 |

Observed difference = **0.124**

We use a t-test to check:

* Is 0.124 a real difference?
* Or could it be random variation?

---

## 🧮 **How to calculate (Python code)**

```python
import numpy as np
from scipy import stats

# Example data (simulated with same mean & std)
mean1, std1, n1 = 2.977, 0.37, 800
mean2, std2, n2 = 2.853, 0.304, 800

# Generate random groups for demonstration
group1 = np.random.normal(mean1, std1, n1)
group2 = np.random.normal(mean2, std2, n2)

# Two-sample t-test (Welch)
t_stat, p_value = stats.ttest_ind(group1, group2, equal_var=False)

print("T-stat:", t_stat)
print("p-value:", p_value)
```

---

# ---------------------------------------

# **2. Confidence Interval (CI 95%)**

# ---------------------------------------

## ✅ **What is it?**

A range of values where the **true mean difference** is expected to lie with **95% confidence**.

## ✅ **Why?**

It gives more information than just p-value:

* p-value = is difference significant?
* CI = how big is the difference?

Your CI: **[0.09, 0.16]**
Meaning:

> The true difference likely lies between 0.09 and 0.16.

---

## 📘 **How to calculate CI (Python)**

```python
import numpy as np
from scipy import stats

# Mean diff
diff = group1.mean() - group2.mean()

# Standard error
se = np.sqrt(group1.var(ddof=1)/n1 + group2.var(ddof=1)/n2)

# CI bounds
ci_low = diff - 1.96 * se
ci_high = diff + 1.96 * se

print("95% CI:", (ci_low, ci_high))
```

---

# ---------------------------------------

# **3. Cohen’s d (Effect Size)**

# ---------------------------------------

## ✅ **What is it?**

A scale-free measure of **how big** the difference is.

| d-value | Interpretation |
| ------- | -------------- |
| 0.2     | Small          |
| 0.5     | Medium         |
| 0.8     | Large          |

Your value: **0.36 → small-to-medium effect**

---

## 🧠 **Why?**

Because “difference is significant” does **not** tell if it is **important**.

---

## 📘 **How to compute Cohen’s d**

```python
# Pooled standard deviation
s1, s2 = group1.std(ddof=1), group2.std(ddof=1)
spooled = np.sqrt(((n1-1)*s1**2 + (n2-1)*s2**2) / (n1+n2-2))

cohen_d = (group1.mean() - group2.mean()) / spooled

print("Cohen's d:", cohen_d)
```

---

# ---------------------------------------

# **4. Bayes Factor BF10**

# ---------------------------------------

## ✅ **What is it?**

A Bayesian metric that measures **strength of evidence**:

> BF10 = How much more likely the alternative hypothesis is compared to the null?

Your BF10 = **1.24e+09**
→ Decisive evidence for difference.

---

## 🧠 **Why use it?**

Because p-values only say “significant or not”.

Bayes Factor measures:

* practical evidence
* model likelihood
* reliability

---

## 📘 **How to compute (Python using BayesFactor library)**

(Example using pingouin)

```python
import pingouin as pg

bf = pg.bayesfactor_ttest(t=t_stat, nx=n1, ny=n2)
print("Bayes Factor BF10:", bf)
```

---

# ---------------------------------------

# **5. Statistical Power**

# ---------------------------------------

## ✅ **What is it?**

Power = probability of detecting a real difference.

Your power = **0.99999**
→ Test is extremely reliable.

---

## 🧠 **Why?**

* Ensures the study is not underpowered
* Large sample sizes → more trustworthy results

---

## 📘 **How to compute power (Python)**

```python
from statsmodels.stats.power import TTestIndPower

effect = cohen_d
analysis = TTestIndPower()

power = analysis.power(effect_size=effect, nobs1=n1, ratio=n2/n1, alpha=0.05)
print("Power:", power)
```

---

# ---------------------------------------

# **6. Putting All Results Together**

# ---------------------------------------

### ✔ The difference in means is **statistically significant**

### ✔ The true difference is likely between **0.09–0.16**

### ✔ Effect size is **small-to-medium**

### ✔ Bayesian evidence is **extremely strong**

### ✔ Power is **almost 1**, confirming reliability

---

# ---------------------------------------

# ⭐ Final Summary (Short Notes)

# ---------------------------------------

### **Two-sample t-test**

* Compares two group means.
* Used when checking if groups differ significantly.

### **Confidence Interval**

* Shows range of true difference.
* CI that excludes 0 → significant difference.

### **Cohen’s d**

* Measures practical importance.
* d = 0.36 = small-medium effect.

### **Bayes Factor**

* Measures strength of evidence.
* > 100 = decisive evidence.

### **Power**

* Checks reliability of test.
* Power near 1 = excellent sample size.

---

