### Clarification of Your Questions

As an experienced data scientist, I'll address your questions systematically. Let's start with your dataset:
- **5 Numerical features**: Continuous values (e.g., age, income)
- **2 Ordinal Categorical**: Ordered categories (e.g., education level: 1=High School, 2=Bachelor, 3=Master)
- **3 Nominal Categorical**: Unordered categories (e.g., city, product type). One is multiclass (e.g., country with >50 classes).

---

### 1. Non-Parametric Models & Hypothesis Testing
**Short Answer**:  
**You generally don't need rigorous hypothesis testing for distributions with ensemble models, but understanding distributions/relationships still matters.**

**Details**:  
- **Tree-based ensembles (RF, XGBoost, LightGBM)** are robust to:
  - Non-normal feature distributions
  - Non-linear relationships
  - Multicollinearity
- **Why you still care**:  
  - **Feature engineering**: Distribution skewness (e.g., log-transform heavy-tailed numerical features) can improve performance.
  - **Model interpretation**: Hypothesis tests (e.g., permutation importance) help identify significant features.
  - **Data quality**: Detecting outliers (e.g., via IQR) prevents splits on noise.
  - **Example**: A heavily skewed `income` column might benefit from log-transformation even in RF to help splits capture patterns better.

**Conclusion**: Skip formal normality/correlation tests for ensembles, but always perform EDA to understand distributions.

---

### 2. Feature Testing Requirements for Specific Algorithms
#### Naive Bayes
- **Categorical Features**: No tests needed. Use count-based likelihoods.
- **Numerical Features**:  
  - **GaussianNB**: Test normality (Shapiro-Wilk or Q-Q plots).  
    *Example*: If a numerical column fails normality test, apply Box-Cox transform.
- **Action**: Preprocess numerical features to match distributional assumptions.

#### KNN
- **Scale Sensitivity**: Test numerical features for scale differences (e.g., variance comparison).  
  *Example*: If `income` (variance=1e9) vs. `age` (variance=10), standardize all features.
- **Categorical Features**: Use distance-compatible encoding (e.g., one-hot for nominal, ordinal encoding for ordinal).

#### K-Means
- **Numerical Features**: Same scaling needs as KNN.
- **Categorical Features**: K-Means is poorly suited for raw categorical data. Use K-Prototypes or encoded representations.
- **Test**: Silhouette analysis to confirm cluster separation.

#### PCA/t-SNE
- **PCA**:  
  - Numerical Features: Test for linear correlations (Pearson/Spearman).  
    *Example*: High correlation (>0.8) between `feature_a` and `feature_b` suggests redundancy.  
  - Categorical Features: Not directly applicable; encode first (PCA assumes continuous inputs).
- **t-SNE**: No statistical tests required, but scale numerical features.

**Summary**:  
| Algorithm      | Required Tests/Checks       | Critical Preprocessing          |  
|----------------|-----------------------------|---------------------------------|  
| Naive Bayes    | Normality (numerical)       | Scale numerical, encode categorical |  
| KNN/K-Means    | Feature scaling             | Standardize/normalize numerical |  
| PCA            | Correlation, linearity      | Scale numerical, handle categorical |  
| t-SNE          | None                        | Scale numerical                 |  

---

### 3. Hypothesis Testing for Parametric Models (Linear Regression, SVM, etc.)
**Stage**: Performed during **EDA** and **model diagnostics** (post-training).

#### Step-by-Step Workflow
1. **EDA Phase (Pre-Modeling)**:  
   - **Numerical Features**:  
     - Test for normality (Shapiro-Wilk) and homoscedasticity (Levene's test).  
       *Example*: If `age` violates normality (p<0.05), apply Yeo-Johnson transform.  
   - **Ordinal Categorical**:  
     - Test monotonic trend (Spearman rank correlation).  
       *Example*: `education_level` vs. target: ρ=0.62 (p<0.001) → strong ordinal relationship.  
   - **Nominal Categorical**:  
     - Test independence from target (Chi-square/ANOVA).  
       *Example*: `country` (multiclass) vs. target: χ²=120 (p=1e-10) → significant feature.  

2. **Model Building Phase**:  
   - **Linear Models (Logistic Regression)**:  
     - Check **multicollinearity** with VIF:  
       ```python
       from statsmodels.stats.outliers_influence import variance_inflation_factor
       # VIF > 5 indicates problematic collinearity
       ```
     - Post-training: Wald test for feature significance (p-values in `statsmodels` output).  

   - **SVM**:  
     - Less emphasis on parametric tests, but still:  
       - Test **feature-target relationships** (ANOVA F-value).  
       - Scale features (SVM is distance-sensitive).  

3. **Post-Model Diagnostics**:  
   - **Residual Analysis**:  
     - Durbin-Watson test for autocorrelation (time-series data).  
     - QQ plots of residuals to check normality.  

**Golden Rule**: Hypothesis testing informs preprocessing and feature engineering, not model selection alone. Always visualize distributions (e.g., `seaborn.pairplot()`) before deciding.

-----------
-----------
Using CHATGPT
----------
--------

## 1. Ensembles and Feature Distributions / Hypothesis Testing

**Bottom line:** Tree-based ensembles (Random Forest, Gradient Boosting, ExtraTrees, etc.) are non-parametric, so you **don’t need** to test whether each feature is normally distributed or meets any other strict statistical assumption.

* **Why not?**

  * They split on value thresholds (e.g. `Age > 42`) rather than fitting coefficients under distributional assumptions.
  * They automatically handle non-linearity, categorical splits (once encoded), outliers, and feature interactions.

* **What you still must do:**

  * **Quality checks** (missingness, extreme outliers, data-leaks)
  * **Encoding** (e.g. ordinal → integer, nominal → one-hot or target encoding)
  * **Class balance** (SMOTE, class weights)
  * **Feature relevance** (drop useless features, check for high cardinality)

---

## 2. “Do I have to test any feature for Naïve Bayes, k-NN, k-Means, PCA, t-SNE?”

| Algorithm               | Key Assumptions / Pre-checks                                                                                                                                           |
| ----------------------- | ---------------------------------------------------------------------------------------------------------------------------------------------------------------------- |
| **Naïve Bayes**         | ● Feature independence given class (strong but often relaxed)<br>● Continuous features ≈ Gaussian (for Gaussian NB)                                                    |
| **k-Nearest Neighbors** | ● Feature scale matters — **always** scale (StandardScaler or MinMax)<br>● No other strict distributional check                                                        |
| **k-Means Clustering**  | ● Assumes spherical clusters of similar size — scaling crucial<br>● Sensitive to outliers<br>● Should check feature variance and scale                                 |
| **PCA**                 | ● Assumes large variance directions contain signal<br>● **Always** center (zero mean) and usually scale (unit variance)<br>● Check covariance structure (correlations) |
| **t-SNE**               | ● No distributional assumptions, but sensitive to scale and perplexity<br>● **Always** scale features<br>● Works best on low-noise, preprocessed data                  |

> **In practice:**
>
> * **Scale first** (except NB where standardization also helps the Gaussian assumption).
> * For NB, you can **visualize** feature-log distributions (histograms / Q–Q plots) to see if Gaussian NB is reasonable—or use a different NB variant (Multinomial, Bernoulli).
> * For k-Means, you might run the **Hopkins statistic** or silhouette analysis to gauge cluster-ability before committing.

---

## 3. Parametric Models (Linear Regression, SVM, Logistic Regression, etc.) and Hypothesis Testing

When you use models that assume a form (linear, kernel-based…), you typically perform hypothesis or assumption checks during **Exploratory Data Analysis (EDA)** and **Model Diagnostics** phases.

### A. **Numerical Features** (5 columns)

1. **Normality**

   * **When:** EDA before modeling.
   * **Test:**

     * Shapiro–Wilk (for small n):

       ```python
       from scipy.stats import shapiro
       stat, p = shapiro(df['num_col'])
       ```
     * Kolmogorov–Smirnov (for larger n).
   * **Why:** Linear models (OLS) assume residuals are normally distributed; NB assumes feature Gaussians.
   * **If violated:**

     * Log / Box–Cox transform
     * Use robust or non-parametric alternatives

2. **Linearity w\.r.t. Target** (for regression or logistic)

   * **When:** EDA & model diagnostics.
   * **Check:**

     * Scatterplot of `num_col` vs. continuous target, or
     * **Lowess** smooth for logistic:

       ```python
       import statsmodels.api as sm
       lowess = sm.nonparametric.lowess
       z = lowess(df['target'], df['num_col'])
       ```
   * **Adjustment:**

     * Polynomial features, splines, or kernel methods (SVM with RBF).

3. **Homoscedasticity** (equal variance of residuals)

   * **When:** Model diagnostics after fitting.
   * **Test:** Breusch–Pagan or White’s test.
   * **Fix:** Weighted least squares or heteroscedasticity-robust standard errors.

### B. **Ordinal Categorical** (2 columns)

* **Approach:** Encode as integers reflecting order (e.g., `Low=1, Med=2, High=3`).
* **Test for Monotonicity:**

  * **When:** EDA
  * **Check:** Boxplots of each ordinal level vs. target mean/proportion.
* **Hypothesis Testing:**

  * **Kruskal–Wallis H-test** (non-parametric ANOVA) to see if median target differs across ordinal groups:

    ```python
    from scipy.stats import kruskal
    groups = [df.loc[df['ord']=='Low','target'], df.loc[df['ord']=='Med','target'], …]
    stat, p = kruskal(*groups)
    ```
* **Use in Model:** If linear trend holds, treat as numeric; else, one-hot encode to let the model learn different effects.

### C. **Nominal Categorical** (3 columns; 1 multiclass)

* **Test of Independence** (χ²):

  * **When:** EDA for feature–target association.
  * ```python
    from scipy.stats import chi2_contingency
    ct = pd.crosstab(df['nom'], df['target'])
    stat, p, dof, ex = chi2_contingency(ct)
    ```
  * **Interpretation:** Low p → strong association → keep feature; high p → weak.
* **Encoding:**

  * One-Hot for small cardinality
  * Target (mean) encoding or embedding for high cardinality

### D. **Multicollinearity** (among any features)

* **When:** Pre-modeling and diagnostics.
* **Check:** Variance Inflation Factor (VIF)

  ```python
  from statsmodels.stats.outliers_influence import variance_inflation_factor
  X = sm.add_constant(encoded_df)
  vif = [variance_inflation_factor(X.values, i) for i in range(X.shape[1])]
  ```
* **Fix:** Drop or combine correlated features.

---

### **Where in the Model Development Pipeline?**

1. **EDA Stage**

   * Distributional tests (normality, independence)
   * Visual checks (histograms, boxplots, scatterplots)
   * Association tests (χ², Kruskal–Wallis)

2. **Pre-processing / Feature Engineering**

   * Transform / encode based on EDA results
   * Scale or normalize where needed

3. **Model Training**

   * Fit your parametric model (e.g. OLS, logistic, SVM)
   * Check assumptions on **residuals** (normality, homoscedasticity)
   * Inspect feature coefficients, p-values (in classical stats frameworks)

4. **Diagnostics & Validation**

   * Re-test residuals and multicollinearity
   * Use cross-validation (CV) to ensure generalization
   * If assumptions fail badly → switch to robust or non-parametric alternative

-------
-------
Example of DeepSeek
-------
------
### Linear Regression Example: Hypothesis Testing Workflow with Code and Interpretation  
**Dataset Example**  
- **Target**: House Price (continuous)  
- **Features**:  
  - `Size_sqft` (numerical)  
  - `Age_years` (numerical)  
  - `Neighborhood` (nominal categorical: A, B, C)  

---

### Step 1: Pre-Modeling Hypothesis Tests (EDA Phase)  
#### (A) **Normality Check (Numerical Features)**  
**Why**: Linear regression assumes features are ~normally distributed for reliable p-values.  
**Test**: Shapiro-Wilk  
```python  
from scipy.stats import shapiro  

# Test for 'Size_sqft'  
stat, p = shapiro(df['Size_sqft'])  
print(f"Size_sqft: p-value = {p:.4f}")  
```  
**Interpretation**:  
- **p > 0.05**: Normally distributed (e.g., p=0.25 → no transformation needed).  
- **p < 0.05 (Fail)**: Non-normal. Try:  
  - Log-transform: `np.log1p(df['Size_sqft'])`  
  - Re-test with transformed data. If still fails, use non-parametric tests (Spearman correlation).  

#### (B) **Homoscedasticity Check**  
**Why**: Ensures residual variance is constant across predictions.  
**Test**: Breusch-Pagan (after model fitting) – *See Step 2*  

#### (C) **Categorical Feature Association**  
**Why**: Verify if categorical feature (`Neighborhood`) relates to target.  
**Test**: One-Way ANOVA  
```python  
from scipy.stats import f_oneway  

group_A = df[df['Neighborhood']=='A']['Price']  
group_B = df[df['Neighborhood']=='B']['Price']  
group_C = df[df['Neighborhood']=='C']['Price']  

f_stat, p = f_oneway(group_A, group_B, group_C)  
print(f"ANOVA for Neighborhood: p-value = {p:.4f}")  
```  
**Interpretation**:  
- **p > 0.05**: No relationship (e.g., p=0.62 → consider dropping feature).  
- **p < 0.05**: Significant relationship (include in model).  

#### (D) **Multicollinearity Check**  
**Why**: High correlation between features skews coefficients.  
**Test**: Pearson/Spearman Correlation  
```python  
corr, p_val = spearmanr(df['Size_sqft'], df['Age_years'])  
print(f"Correlation: r={corr:.2f}, p={p_val:.4f}")  
```  
**Interpretation**:  
- **|r| > 0.7 & p < 0.05**: Severe multicollinearity.  
  - **Solution**: Drop one feature or use PCA.  

---

### Step 2: Post-Modeling Tests (Diagnostics)  
**After fitting model**:  
```python  
import statsmodels.api as sm  

X = df[['Size_sqft', 'Age_years']]  
X = sm.add_constant(X)  # Adds intercept term  
y = df['Price']  

model = sm.OLS(y, X).fit()  
print(model.summary())  # Key output!  
```  

#### (A) **Residual Normality (Q-Q Plot & Shapiro-Wilk)**  
```python  
from statsmodels.graphics.gofplots import qqplot  
import matplotlib.pyplot as plt  

residuals = model.resid  
qqplot(residuals, line='s'); plt.show()  
shapiro(residuals)  
```  
**Interpretation**:  
- **Points align with red line**: Residuals normal.  
- **Deviations at tails**: Non-normality → impacts p-values.  
- **Action if failed**: Use robust regression or transform target.  

#### (B) **Homoscedasticity (Breusch-Pagan)**  
```python  
from statsmodels.stats.diagnostic import het_breuschpagan  

_, p, _, _ = het_breuschpagan(residuals, model.model.exog)  
print(f"Homoscedasticity p-value: {p:.4f}")  
```  
**Interpretation**:  
- **p > 0.05**: Residual variance is constant.  
- **p < 0.05 (Fail)**: Heteroscedasticity → use Huber-White standard errors.  

#### (C) **Feature Significance (Wald Test)**  
**Output in `model.summary()`**:  
```  
                 coef    std err          t      P>|t|  
const        -12.5000      5.000     -2.500      0.013  
Size_sqft      0.7500      0.100      7.500      0.000  
Age_years     -0.5000      0.200     -2.500      0.013  
```  
**Interpretation**:  
- **P>|t| < 0.05**: Feature is significant (e.g., `Size_sqft` and `Age_years`).  
- **P>|t| > 0.05**: Feature not significant (e.g., p=0.24 → remove feature).  

---

### Key Outcomes & Why These Tests Matter  
1. **Pre-Modeling Outcomes**:  
   - Identify redundant features (low ANOVA p-value → drop).  
   - Detect skewed features → apply transformations.  
   - Flag multicollinearity → avoid unstable coefficients.  

2. **Post-Modeling Outcomes**:  
   - Validate model assumptions (normality/homoscedasticity).  
   - Identify unreliable p-values → switch to robust methods.  
   - Confirm feature significance → simplify model.  

**Why Essential for Linear Regression**:  
- ⚠️ **Biased Coefficients**: Non-normal features or multicollinearity distort β-values.  
- 📉 **Invalid p-values**: Heteroscedasticity inflates significance.  
- 🗑️ **Garbage-in-garbage-out**: Including non-significant features overfits model.  

---

### Workflow Summary  
| **Stage**         | **Test**               | **If Failed (p<0.05)**              | **Solution**                     |  
|-------------------|------------------------|-------------------------------------|----------------------------------|  
| Pre-Modeling      | Shapiro-Wilk (feature) | Feature non-normal                  | Log/Box-Cox transform            |  
| Pre-Modeling      | ANOVA (categorical)    | No relationship with target         | Drop feature                     |  
| Pre-Modeling      | Spearman correlation   | Multicollinearity (|r|>0.7)        | Drop feature or use PCA          |  
| Post-Modeling     | Shapiro-Wilk (residual)| Residuals non-normal                | Transform target variable        |  
| Post-Modeling     | Breusch-Pagan          | Heteroscedasticity                  | Robust standard errors           |  
| Post-Modeling     | Wald test (t-stat)     | Feature insignificant (p>0.05)      | Remove feature from model        |  

> **Golden Rule**: Always visualize first (histograms, scatterplots) before hypothesis tests. Tests quantify what plots reveal!

