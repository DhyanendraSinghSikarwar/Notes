Linear regression workflow

---

## 📈 Linear Regression Modeling – End-to-End Approach

### 0. Exploratory Data Analysis (EDA)

* Understand data structure and variable types
* Analyze distributions, trends, and relationships
* Identify missing values and potential outliers
* Perform univariate and bivariate analysis

---

### 1. Missing Value Treatment

* Identify null values in numerical and categorical features
* Apply appropriate imputation strategies:

  * Mean / Median (numerical)
  * Mode / “Unknown” category (categorical)
* Ensure imputation is learned **only from training data**

---

### 2. Outlier Detection & Treatment

* Detect outliers using:

  * **Z-score method**
  * **IQR method**:
    [
    \text{Outlier} > Q3 + 1.5 \times IQR \quad \text{or} \quad < Q1 - 1.5 \times IQR
    ]
* Treat outliers using capping, transformation, or removal
* Apply outlier treatment **including the target variable**, if required

---

### 3. Train–Validation–Test Split

* Split the dataset into:

  * Training set
  * Validation set
  * Test set
* Ensure randomness and reproducibility using a fixed `random_state`

---

### 4. Feature Scaling & Preprocessing (Train Data Only)

* Apply scaling **only on numerical features**
* Choose scaler based on data characteristics:

  * **StandardScaler**

    * Used when features are approximately normally distributed
    * Mean = 0, Standard Deviation = 1
    * Commonly used with Linear Regression
  * **MinMaxScaler**

    * Scales features to a fixed range (e.g., 0–1)
    * Preserves data shape
    * Faster convergence for advanced algorithms
    * Avoid when data contains extreme outliers
* Use **Pipeline / ColumnTransformer** to handle:

  * Numerical features
  * Categorical features (separately)

---

### 5. Handling Class Imbalance (If Applicable)

* Apply resampling techniques **only on training data**:

  * SMOTE
  * ADASYN
* Note: Typically relevant for **classification**, not regression

---

### 6. Model Training & Evaluation

* Fit Linear Regression model on training data
* Evaluate model using:

  * R² score
  * Adjusted R²
  * RMSE / MAE (if required)

---

### 7. Model Validation

* Validate model stability using:

  * **K-Fold Cross-Validation**
  * **Stratified K-Fold** (for imbalanced classification targets)
* Used primarily for:

  * Performance validation
  * Hyperparameter tuning

---

### 8. Checking Linear Regression Assumptions

* Verify assumptions:

  * Linearity
  * Independence of errors
  * Homoscedasticity
  * Normality of residuals
  * No multicollinearity
    
* If assumptions are violated:

  * Apply transformations (log, sqrt, Box-Cox)
  * Add polynomial features
  * Remove or engineer features

---

### 9. Regularization (Overfitting Control)

* Apply:

  * **Ridge Regression (L2)**
  * **Lasso Regression (L1)**
* Tune regularization strength (`alpha` / `lambda`) using:

  * GridSearchCV
  * RandomizedSearchCV
* Select model based on validation performance

---

## ✅ Final Notes

* All preprocessing steps must be learned from **training data only**
* Use pipelines to ensure reproducibility and prevent data leakage
* Clearly document assumptions, transformations, and decisions

---

## 🔍 Methods to Check Linear Regression Assumptions

### 1. Linearity

**Assumption:**
The relationship between independent variables and the target variable is linear.

**How to Check:**

* Compute **correlation coefficients** (Pearson correlation) between numerical features and the target.
* Visual inspection using scatter plots between features and target.

**Indicators:**

* High absolute correlation value → stronger linear relationship
* Very low correlation → weak linear relationship (consider transformation or non-linear models)

---

### 2. Independence of Errors & Homoscedasticity

**Assumptions:**

* Errors (residuals) are independent
* Variance of errors is constant across all levels of predictions (homoscedasticity)

**How to Check:**

* Plot **Residuals vs Predicted Values (`y_pred`)**

**Interpretation:**

* Random scatter around zero → assumptions satisfied
* Funnel-shaped or patterned spread → heteroscedasticity
* Clear patterns → model misspecification

---

### 3. Normality of Residuals

**Assumption:**
Residuals are normally distributed (mainly required for hypothesis testing and confidence intervals).

**How to Check:**

* **Histogram of residuals**
* **Q–Q (Quantile–Quantile) plot**
* **Shapiro–Wilk statistical test**

#### Shapiro–Wilk Test

* **Null hypothesis (H₀):** Residuals follow a normal distribution
* **Alternative hypothesis (H₁):** Residuals are not normally distributed

**Decision Rule:**

* `p-value > 0.05` → Fail to reject H₀ → Residuals are approximately normal
* `p-value ≤ 0.05` → Reject H₀ → Residuals are not normal

**Note:**

* Shapiro–Wilk is sensitive to large sample sizes
* For large datasets, visual checks (Q–Q plot) are often more reliable

**Interpretation of Histogram and Q-Q plot:**

* Bell-shaped histogram → normal distribution
* Points lying close to the diagonal line in Q–Q plot → normality satisfied
* Heavy tails or skewness → consider transformation

---

### 4. Multicollinearity (For Explainability)

**Assumption:**
Independent variables should not be highly correlated with each other.

**How to Check:**

* Calculate **Variance Inflation Factor (VIF)** for numerical features

**Interpretation:**

* VIF ≈ 1 → No multicollinearity
* VIF between 1–5 → Moderate multicollinearity
* VIF > 5 (or >10) → High multicollinearity (remove or combine features)

**Usage Note:**

* VIF is primarily used when **model interpretability and coefficient stability** are important.

---

## 📌 Summary Table

| Assumption                      | Method Used                | Tool                |
| ------------------------------- | -------------------------- | ------------------- |
| Linearity                       | Correlation, Scatter plot  | Pearson Correlation |
| Independence & Homoscedasticity | Residual vs Predicted Plot | Scatter Plot        |
| Normality of Residuals          | Histogram, Q–Q Plot        | Visual Inspection   |
| Multicollinearity               | VIF                        | Statsmodels         |

