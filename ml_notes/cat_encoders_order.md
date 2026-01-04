| Encoding type          | Before split? | After split?    | sklearn / library function                 | Notes                                                    |
| ---------------------- | ------------- | --------------- | ------------------------------------------ | -------------------------------------------------------- |
| **Label Encoding**     | ❌ No          | ✅ Yes           | `sklearn.preprocessing.LabelEncoder`       | Use mainly for **target** or **tree models**, not linear |
| **Ordinal Encoding**   | ❌ No          | ✅ Yes           | `sklearn.preprocessing.OrdinalEncoder`     | Explicit order must be provided                          |
| **One-Hot Encoding**   | ❌ No          | ✅ Yes           | `sklearn.preprocessing.OneHotEncoder`      | Preferred over `pd.get_dummies`                          |
| **Frequency Encoding** | ❌ No          | ✅ Yes           | ❌ Not built-in (use pandas)                | Map category → frequency                                 |
| **Target Encoding**    | ❌❌ NEVER      | ✅ Yes (CV only) | `category_encoders.TargetEncoder`          | High leakage risk                                        |
| **Binary Encoding**    | ❌ No          | ✅ Yes           | `category_encoders.BinaryEncoder`          | Compact alternative to OHE                               |
| **Hashing Encoder**    | ✅ Yes         | ✅ Yes           | `sklearn.feature_extraction.FeatureHasher` | Stateless, collision risk                                |



Note:
**# frequency encoded**
```
freq_map = X_train['fuel'].value_counts(normalize=True)
X_train['fuel_freq'] = X_train['fuel'].map(freq_map)
X_test['fuel_freq']  = X_test['fuel'].map(freq_map)
```
