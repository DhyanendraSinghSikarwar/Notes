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
**# Target encoding**
Here, we split the data into train and test then apply target encoding with KFold CV
```
from sklearn.model_selection import KFold

X_train_correct = X_train.copy()
X_train_correct['model_te'] = np.nan

kf = KFold(n_splits=5, shuffle=True, random_state=42)

for train_idx, val_idx in kf.split(X_train):
    X_tr = X_train.iloc[train_idx]
    y_tr = y_train.iloc[train_idx]

    X_val = X_train.iloc[val_idx]

    fold_map = X_tr.join(y_tr).groupby('model')['price'].mean()

    X_train_correct.iloc[val_idx, 
        X_train_correct.columns.get_loc('model_te')
    ] = X_val['model'].map(fold_map)

X_train_correct['model_te'].fillna(y_train.mean(), inplace=True)
X_train_correct.drop(columns='model', inplace=True)

final_map = X_train.join(y_train).groupby('model')['price'].mean()

X_test_correct = X_test.copy()
X_test_correct['model_te'] = X_test_correct['model'].map(final_map)
X_test_correct['model_te'].fillna(y_train.mean(), inplace=True)
X_test_correct.drop(columns='model', inplace=True)

lr_correct = LinearRegression()
lr_correct.fit(X_train_correct, y_train)

y_pred_correct = lr_correct.predict(X_test_correct)
print("Test R2 (CORRECT):", r2_score(y_test, y_pred_correct))

```

**# frequency encoded**
```
freq_map = X_train['fuel'].value_counts(normalize=True)
X_train['fuel_freq'] = X_train['fuel'].map(freq_map)
X_test['fuel_freq']  = X_test['fuel'].map(freq_map)
```
