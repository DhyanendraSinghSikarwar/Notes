

## 📘 **Pandas `.transform()` Function — Complete Notes**

### **Purpose**

`transform()` is used to **apply a function to a Series or group of Series** while **returning a result with the same shape as the original**.
It’s commonly used for **group-wise operations** or **row-wise transformations** where you want to preserve the DataFrame alignment.

---

### **Syntax**

```python
Series.transform(func, *args, **kwargs)
DataFrame.transform(func, *args, **kwargs)
```

---

### **Parameters**

| Parameter      | Description                                                               | Example                              |
| -------------- | ------------------------------------------------------------------------- | ------------------------------------ |
| **`func`**     | Function, string (like 'mean', 'sum'), or list/dict of functions to apply | `'mean'` or `lambda x: x - x.mean()` |
| **`*args`**    | Positional arguments passed to `func`                                     | `transform(my_func, 10)`             |
| **`**kwargs`** | Keyword arguments passed to `func`                                        | `transform(my_func, scale=2)`        |

---

### **Returns**

* For **Series** → Series of **same length**
* For **DataFrame** → DataFrame with **same shape**
* Index/order remains aligned with original data.

---

### **1. Basic Example — Series**

```python
import pandas as pd

s = pd.Series([10, 15, 20])
s.transform(lambda x: x * 2)
```

Output:

```
0    20
1    30
2    40
dtype: int64
```

---

### **2. Using `transform()` on a DataFrame**

```python
df = pd.DataFrame({'a':[1,2,3], 'b':[4,5,6]})
df.transform(lambda x: x + 10)
```

Output:

```
    a   b
0  11  14
1  12  15
2  13  16
```

---

### **3. Group-wise Transform**

One of the most powerful uses: **groupby + transform**.

```python
df = pd.DataFrame({
    'team': ['A', 'A', 'B', 'B'],
    'points': [10, 15, 8, 12]
})

df['team_mean'] = df.groupby('team')['points'].transform('mean')
```

Output:

```
  team  points  team_mean
0    A      10       12.5
1    A      15       12.5
2    B       8       10.0
3    B      12       10.0
```

* Each row receives the **group-level mean**, while keeping original DataFrame length.

---

### **4. Using Custom Functions**

```python
df['zscore'] = df.groupby('team')['points'].transform(lambda x: (x - x.mean()) / x.std())
```

* Computes **z-score per group**
* Returns a Series aligned with original DataFrame

---

### **5. Multiple Functions at Once**

```python
df.groupby('team')['points'].transform(['mean', 'max'])
```

* Returns a DataFrame with the same index, containing results for each function.

---

### **6. Common Use Cases**

| Goal              | Example                                                                               |
| ----------------- | ------------------------------------------------------------------------------------- |
| Group mean        | `df.groupby('team')['points'].transform('mean')`                                      |
| Group sum         | `df.groupby('team')['points'].transform('sum')`                                       |
| Z-score           | `df.groupby('team')['points'].transform(lambda x: (x-x.mean())/x.std())`              |
| Normalization     | `df.groupby('team')['points'].transform(lambda x: (x - x.min()) / (x.max()-x.min()))` |
| Rank / Percentile | `df.groupby('team')['points'].transform(lambda x: x.rank(pct=True))`                  |

---

### **7. Difference Between `apply()` and `transform()`**

| Feature      | `apply()`                                 | `transform()`                                    |
| ------------ | ----------------------------------------- | ------------------------------------------------ |
| Return shape | Can be any shape                          | Must return same shape as original               |
| Use case     | Aggregations, flexible                    | Row-aligned transformations, group-wise features |
| Example      | `df.groupby('team')['points'].apply(sum)` | `df.groupby('team')['points'].transform('sum')`  |

---

### **8. Combining with `assign()`**

`transform()` is often used in **feature engineering pipelines**:

```python
df = df.assign(
    team_mean = lambda d: d.groupby('team')['points'].transform('mean'),
    zscore = lambda d: d.groupby('team')['points'].transform(lambda x: (x-x.mean())/x.std())
)
```

* Keeps the DataFrame chainable
* Works well with `.pipe()` and `.query()` in functional pipelines

---

### **9. Notes / Best Practices**

* Always preserves **row alignment**
* Ideal for **group-level features in ML pipelines**
* Can be used with **built-in aggregation strings** (`'mean'`, `'sum'`) or **custom functions**
* Works seamlessly in **method chains with assign() or pipe()**

---

### **10. Quick Recap**

| Feature                | Description                       | Example                                                                                               |
| ---------------------- | --------------------------------- | ----------------------------------------------------------------------------------------------------- |
| **Shape**              | Output same length/index as input | `df['col'].transform('mean')`                                                                         |
| **Group-wise**         | Works with `groupby()`            | `df.groupby('team')['points'].transform('mean')`                                                      |
| **Custom functions**   | Lambda, callable                  | `lambda x: (x-x.mean())/x.std()`                                                                      |
| **Multiple functions** | `['mean','max']`                  | Returns DataFrame aligned with index                                                                  |
| **Use in pipeline**    | Works with `assign()` / `pipe()`  | `df.assign(zscore = lambda d: d.groupby('team')['points'].transform(lambda x: (x-x.mean())/x.std()))` |

---

