

## 📘 **Pandas `assign()` Function — Complete Notes**

### **Purpose**

`assign()` is used to **add or modify columns** in a pandas DataFrame in a clean, *chainable* way.
It’s often combined with methods like `.query()`, `.pipe()`, `.groupby()`, and `.transform()`.

---

### **Syntax**

```python
DataFrame.assign(**kwargs)
```

### **Parameters**

* `**kwargs` → Each key-value pair defines a column:

  * **key:** new or existing column name
  * **value:** expression, Series, scalar, or function returning a Series

If the value is a **function**, it receives the **entire DataFrame** as an argument.

---

### **Returns**

A **new DataFrame** (does not modify the original).

---

### **Common Usage Patterns**

#### 1. **Add a new column**

```python
df.assign(new_col = df['col1'] + df['col2'])
```

#### 2. **Use lambda for cleaner chaining**

```python
df.assign(new_col = lambda x: x['col1'] * 2)
```

#### 3. **Modify existing column**

```python
df.assign(col1 = lambda d: d['col1'] * 1.1)
```

---

### **Conditional Assignment using `np.where`**

You can create new columns based on conditions from other columns.

```python
import numpy as np

df = df.assign(
    discount = lambda d: np.where(d['region'] == 'East', 10, 5)
)
```

Explanation:

* `np.where(condition, value_if_true, value_if_false)`
* Here, rows with `region == 'East'` get discount `10`, otherwise `5`.

---

### **Group-wise Logic using `transform()`**

`transform()` applies a function **per group** and returns a result **aligned with the original DataFrame**, so each row keeps its position.

#### Example 1 — Group Mean

```python
import pandas as pd
import numpy as np

df = pd.DataFrame({
    'team': ['A', 'A', 'B', 'B', 'C', 'C'],
    'points': [10, 15, 8, 12, 20, 18]
})

df = df.assign(
    team_mean = lambda d: d.groupby('team')['points'].transform('mean')
)
```

→ Adds a column showing each team’s mean `points`.

#### Example 2 — Group-wise Z-Score

```python
df = df.assign(
    zscore = lambda d: d.groupby('team')['points']
                       .transform(lambda x: (x - x.mean()) / x.std())
)
```

→ Calculates z-score of each player’s points within their team.

#### Example 3 — Conditional Group Comparison

```python
df = df.assign(
    performance = lambda d: np.where(
        d['points'] > d.groupby('team')['points'].transform('mean'),
        'Above Avg',
        'Below Avg'
    )
)
```

---

### **Chaining Multiple Columns**

You can define several derived columns at once:

```python
df = (
    df
    .assign(
        team_mean = lambda d: d.groupby('team')['points'].transform('mean'),
        team_max  = lambda d: d.groupby('team')['points'].transform('max'),
        diff_from_max = lambda d: d['team_max'] - d['points']
    )
)
```

Each column can refer to previously assigned ones within the same `.assign()` call.

---

### **Advanced Transform Examples**

| Goal                   | Example                                                                                         |
| ---------------------- | ----------------------------------------------------------------------------------------------- |
| Rolling mean per group | `lambda d: d.groupby('team')['points'].transform(lambda x: x.rolling(2, min_periods=1).mean())` |
| Normalization          | `lambda d: (d['points'] - d['points'].min()) / (d['points'].max() - d['points'].min())`         |
| Percentile rank        | `lambda d: d.groupby('team')['points'].transform(lambda x: x.rank(pct=True))`                   |
| Standard scaling       | `lambda d: (d['points'] - d['points'].mean()) / d['points'].std()`                              |

---

### **Why Use `assign()`**

* Keeps data pipelines **clean and functional**
* Avoids intermediate variables
* Works well with method chaining:

  ```python
  df.query("points > 5") \
    .assign(rank = lambda d: d.groupby('team')['points'].transform('rank')) \
    .sort_values(['team', 'rank'])
  ```

---

### **Quick Recap**

| Feature               | Description                                    | Example                                         |
| --------------------- | ---------------------------------------------- | ----------------------------------------------- |
| **Parameter**         | `**kwargs` – name & expression for each column | `df.assign(col = ...)`                          |
| **Conditional logic** | Use `np.where`                                 | `np.where(d['a'] > 0, 1, 0)`                    |
| **Group-wise ops**    | Use `groupby().transform()`                    | `d.groupby('team')['points'].transform('mean')` |
| **Multiple columns**  | Add several at once                            | `df.assign(x=..., y=...)`                       |
| **Chaining**          | Combine with query/sort                        | `.query(...).assign(...).sort_values(...)`      |

---


