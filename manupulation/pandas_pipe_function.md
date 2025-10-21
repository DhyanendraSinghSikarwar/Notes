

## 📘 **Pandas `.pipe()` Function — Complete Notes**

### **Purpose**

`.pipe()` lets you **pass a DataFrame through custom functions** in a method chain — just like a data pipeline.
It’s great for applying reusable transformations while keeping your code clean, readable, and functional.

---

### **Syntax**

```python
DataFrame.pipe(func, *args, **kwargs)
```

---

### **Parameters**

| Parameter      | Description                                 | Example                          |
| -------------- | ------------------------------------------- | -------------------------------- |
| **`func`**     | Function to apply to the DataFrame          | `lambda d: d.sort_values('age')` |
| **`*args`**    | Positional arguments passed to the function | `pipe(my_func, 10, 20)`          |
| **`**kwargs`** | Keyword arguments passed to the function    | `pipe(my_func, col='sales')`     |

---

### **Returns**

The **result** of the function call — often another DataFrame, but could be any object depending on the function.

---

### **1. Basic Example**

```python
def clean_names(df):
    df.columns = df.columns.str.strip().str.lower().str.replace(' ', '_')
    return df

df = df.pipe(clean_names)
```

→ The DataFrame is passed into `clean_names(df)` automatically.

Equivalent to:

```python
clean_names(df)
```

But `.pipe()` keeps it chainable.

---

### **2. Why Use `.pipe()`**

Without `pipe()`:

```python
df = clean_names(df)
df = add_discount(df)
df = filter_valid_rows(df)
```

With `pipe()`:

```python
(df
 .pipe(clean_names)
 .pipe(add_discount)
 .pipe(filter_valid_rows)
)
```

Cleaner, composable, and easy to reorder or reuse steps.

---

### **3. Using Inline Lambda Functions**

You can use small transformations directly inside `pipe()`:

```python
(df
 .pipe(lambda d: d.query("sales > 100"))
 .pipe(lambda d: d.assign(profit=lambda x: x['sales'] - x['cost']))
)
```

Equivalent to chaining `.query()` and `.assign()`, but can handle **custom logic** easily.

---

### **4. Passing Extra Arguments**

You can pass extra parameters to the function:

```python
def top_n(df, col, n=5):
    return df.nlargest(n, col)

df.pipe(top_n, col='sales', n=3)
```

→ Internally calls `top_n(df, col='sales', n=3)`.

---

### **5. Using `pipe()` for Multi-step Calculations**

You can chain intermediate transformations while keeping readability:

```python
def compute_profit(df):
    return df.assign(profit=lambda d: d['sales'] - d['cost'])

def filter_profitable(df, min_profit):
    return df.query("profit > @min_profit")

(df
 .pipe(compute_profit)
 .pipe(filter_profitable, min_profit=50)
 .sort_values('profit', ascending=False)
)
```

---

### **6. Combining with `.assign()` and `.query()`**

They fit together nicely for expressive pipelines:

```python
(df
 .pipe(lambda d: d.query("region == 'East'"))
 .assign(revenue=lambda d: d['sales'] * d['price'])
 .pipe(lambda d: d.groupby('region', as_index=False)['revenue'].sum())
)
```

---

### **7. `pipe()` with Functions Returning Other Objects**

You can even use `pipe()` to move into a completely different context:

```python
(df
 .pipe(lambda d: d[['sales', 'profit']])
 .pipe(lambda d: d.corr())     # returns a correlation matrix
)
```

→ The chain continues as long as each step returns a valid object for the next.

---

### **8. Example: Parameterized Transformation**

Suppose you want to apply a generic normalization:

```python
def normalize(df, col):
    df[col] = (df[col] - df[col].mean()) / df[col].std()
    return df

df.pipe(normalize, col='sales')
```

→ Passes the DataFrame and `col` to the function.

---

### **9. Advanced Pattern — Tuple Unpacking**

You can pipe multiple outputs if your function returns a tuple:

```python
def split_data(df):
    train = df.sample(frac=0.8)
    test = df.drop(train.index)
    return train, test

train, test = df.pipe(split_data)
```

---

### **10. Real-world Example (Clean Data Pipeline)**

```python
(df
 .pipe(lambda d: d.rename(columns=str.lower))
 .pipe(lambda d: d.query("sales > 0"))
 .assign(profit=lambda d: d['sales'] - d['cost'])
 .pipe(lambda d: d.groupby('region', as_index=False)['profit'].mean())
)
```

→ One continuous flow — no temporary variables.

---

### **Quick Recap**

| Feature                | Description                          | Example                               |
| ---------------------- | ------------------------------------ | ------------------------------------- |
| **Purpose**            | Chain custom functions cleanly       | `df.pipe(func)`                       |
| **Pass args**          | Add parameters to the function       | `df.pipe(my_func, 10, name='abc')`    |
| **Lambda inline**      | Small inline logic                   | `.pipe(lambda d: d.query("x > 0"))`   |
| **Reusable pipelines** | Combine with `.assign()`, `.query()` | `.pipe(clean).assign(...).query(...)` |
| **Tuple unpacking**    | Handle train-test splits, etc.       | `train, test = df.pipe(split_data)`   |

---

### **When to Use `.pipe()`**

* You want to build **modular, functional-style data pipelines**.
* You use **custom functions** repeatedly on DataFrames.
* You’re combining `.query()` / `.assign()` / `.sort_values()` logically.
* You want to **avoid mutation and temporary variables**.

---

### **Concept Summary**

Think of `.pipe()` as:

> “Take this DataFrame, pass it through a function, and give me back the result — all without breaking the chain.”

---
