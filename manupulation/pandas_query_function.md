

## 📘 **Pandas `.query()` Function — Complete Notes**

### **Purpose**

`DataFrame.query()` lets you **filter rows** in a pandas DataFrame using a **string-based expression**, similar to how you’d write SQL `WHERE` conditions.
It makes filtering more readable and often faster than using traditional boolean indexing.

---

### **Syntax**

```python
DataFrame.query(expr, inplace=False, **kwargs)
```

---

### **Parameters**

| Parameter      | Description                                                                      | Example                            |
| -------------- | -------------------------------------------------------------------------------- | ---------------------------------- |
| **`expr`**     | A string expression to evaluate. Must return a boolean mask.                     | `"age > 30 and city == 'Delhi'"`   |
| **`inplace`**  | If `True`, modifies the DataFrame in place. Default is `False`.                  | `df.query("x > 10", inplace=True)` |
| **`**kwargs`** | Used to pass local variables or column names that conflict with Python keywords. | `df.query("score > @min_score")`   |

---

### **Returns**

A **filtered DataFrame** (unless `inplace=True`).

---

### **1. Basic Filtering**

```python
df.query("age > 25")
```

Equivalent to:

```python
df[df['age'] > 25]
```

---

### **2. Multiple Conditions**

```python
df.query("age > 25 and city == 'Delhi'")
```

Logical operators:

* `and` → `&`
* `or` → `|`
* `not` → `~`

So this is cleaner than:

```python
df[(df['age'] > 25) & (df['city'] == 'Delhi')]
```

---

### **3. Using Variables with `@`**

If you want to use a Python variable in the query, prefix it with `@`:

```python
min_age = 30
df.query("age >= @min_age")
```

---

### **4. Filtering on Multiple Columns**

You can use expressions involving multiple columns:

```python
df.query("sales > cost * 1.2")
```

→ Selects rows where `sales` are at least 20% higher than `cost`.

---

### **5. Query with String Columns**

```python
df.query("region == 'East' or region == 'West'")
```

Quotes around string values are mandatory inside the expression.

---

### **6. Membership Tests (`in`, `not in`)**

```python
df.query("city in ['Delhi', 'Mumbai']")
df.query("team not in ['A', 'C']")
```

---

### **7. Filtering Using Index**

You can also reference the index directly:

```python
df.query("index > 2")
```

If your index has a name, you can use it:

```python
df.query("id == 101")
```

---

### **8. Combining with `.assign()` and Method Chaining**

`.query()` shines in clean method chains:

```python
(df
 .query("points > 10 and team != 'C'")
 .assign(zscore=lambda d: (d['points'] - d['points'].mean()) / d['points'].std())
 .sort_values('zscore', ascending=False)
)
```

→ Reads almost like a declarative data pipeline.

---

### **9. Handling Column Names with Spaces or Special Characters**

If column names contain spaces or punctuation, wrap them in **backticks**:

```python
df = pd.DataFrame({'player name': ['A', 'B'], 'score': [10, 20]})
df.query("`player name` == 'A'")
```

---

### **10. Example: Conditional Filtering with Variables**

```python
min_points = 10
allowed_teams = ['A', 'B']

df.query("points >= @min_points and team in @allowed_teams")
```

---

### **11. Inplace Modification**

```python
df.query("points > 5", inplace=True)
```

Filters the DataFrame directly without returning a new one.

---

### **Performance Note**

`query()` uses **NumExpr** (if installed) for faster evaluation of large numerical expressions, making it more efficient than chained boolean indexing.

---

### **Quick Recap**

| Feature                           | Description                                | Example                                      |
| --------------------------------- | ------------------------------------------ | -------------------------------------------- |
| **Filter rows by expression**     | Uses string-based logical conditions       | `df.query("age > 30")`                       |
| **Use variables**                 | Prefix with `@`                            | `df.query("sales > @threshold")`             |
| **Logical operators**             | `and`, `or`, `not`                         | `"x > 0 and y < 10"`                         |
| **Membership**                    | `in`, `not in`                             | `"city in ['Delhi', 'Mumbai']"`              |
| **Index filtering**               | Use `index` or index name                  | `"index < 5"`                                |
| **Backticks for special columns** | Wrap names with spaces                     | ``df.query("`player name` == 'A'")``         |
| **Chaining**                      | Combine with `.assign()`, `.sort_values()` | `df.query(...).assign(...).sort_values(...)` |

---

### **When to Use `.query()`**

* You want SQL-like filtering syntax.
* You’re building readable, method-chained pipelines.
* You’re working with large numeric datasets (faster with NumExpr).
* You need clean conditions with multiple logical operators.

---

