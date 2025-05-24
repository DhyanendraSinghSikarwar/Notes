# **Object-Oriented Programming (OOP) in Python - MLOps Playlist Notes**  
*(For Data Scientists & ML Engineers)*  

---

## **1. Introduction to OOP**  
### **Why OOP?**  
- **Modularity**: Break code into reusable classes.  
- **Reusability**: Avoid rewriting code (DRY principle).  
- **Debugging**: Easier to trace errors in structured code.  
- **Industry Standard**: Used in professional software development.  

### **Key Terminologies**  
1. **Class**: Blueprint for creating objects (e.g., `Employee`, `Phone`).  
2. **Object**: Instance of a class (e.g., `sam = Employee()`).  
3. **Attributes (Data)**: Properties of an object (e.g., `name`, `salary`).  
4. **Methods (Functions)**: Actions an object can perform (e.g., `travel()`).  

---

## **2. Class & Object Basics**  
### **Example: Employee Class**  
```python
class Employee:
    def __init__(self, name, salary):  # Constructor (special method)
        self.name = name      # Attribute
        self.salary = salary  

    def travel(self, destination):    # Method
        print(f"{self.name} is traveling to {destination}")

# Creating an object
sam = Employee("Sam", 50000)
sam.travel("Kerala")  # Output: "Sam is traveling to Kerala"
```
- **`__init__`**: Automatically called when an object is created.  
- **`self`**: Refers to the object (e.g., `sam`).  

---

## **3. Why Python is OOP?**  
- **Everything in Python is an object** (even `int`, `list`, `str` are classes).  
- Example:  
  ```python
  lst = [1, 2, 3]
  print(type(lst))  # Output: <class 'list'>
  lst.append(4)     # Method of list class
  ```

---

## **4. Advantages of OOP**  
1. **Custom Data Types**: Create domain-specific types (e.g., `Algebra` class for `x + y = x + y`).  
2. **Code Reusability**: Reuse classes across projects.  
3. **Security**: Hide sensitive data (encapsulation).  
4. **Collaboration**: Teams can work on different classes.  

---

## **5. Key OOP Concepts**  
### **(1) Encapsulation**  
- **Hiding attributes/methods** using `__` (double underscore).  
```python
class BankAccount:
    def __init__(self, balance):
        self.__balance = balance  # Private attribute

    def get_balance(self):        # Getter method
        return self.__balance

account = BankAccount(1000)
print(account.get_balance())  # Works
print(account.__balance)      # Error: Attribute is private
```

### **(2) Getter & Setter**  
- Control access to private attributes.  
```python
class Employee:
    def __init__(self, name):
        self.__name = name

    @property
    def name(self):          # Getter
        return self.__name

    @name.setter
    def name(self, new_name):  # Setter
        self.__name = new_name

emp = Employee("Alice")
emp.name = "Bob"  # Uses setter
print(emp.name)   # Uses getter
```

### **(3) Static Methods**  
- **No `self` needed**: Called directly from the class.  
```python
class MathUtils:
    @staticmethod
    def add(a, b):
        return a + b

print(MathUtils.add(2, 3))  # Output: 5
```

### **(4) Inheritance**  
- **Reuse parent class attributes/methods**.  
```python
class Animal:                     # Parent class
    def speak(self):
        print("Animal speaks")

class Dog(Animal):                # Child class
    def speak(self):              # Method overriding
        print("Dog barks")

dog = Dog()
dog.speak()  # Output: "Dog barks"
```

#### **Types of Inheritance**  
1. **Single**: `Child → Parent`  
2. **Multilevel**: `Child → Parent → Grandparent`  
3. **Hierarchical**: `Child1 → Parent ← Child2`  
4. **Multiple**: `Child → (Parent1, Parent2)` *(Diamond Problem)*  

### **(5) Super() Keyword**  
- Access parent class methods.  
```python
class Parent:
    def greet(self):
        print("Hello from Parent")

class Child(Parent):
    def greet(self):
        super().greet()  # Calls Parent's greet()
        print("Hello from Child")

child = Child()
child.greet()
# Output: 
# "Hello from Parent"  
# "Hello from Child"
```

---

## **6. Practical Example: Social Media Platform**  
```python
class ChatBook:
    def __init__(self):
        self.__username = None  # Private attr

    def sign_up(self, email, password):
        self.__username = email
        self.__password = password

    def post(self, message):
        if self.__username:
            print(f"Posted: {message}")
        else:
            print("Sign in first!")

user = ChatBook()
user.sign_up("alice@example.com", "123")
user.post("Hello world!")  # Works
```

---
---
---
---
---

# **Again OOPS in detailed way**  
---
---
---

# **Complete Guide to Object-Oriented Programming (OOP) in Python**  
*(With Simple Explanations & Practical Examples for Data Scientists & ML Engineers)*  

---

## **1. What is OOP and Why Use It?**  

### **OOP = Organizing Code Like Real-World Objects**  
Imagine building a car:  
- **Class** = Blueprint (design of the car)  
- **Object** = Actual car made from the blueprint  
- **Attributes** = Properties (color, model)  
- **Methods** = Actions (drive, brake)  

### **Why OOP?**  
1. **Modularity**: Break code into small, reusable pieces (like Lego blocks).  
   - Example: A `Preprocessing` class can be reused in multiple ML projects.  
2. **Avoid Repetition**: Write once, use everywhere (DRY = Don’t Repeat Yourself).  
3. **Security**: Hide sensitive data (like passwords) using encapsulation.  
4. **Teamwork**: Different people can work on different classes.  

---

## **2. Classes & Objects: The Basics**  

### **Class = Recipe, Object = Dish**  
```python
class Employee:                   # Class (recipe)
    def __init__(self, name, salary):  # Constructor (initial setup)
        self.name = name          # Attribute (property)
        self.salary = salary

    def promote(self, raise_amount):   # Method (action)
        self.salary += raise_amount

# Creating objects (actual employees)
alice = Employee("Alice", 50000)  
bob = Employee("Bob", 60000)

alice.promote(10000)
print(alice.salary)  # Output: 60000
```

**Key Points**:  
- `__init__`: Automatically runs when creating an object (like turning on a car when you start it).  
- `self`: Refers to the object (e.g., `alice` or `bob`).  

---

## **3. Python is Fully OOP**  

### **Everything is an Object!**  
Even basic things like integers and lists are objects:  
```python
num = 10
print(type(num))  # Output: <class 'int'> (yes, numbers are objects!)

lst = [1, 2, 3]
lst.append(4)     # .append() is a method of the list class
```

---

## **4. The 4 Pillars of OOP**  

### **(1) Encapsulation: Hide Sensitive Data**  
- Use `__` to make attributes private.  
- Provide controlled access via methods.  

**Example: Bank Account**  
```python
class BankAccount:
    def __init__(self, balance):
        self.__balance = balance  # Private (hidden)

    def deposit(self, amount):
        if amount > 0:
            self.__balance += amount

    def get_balance(self):       # Getter (safe access)
        return self.__balance

account = BankAccount(1000)
account.deposit(500)
print(account.get_balance())  # Works: 1500
print(account.__balance)      # Error: Can’t access directly!
```

### **(2) Inheritance: Reuse Code**  
- Child classes inherit from parents.  

**Example: Animals**  
```python
class Animal:                     # Parent
    def speak(self):
        print("Some sound")

class Dog(Animal):                # Child
    def speak(self):              # Override parent method
        print("Bark!")

class Cat(Animal):                # Child
    def speak(self):
        print("Meow!")

dog = Dog()
dog.speak()  # Output: "Bark!"
```

### **(3) Polymorphism: Same Method, Different Behaviors**  
- Methods with the same name work differently for different classes.  

**Example: `+` Operator**  
```python
print(3 + 5)    # Adds numbers → 8
print("Hi" + "Alice")  # Concatenates strings → "HiAlice"
```

### **(4) Abstraction: Hide Complex Details**  
- Show only what’s necessary.  

**Example: ML Model Class**  
```python
class MLModel:
    def train(self, data):
        # Complex training logic hidden
        print("Model trained!")

    def predict(self, input):
        # Complex prediction logic hidden
        return 0.95

model = MLModel()
model.train(data)  # User doesn’t see how training works
```

---

## **5. Advanced OOP Concepts**  

### **(1) Static Methods (No `self` Needed)**  
- Used for utility functions.  

**Example: Math Operations**  
```python
class MathUtils:
    @staticmethod
    def add(a, b):
        return a + b

print(MathUtils.add(2, 3))  # Output: 5 (no object needed)
```

### **(2) Class Methods (Work on Class, Not Instance)**  
```python
class Pizza:
    def __init__(self, toppings):
        self.toppings = toppings

    @classmethod
    def margherita(cls):    # 'cls' refers to class, not object
        return cls(["cheese", "tomato"])

margherita = Pizza.margherita()  # Creates Pizza object
print(margherita.toppings)  # Output: ["cheese", "tomato"]
```

### **(3) Property Decorators (Advanced Getters/Setters)**  
```python
class Temperature:
    def __init__(self, celsius):
        self._celsius = celsius   # Protected attribute

    @property
    def celsius(self):           # Getter
        return self._celsius

    @celsius.setter
    def celsius(self, value):    # Setter
        if value < -273.15:
            raise ValueError("Too cold!")
        self._celsius = value

temp = Temperature(25)
temp.celsius = 30  # Uses setter
print(temp.celsius)  # Uses getter → 30
```

---

## **6. Real-World Example: Social Media App**  

```python
class SocialMediaUser:
    def __init__(self, username):
        self.username = username
        self.__password = None  # Private

    def set_password(self, password):
        if len(password) >= 8:
            self.__password = password
        else:
            print("Password too weak!")

    def post(self, message):
        if self.__password:
            print(f"{self.username} posted: {message}")
        else:
            print("Set password first!")

# Usage
user1 = SocialMediaUser("Alice")
user1.set_password("secure123")
user1.post("Hello world!")  # Works
```

---

## **7. Common Interview Questions**  

1. **What is `self`?**  
   - `self` refers to the object (like `alice` in `alice.promote()`).  

2. **Difference between `@staticmethod` and `@classmethod`?**  
   - `@staticmethod`: No access to class/instance.  
   - `@classmethod`: Takes `cls` (can modify class state).  

3. **What is method overriding?**  
   - Child class redefines a parent’s method (e.g., `Dog.speak()` overrides `Animal.speak()`).  

---

## **8. Summary Cheat Sheet**  

| Concept       | Example                          | Purpose                          |
|--------------|----------------------------------|----------------------------------|
| **Class**    | `class Car:`                    | Blueprint for objects            |
| **Object**   | `my_car = Car()`                | Instance of a class              |
| **Inheritance** | `class Tesla(Car):`           | Reuse parent class code          |
| **Encapsulation** | `self.__password`          | Hide sensitive data              |
| **Polymorphism** | `+` works for numbers & strings | One interface, multiple forms    |

---
