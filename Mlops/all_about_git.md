# **MLOps Series - Day 2: Git & GitHub Comprehensive Guide**  


---

## **1. Introduction to Git**  
### **What is Git?**  
- **Git** is a **Version Control System (VCS)** or **Source Code Manager (SCM)**.  
- Tracks changes in code, enables collaboration, and maintains history.  

### **Why Git for Data Scientists/ML Engineers?**  
1. **Bug Fixing & Rollbacks**  
   - Save snapshots (commits) to revert to previous versions if bugs occur.  
   - Example: Roll back to a pre-processing stage if model performance degrades.  

2. **Collaboration**  
   - Multiple team members can work on the same project.  
   - Example: One handles data cleaning, another works on feature engineering.  

3. **Non-linear Development (Branching)**  
   - Create parallel branches for experiments (e.g., trying different algorithms).  

---

## **2. Git Architecture & Key Concepts**  
### **Stages in Git**  
1. **Workspace**: Local directory where you edit files.  
2. **Staging Area**: Files marked for tracking (`git add`).  
3. **Local Repository**: Committed snapshots (`git commit`).  
4. **Remote Repository**: Cloud storage (e.g., GitHub).  

**Analogy**:  
- **Workspace** → Grocery store (untracked items).  
- **Staging Area** → Shopping bag (selected items).  
- **Local Repo** → Home fridge (saved items).  
- **Remote Repo** → Cloud storage (shared with others).  

---

## **3. Basic Git Commands**  
### **Setup & Initialization**  
```bash
git config --global user.name "YourName"  
git config --global user.email "your@email.com"  
git init  # Initialize a Git repo
```

### **Tracking Changes**  
```bash
git status              # Check untracked/modified files  
git add <file>          # Add file to staging  
git add .               # Add all files  
git commit -m "Message" # Save snapshot
```

### **View History**  
```bash
git log                 # Detailed commit history  
git log --oneline       # Compact history  
git log --graph --all   # Visualize branches
```

### **Branching & Merging**  
```bash
git branch <name>       # Create branch  
git checkout <branch>   # Switch branch  
git merge <branch>      # Merge branch into current
```

### **Undoing Changes**  
```bash
git restore <file>      # Discard unstaged changes  
git reset --hard HEAD   # Revert to last commit
```

---

## **4. Branching & Merging Deep Dive**  
### **Scenario**  
- **Master Branch**: Main project line.  
- **Helper Branch**: Experiment with new features.  

**Steps**:  
1. Create branch:  
   ```bash
   git branch helper  
   git checkout helper
   ```
2. Make changes and commit:  
   ```bash
   git commit -m "Feature X added"
   ```
3. Merge back to master:  
   ```bash
   git checkout master  
   git merge helper
   ```

### **Conflict Resolution**  
- Occurs when two branches modify the same line.  
- VS Code shows:  
  ```plaintext
  <<<<<<< HEAD  
  Current branch code  
  =======  
  Incoming branch code  
  >>>>>>> helper
  ```
- Manually edit to resolve, then commit.  

---

## **5. Remote Repositories (GitHub)**  
### **Push Local Code to GitHub**  
1. Create a repo on GitHub.  
2. Link local repo:  
   ```bash
   git remote add origin <GitHub-URL>
   ```
3. Push code:  
   ```bash
   git push origin master
   ```

### **Clone & Collaborate**  
```bash
git clone <GitHub-URL>   # Download repo  
git pull origin master   # Fetch updates
```

---

## **6. Advanced Topics**  
### **.gitignore**  
- Exclude files (e.g., credentials, large datasets):  
  ```plaintext
  # .gitignore example  
  *.csv  
  .env
  ```

### **Tagging (Software Versioning)**  
- Mark releases (e.g., `v1.0.0`):  
  ```bash
  git tag -a v1.0.0 -m "First stable release"  
  git push origin v1.0.0
  ```

---

## **7. Key Takeaways**  
1. **Git** = Local version control. **GitHub** = Cloud hosting.  
2. **Branching** enables parallel experimentation.  
3. **Merge conflicts** are resolved manually.  
4. **.gitignore** prevents tracking unwanted files.  

**Pro Tip**: Use `git log --graph --all` to visualize complex branch histories.  

---

### **Next Steps**  
- Practice with real projects (e.g., ML model deployments).  
- Explore CI/CD pipelines (GitHub Actions).  

--- 
