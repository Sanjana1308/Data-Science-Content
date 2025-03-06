# Machine Learning - II

## **Random Forests and Ensemble Learning**

### **1️⃣ What is an Ensemble?**

An **ensemble** is a collection of models working together to make predictions, rather than relying on a single model. The idea is that multiple models can complement each other and improve overall performance.

- Ensembles can consist of **different types of models** (e.g., logistic regression, neural networks, decision trees).
- The most popular ensemble model is the **random forest**, which combines multiple decision trees.

### **2️⃣ Why Do Ensembles Work Better?**

Two key principles make ensembles effective:

### **🔹 Diversity**

- Just like a football team needs different players (defenders, attackers, goalkeeper), a strong ensemble needs diverse models.
- **Diversity ensures independence**, meaning that even if some models overfit, others can counterbalance the effect.
- In a **random forest**, trees are trained differently, ensuring that they don't all make the same mistakes.
- This reduces **variance**, making the model more **robust** and **less prone to overfitting**.

### **🔹 Acceptability**

- Each individual model in the ensemble should be **better than a random guess**.
- Even weak models (as long as they are slightly better than random) can contribute to an ensemble’s strength.

### **3️⃣ How to Ensure Diversity in an Ensemble?**

To build a strong ensemble, diversity can be introduced in the following ways:

✔ **Use different subsets of training data** (e.g., bagging).

✔ **Vary hyperparameters** across models.

✔ **Use different classifiers** (e.g., SVM, decision trees, neural networks).

✔ **Select different feature sets** for training.

### **4️⃣ Why Are Ensembles More Effective?**

- **Lower Variance:** Since individual models make different errors, combining them reduces the overall variance.
- **More Stability:** Ensembles are less sensitive to changes in training data.
- **Improved Generalization:** They perform better on unseen data than individual models.

---