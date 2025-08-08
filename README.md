# 🏭 Capstone Project 2 – Semiconductor Manufacturing Yield Prediction  

## 📌 Project Description  
This project focuses on **predicting pass/fail yield outcomes** in a semiconductor manufacturing process using high-dimensional sensor data.  
The dataset contains readings from **591 different sensors/features** per production entity. Not all signals are equally important, so the project also aims to perform **feature selection** to identify the most relevant signals impacting yield.  

The ultimate goal is to:
- Build accurate machine learning models for yield prediction.
- Reduce dimensionality for efficiency.
- Provide insights for process improvement to increase throughput, reduce production costs, and enhance quality control.  

---

## 📂 Dataset  

**File:** `signal-data.csv`  
- **Shape:** 1567 rows × 592 columns  
- **Features:** 591 continuous sensor readings  
- **Target:** Pass/Fail outcome  
  - `-1` → Pass  
  - `1` → Fail  

---



## 🛠 Procedure 

1. **Data Understanding & Cleaning**  
   - Loaded `sensor-data.csv` and inspected shape, types, and missing values.  
   - Removed irrelevant columns based on domain knowledge.  
   - Filled missing values using median imputation.  

2. **Exploratory Data Analysis (EDA)**  
   - Checked class balance and visualized target distribution.  
   - Performed univariate, bivariate, and multivariate analysis using plots and statistical summaries.  

3. **Data Preprocessing**  
   - Split predictors (X) and target (y).  
   - Applied **SMOTE** to balance classes.  
   - Standardized features using StandardScaler.  
   - Performed train-test split and validated data consistency.  

4. **Model Training & Optimization**  
   - Trained multiple models (**Random Forest, SVM, Naïve Bayes**).  
   - Used **cross-validation** for performance estimation.  
   - Applied **GridSearchCV** for hyperparameter tuning.  

5. **Model Evaluation**  
   - Compared models on accuracy, precision, recall, and F1-score.  
   - Selected **Random Forest** as the final model.  
   - Saved the model using `joblib` for future use.  

---



## 🛠 Tech Stack & Libraries  
- **Python 3.x**  
- NumPy  
- Pandas  
- Matplotlib & Seaborn  
- Scikit-learn  
- Imbalanced-learn (SMOTE)  
- Joblib / Pickle  



## 📷 Sample Output Images  

### **1. Target Distribution Before & After SMOTE**  
![Target Distribution](images/target_distribution.png)  

### **2. Feature Importance (Random Forest)**  
![Feature Importance](images/feature_importance.png)  

### **3. Confusion Matrix – Best Model**  
![Confusion Matrix](images/confusion_matrix.png)  

### **4. Model Comparison Chart**  
![Model Comparison](images/model_comparison.png)  

