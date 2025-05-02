---

# 💳 Credit Card Fraud Detection

This project builds a machine learning model to detect fraudulent credit card transactions. It uses the LightGBM classifier and SMOTE to address the imbalanced class distribution. The model is evaluated using standard classification metrics and ROC analysis.

---

## 📁 Dataset

* **Source**: [Kaggle - Credit Card Fraud Detection](https://www.kaggle.com/datasets/mlg-ulb/creditcardfraud)
* **Filename**: `creditcard.csv`
* **Details**:

  * Contains 284,807 transactions
  * Only 492 are fraudulent (\~0.17%)
  * Features are numerical and anonymized (PCA-transformed), except `Time`, `Amount`, and `Class`

---

## 💡 Project Goals

* Detect fraudulent transactions with high recall and precision
* Handle severe class imbalance using **SMOTE**
* Use **LightGBM**, a fast and efficient gradient boosting model
* Visualize performance using confusion matrix and ROC curve

---

## 📦 Libraries Used

* Python 3
* `pandas`, `numpy`, `matplotlib`, `seaborn`
* `scikit-learn` (metrics, preprocessing, model selection)
* `lightgbm`
* `imblearn` (for SMOTE)

---

## ⚙️ Workflow

### 1. Load and Inspect Data

* Checked for missing values (none)
* Visualized class imbalance (0: legit, 1: fraud)

### 2. Preprocessing

* Scaled features using `StandardScaler`
* Split into training and testing sets (80-20 split)
* Applied **SMOTE** to balance classes in the training set

### 3. Model Training

* Used `LightGBMClassifier` with `class_weight='balanced'`
* Trained on resampled data

### 4. Evaluation Metrics

* **Classification Report**: Precision, Recall, F1-score
* **ROC Curve & AUC**: To evaluate probability-based performance
* **Confusion Matrix**: For clear visualization of TP, TN, FP, FN

### 5. Feature Importance

* Plotted top 20 features by both **split** and **gain**

---

## 📈 Model Performance

* **Accuracy**: High, but not sufficient alone due to imbalance
* **Recall (Fraud class)**: \~0.81
* **Precision (Fraud class)**: \~0.62
* **AUC**: \~0.98 (strong performance)

⚠️ *Note: Model focuses on high **recall** to minimize missed frauds, even at the cost of some false positives.*

---

## 📊 Visualizations

* 📌 Class Distribution Plot
* 🔍 Confusion Matrix Heatmap
* 📈 ROC Curve
* 📊 Feature Importance (Split & Gain-based)

---

## 🧪 How to Run

1. **Download the dataset from Kaggle**:

```bash
kaggle datasets download mlg-ulb/creditcardfraud
```

2. **Extract and load the data**:

```python
import zipfile
with zipfile.ZipFile('creditcardfraud.zip', 'r') as zip_ref:
    zip_ref.extractall()
```

3. **Install required packages** (if needed):

```bash
pip install pandas numpy matplotlib seaborn scikit-learn lightgbm imbalanced-learn
```

4. **Run your script or notebook containing the workflow above**

---

## 📜 License

This dataset is made available by the ULB Machine Learning Group under the [Open Data Commons](https://www.kaggle.com/datasets/mlg-ulb/creditcardfraud). This project is for **educational** and **research** purposes.

---

## 👤 Author

* **Divyansh Saxena**
* GitHub: [@divyanshsaxena21](https://github.com/divyanshsaxena21)

---
