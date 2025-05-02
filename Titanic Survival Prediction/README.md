---

# 🚢 Titanic Survival Prediction

This project analyzes the Titanic passenger dataset to predict survival outcomes using a machine learning model. It includes data preprocessing, model training with a Random Forest classifier, and evaluation metrics for classification.

---

## 📁 Dataset

* **Source**: [Kaggle - brendan45774/test-file](https://www.kaggle.com/datasets/brendan45774/test-file)
* **File used**: `tested.csv` (a version of the Titanic dataset)

---

## 📊 Features Used

After preprocessing, the following features are used for modeling:

* `Pclass` – Passenger class (1st, 2nd, 3rd)
* `Sex` – Encoded as binary (0 = female, 1 = male)
* `Age` – Imputed with median values
* `SibSp` – # of siblings/spouses aboard
* `Parch` – # of parents/children aboard
* `Fare` – Ticket fare (imputed with median)
* `Embarked` – Port of Embarkation (encoded)

---

## ⚙️ Technologies Used

* Python 3
* Pandas, NumPy, Seaborn, Matplotlib
* Scikit-learn (RandomForestClassifier, metrics)

---

## 🚀 How to Run

1. **Download the dataset:**

```bash
kaggle datasets download brendan45774/test-file
```

2. **Extract and load data:**

```python
import zipfile
zip_ref = zipfile.ZipFile('/content/test-file.zip', 'r')
zip_ref.extractall('/content')
zip_ref.close()
```

3. **Run the notebook or Python script that includes:**

   * Data cleaning
   * Feature encoding
   * Model training and testing
   * Evaluation and visualization

---

## 📈 Model Performance

After training a Random Forest model with default parameters:

* **Accuracy**: \~1.00 (on test set)
* **Precision**: \~1.00
* **Recall**: \~1.00
* **F1 Score**: \~1.00
* **RMSE**: \~0.00

⚠️ Note: These perfect scores indicate potential overfitting. Model generalization should be further validated.

---

## 🧠 Feature Importance

Visualized using `model.feature_importances_`:

* Top contributing features (typically): `Sex`, `Fare`, `Pclass`, `Age`

---

## 🔍 Confusion Matrix

Heatmap visualization helps identify false positives/negatives.

---

## 📜 License

This project is provided for educational purposes. Dataset is publicly available via Kaggle under their terms of use.

---

## 🙋‍♂️ Author

* **Divyansh Saxena**
* GitHub: [@divyanshsaxena21](https://github.com/divyanshsaxena21)

---