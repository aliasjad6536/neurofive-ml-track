# Neurofive ML Track — Titanic Survival Prediction

This repo tracks my progress through the Neurofive Solutions Machine Learning internship, using the Titanic dataset ("Titanic - Machine Learning from Disaster") as a running project across three weeks.

Dataset source (direct CSV): `https://raw.githubusercontent.com/datasciencedojo/datasets/master/titanic.csv`

## Week 1 — Exploratory Data Analysis
- Loaded the dataset with `pandas.read_csv()` and inspected it with `.info()`, `.describe()`, `.head()`.
- Identified 891 rows, 12 columns, missing values in `Age`, `Cabin`, and `Embarked`, and split columns into numerical vs. categorical.

## Week 2 — Data Cleaning & Visualization
- Handled missing values with justification:
  - `Age` → filled with the **median age per `Pclass`** (age correlates with class, so a per-class median is more accurate than a single global fill).
  - `Embarked` → filled with the **mode** (only 2 missing rows, negligible impact).
  - `Cabin` → **dropped**, replaced with a binary `has_cabin` flag instead of fabricating 77% of a column.
- Detected outliers in `Fare` using a boxplot — real high-fare bookings (up to ~$500), kept rather than removed.
- Built 4 visualizations: histogram (Age), boxplot (Fare by class), bar chart (survival rate by sex), and a correlation heatmap.
- Concluded `Sex` is the strongest single driver of survival, followed by `Pclass`/`Fare`.

## Week 3 — Predict Survival (Classification Model)

**Approach:**
1. Reused the same cleaning steps from Week 2 (median-per-class `Age`, mode `Embarked`, `has_cabin` flag).
2. Dropped `PassengerId`, `Name`, and `Ticket` as raw features (index/high-cardinality text, not useful for a first model).
3. One-hot encoded `Sex` and `Embarked` using `pd.get_dummies(..., drop_first=True)` to avoid multicollinearity.
4. Split the data 80/20 into train/test sets with `train_test_split`, stratified on `Survived` to preserve class balance.
5. Trained a `LogisticRegression` model (`scikit-learn`) on the training set.
6. Evaluated on the held-out test set with `accuracy_score`, a confusion matrix, and a full classification report.

**Results:**
- **Accuracy: 81.01%**
- **Confusion matrix** (rows = actual, columns = predicted):

|                     | Predicted: Did Not Survive | Predicted: Survived |
|---------------------|:---------------------------:|:--------------------:|
| **Actual: Did Not Survive** | 96 | 14 |
| **Actual: Survived**        | 20 | 49 |

- Precision/recall: 0.83 precision / 0.87 recall for "Did Not Survive", 0.78 precision / 0.71 recall for "Survived".

**What the confusion matrix tells me:** the model is more reliable at spotting passengers who did *not* survive than at catching every actual survivor — 20 real survivors were misclassified as non-survivors (false negatives), more than the 14 non-survivors misclassified as survivors (false positives). For a safety-related prediction like this, false negatives are the more meaningful error to reduce, so accuracy alone understates where the model needs improvement.

**Next steps:** try a Random Forest or Gradient Boosting classifier as a stronger baseline, and engineer a `family_size` feature from `SibSp` + `Parch`.

## Files
- `titanic_eda.ipynb` — full notebook covering all three weeks (EDA → cleaning/visualization → classification model)
- `titanic.csv` — the dataset used throughout
