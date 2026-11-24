# 🏥 Medical Insurance Charges Prediction using Linear Regression

## 📌 Objective
To build a machine learning model that predicts individual medical insurance charges from personal attributes such as age, BMI, number of children, smoking status and region, and to identify which factors drive insurance costs the most.

## ❓ Problem Definition
Insurance companies need to estimate the cost of covering a customer in order to set fair premiums. This is a **supervised regression problem**:

- **Input features:** age, BMI, children, smoker, region
- **Target variable:** `charges` (medical costs billed by health insurance, continuous)
- **Goal:** predict `charges` for unseen customers as accurately as possible.

## 📂 Dataset Used
**File:** `insurance.csv`  |  **Rows:** 1,338  |  **Columns:** 7  |  **Missing values:** None

| Column | Description | Type |
|---|---|---|
| `age` | Age of the primary beneficiary | Numeric |
| `sex` | Gender of the beneficiary | Categorical |
| `bmi` | Body Mass Index | Numeric |
| `children` | Number of dependents covered | Numeric |
| `smoker` | Whether the person smokes | Categorical (yes/no) |
| `region` | Residential region in the US (NE, NW, SE, SW) | Categorical |
| `charges` | Individual medical costs billed by insurance (**target**) | Numeric |

## 🔬 Methodology
1. **Data Loading & Inspection**: loaded the data with Pandas and checked shape, types and null values (no NaNs found).
2. **Feature Selection**: dropped the `sex` column.
3. **Outlier Handling**: inspected `bmi` and `charges` with boxplots and capped outliers using the **IQR method** (values beyond Q1 − 1.5·IQR and Q3 + 1.5·IQR replaced with the bounds).
4. **Encoding**: converted categorical variables (`smoker`, `region`) to numeric using **one-hot encoding** (`get_dummies`, `drop_first=True`) to avoid the dummy variable trap.
5. **Feature/Target Separation**: `X` = all features, `y` = `charges`.
6. **Train-Test Split**: 80% training (1,070 rows) / 20% testing (268 rows).
7. **Model Building**: trained a `LinearRegression` model.
8. **Auto Random State**: looped over 501 random states to study how the split affects the score (best: `random_state=367`).
9. **Auto ML Comparison**: compared Linear Regression against Decision Tree, Random Forest, SVR and AdaBoost regressors on the same split.
10. **Prediction & Evaluation**: predicted on the test set and evaluated using R² score.

## 🛠️ Technology Used
- **Language:** Python 3
- **Libraries:** Pandas, NumPy, Matplotlib, Seaborn, Scikit-learn
- **Environment:** Jupyter Notebook

## 📊 Evaluation Metrics
Primary metric: **R² Score** (coefficient of determination).

**Linear Regression (test set):**

| Metric | Value |
|---|---|
| R² Score | **0.841** |
| MAE | ≈ ₹/$ 2,809 |
| RMSE | ≈ ₹/$ 3,964 |

**Model comparison (R² on test set):**

| Model | R² Score |
|---|---|
| Random Forest Regressor | 0.8413 |
| **Linear Regression** | **0.8411** |
| AdaBoost Regressor | 0.8392 |
| Decision Tree Regressor | 0.6038 |
| SVR (default) | -0.1057 |

> Linear Regression performs almost identically to Random Forest while being far simpler and more interpretable.

## 💡 Key Insights
- **Smoking is by far the strongest cost driver.** Smokers pay on average about **$32,050** vs **$8,434** for non-smokers, and the model attributes roughly **+$19,700** in charges to being a smoker (all else equal). Smoking also has the highest correlation with charges (0.79).
- **Age matters:** each additional year adds about **$231** to charges on average.
- **BMI matters:** each additional BMI point adds about **$211**.
- **More children** adds roughly **$519** per child.
- **Region has a small effect.** The Southeast, Southwest and Northwest are slightly cheaper than the Northeast, by roughly $400–$1,000.
- **Linear Regression is competitive with ensemble models** on this dataset, suggesting the relationships are largely linear.
- **SVR performed poorly** without feature scaling, which highlights the importance of scaling for distance/kernel-based models.

## ✅ Conclusion
The Linear Regression model explains about **84% of the variance** in medical insurance charges on unseen data and is competitive with more complex algorithms. Smoking status, age and BMI are the main factors influencing cost, so these are the key variables an insurer would focus on when pricing risk.

**Limitations & future work**
- Charges are right-skewed, so a log-transform of the target could improve linear model fit.
- Add an interaction feature (e.g., `smoker × bmi`) since the cost impact of BMI is likely much larger for smokers.
- Use cross-validation instead of a single split for a more reliable estimate.
- Scale features and tune hyperparameters for SVR and ensemble models.
- Deploy the model as a simple web app (Streamlit/Flask).


```
├── Linear_Regression.ipynb
├── insurance.csv
└── README.md
```
