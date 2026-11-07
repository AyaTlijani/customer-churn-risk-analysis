# Customer Churn Risk Analysis

**Which telecom customers are about to leave, and why?**

An end-to-end analysis of the [IBM Telco Customer Churn dataset](https://www.kaggle.com/datasets/blastchar/telco-customer-churn) (7,043 customers, 21 columns): cleaning, exploratory analysis of what drives churn, and a comparison of seven classifiers evaluated honestly on imbalanced data.

## What drives churn

- **Month-to-month contracts** churn far more than one- or two-year contracts.
- **Electronic check** payers leave more than customers on automatic payments.
- **Fiber optic** customers have a higher churn rate.
- **No tech support** and **no online security** both go with more churn.
- **Higher monthly charges** usually mean more churn, except for the very top-paying customers, who rarely leave.
- **Gender** makes no difference.

## Modelling

- **Imbalance:** far more customers stay than leave. A model that always says "stays" would look accurate and be useless, so models are compared on **F1 for the churn class**.
- **Resampling:** SMOTEENN rebalances the training set only. The test set (20%, stratified) keeps the real class balance, so the scores reflect real-world conditions.

| Model | Accuracy | Balanced accuracy | Precision | Recall | F1 (churn) |
|---|---|---|---|---|---|
| **Random Forest** | 0.730 | 0.756 | 0.495 | 0.810 | **0.615** |
| Logistic Regression | 0.710 | 0.743 | 0.474 | 0.813 | 0.598 |
| Gradient Boosting | 0.701 | 0.742 | 0.465 | 0.829 | 0.596 |
| Decision Tree | 0.719 | 0.738 | 0.482 | 0.778 | 0.595 |
| SVM | 0.716 | 0.734 | 0.479 | 0.770 | 0.591 |
| Naive Bayes | 0.716 | 0.733 | 0.478 | 0.770 | 0.590 |
| K-Nearest Neighbors | 0.677 | 0.729 | 0.444 | **0.840** | 0.580 |

The best models catch about 8 in 10 customers who actually leave, at the cost of more false alarms. That trade-off fits a retention campaign, where contacting a loyal customer costs little and losing one costs a lot.

## Run it

Download `WA_Fn-UseC_-Telco-Customer-Churn.csv` from Kaggle next to the notebook, then:

```bash
pip install pandas numpy scikit-learn imbalanced-learn seaborn matplotlib
jupyter notebook customerchurn.ipynb
```

## Stack

Python · pandas · scikit-learn · imbalanced-learn · seaborn · matplotlib
