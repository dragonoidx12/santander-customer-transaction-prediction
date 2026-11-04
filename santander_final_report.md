# Santander Customer Transaction Prediction

## Project Objective
Predict whether a customer will make
a specific transaction in the future.

## Dataset
Training: 200,000 observations
Test: 200,000 observations
Features: 200 anonymous numerical variables
Positive target rate: approximately 10.05%

## Data Preparation
- Exploratory Data Analysis
- Missing Value Inspection
- Target Distribution
- Correlation Analysis
- Stratified Validation

## Models
- Baseline
- Logistic Regression
- Gaussian Naive Bayes
- Random Forest
- HistGradientBoosting

## Initial Model Results

| Model | ROC-AUC | PR-AUC |
|---|---:|---:|
| Gaussian NB | 0.887335 | 0.574573 |
| HistGradientBoosting | 0.867282 | 0.517770 |
| Logistic Regression | 0.859547 | 0.499372 |
| Random Forest | 0.810767 | 0.401552 |
| Baseline | 0.500000 | 0.100500 |

## Optimization
- QuantileTransformer
- Normal distribution transformation
- GaussianNB variance smoothing optimization
- Internal training split for parameter selection

Selected parameters:
Transformation: QuantileNormal
var_smoothing: 0.001

## Final Results
Selected model: Optimized Gaussian Naive Bayes
Validation ROC-AUC: 0.888100
Validation PR-AUC: 0.575299

ROC-AUC improvement: 0.000765
PR-AUC improvement: 0.000726

## Conclusion
Optimized Gaussian Naive Bayes achieved
the highest validation ROC-AUC.

## Limitations
- Anonymized features have no business interpretation.
- Model selection used one validation dataset.
- The improvement is small.
- No independent final test evaluation.
- No official Kaggle leaderboard score.
- Scores are not guaranteed to be calibrated.
- Educational research only.

## Competition
https://www.kaggle.com/competitions/santander-customer-transaction-prediction
