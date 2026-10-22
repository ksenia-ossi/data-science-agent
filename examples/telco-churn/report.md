# Telco churn prediction

**Task:** rank customers by churn risk so the retention team can contact the riskiest first. Target `Churn` (Yes = 26.5%).
**Data:** 7,043 customers, one row each, 19 usable features. `customerID` was dropped. There were no duplicates. `TotalCharges` had 11 blanks, all for customers with tenure 0, and I filled them with 0.
**Split:** stratified random 80/20 split (5,634 train, 1,409 test), done before any analysis. I used a random split because the data has no time column and no repeated customers. The test set was used once, at the end.
**Metrics:** ROC AUC, which measures ranking quality, and PR AUC, which suits the 27% positive rate. I also report top-k% lift, because the use case is contacting the riskiest customers first.

## Leaderboard (5-fold CV on train)
| model | ROC AUC | PR AUC |
|---|---|---|
| dummy (prior) | 0.500 | 0.265 |
| logistic regression | 0.846 | 0.662 |
| logistic regression + engineered features | 0.846 | 0.660 |
| LightGBM (default-ish) | 0.844 | 0.656 |
| random forest | 0.845 | 0.658 |
| **LightGBM, Optuna-tuned (30 trials)** | **0.850** | **0.669** |

The engineered features were service count, average charge per month of tenure, and the gap between monthly and average charge. They did not help.

## Chosen model and test result
I chose the tuned LightGBM because it had the best CV score. The margin over logistic regression is small, about 0.004 AUC against a fold standard deviation of about 0.012, so the models are close to equivalent. The tuned model is somewhat optimistic on CV because the same folds were used for tuning.

| | test ROC AUC | test PR AUC |
|---|---|---|
| dummy | 0.50 | 0.27 |
| logistic regression | 0.842 | 0.634 |
| **tuned LightGBM** | **0.848** | **0.664** |

**Lift on the test set (tuned LightGBM):**
| contacted | precision | share of churners caught | lift |
|---|---|---|---|
| top 10% | 0.76 | 29% | 2.9x |
| top 20% | 0.67 | 51% | 2.5x |
| top 30% | 0.61 | 68% | 2.3x |

At a 0.5 probability threshold, precision is 0.66 and recall is 0.52.

## Key drivers
Permutation importance (`outputs/figures/importance.png`) shows that contract type and tenure dominate. Internet service type, online security, payment method and tech support follow at much lower importance. Churn is highest for month-to-month contracts (43%), fiber optic customers (42%), electronic-check payers (46%) and short-tenure customers (`outputs/figures/churn_by_cat.png`, `outputs/figures/num_by_churn.png`). I found no leakage. No single feature came close to a perfect score.

## Limitations
- The data is one snapshot with a random split. I could not check how the model holds up over time.
- The model ranks risk. It does not show which customers a retention offer would actually change. Uplift modelling or a trial would be needed for that.
- Probabilities are not explicitly calibrated.
- The test set has 1,409 rows, so the metrics carry noticeable uncertainty, roughly ±0.02 AUC.
- Contract and tenure carry most of the signal, so a simple rule-based segmentation would capture much of the value.

## Next steps
- Choose the contact budget (the top-k%) from the cost of an offer against the value of a retained customer.
- Calibrate the probabilities.
- Validate on later cohorts.
- Run a retention A/B test.

**Saved model:** `outputs/model.joblib`. It is a full sklearn Pipeline that takes the raw feature columns, with the engineered features, encoding and LightGBM inside.
