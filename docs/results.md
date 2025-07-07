# High-Propensity Client List and Targeting Strategy

## Model Performance

### Propensity Models
*Logistic Regression* shows the highest average F1 test score at **0.2912**.

| Product | Model                      | Train Score | Test Score |
| :------ | :------------------------- | ----------: | ---------: |
| CC      | LogisticRegression         | 0.361993531 | 0.278824759 |
| MF      | LogisticRegression         | 0.298809960 | 0.193975182 |
| CL      | LogisticRegression         | 0.484243678 | 0.400904249 |
| CC      | GradientBoostingClassifier | 0.761225077 | 0.295113161 |
| MF      | GradientBoostingClassifier | 0.732276525 | 0.221818182 |
| CL      | GradientBoostingClassifier | 0.839665384 | 0.352002120 |
| CC      | RandomForestClassifier     | 0.726986654 | 0.233897269 |
| MF      | RandomForestClassifier     | 0.722587974 | 0.088282203 |
| CL      | RandomForestClassifier     | 0.876338500 | 0.232403032 |

### Revenue Models
*Gradient Boosting Regressor* shows the highest average MSE test score at **11.1356**.

| Product | Model                      | Train Score | Test Score |
| :------ | :------------------------- | ----------: | ---------: |
| CC      | RandomForestRegressor      | 5.873908508 | 13.70508232 |
| MF      | RandomForestRegressor      | 2.722701099 | 6.654974329 |
| CL      | RandomForestRegressor      | 2.973501447 | 7.713070575 |
| CC      | GradientBoostingRegressor  | 3.118113879 | 17.86913947 |
| MF      | GradientBoostingRegressor  | 2.750509515 | 7.359297701 |
| CL      | GradientBoostingRegressor  | 4.071483835 | 8.178325553 |
| CC      | LinearRegression           | 15.756555950| 13.57520084 |
| MF      | LinearRegression           | 6.631619542 | 6.624548643 |
| CL      | LinearRegression           | 7.459320198 | 8.406698631 |

**Strategy**:
- **Propensity Model**: Logistic Regression 
- **Revenue Model** : Gradient Boosting Regressor. 

Refer to run name: [`logreg-gbr`](outputs/sample/logreg-gbr).


## Credit Card:

**Which clients have a higher propensity to buy a credit card?**

![propensity for cc](/images/results/propensity_cc.png)

Clients who have a high number of live overdrafts, credit transactions and debit cash transactions via card are most likely to sign up for credit cards. Targeting clients for consumer cards should focus on consumers with overdrafts who likely rely on short-term credit, with transaction activity being the next strong behavioral indicator. However, there might be a case of simultaneity bias between OVD and credit card ownership given that people with credit cards are also likely to get overdrafts, so relationships might not be causal.

## Mutual Fund:

**Which clients have a higher propensity to buy mutual funds?**

![propensity for MF](/images/results/propensity_mf.png)

Clients who have a high number of consumer loans are the strongest predictor of mutual-fund sales, suggesting that clients with more loans are likelier to purchase mutual funds. High number of monthly debit cashless-via-card transactions ranks second in correlation. The count of mutual funds also shows a trivial and expected correlation. High credit transactions and payment-order debits show a lesser correlation. Targeting clients for mutual funds should focus on consumers with existing debt who might be open to using financial instruments to balance liabilities with investments (using financial returns to offset interest cost of debt). High credit usage also signals financial familiarity with digital finance.

## Consumer Loan:

**Which clients have a higher propensity to buy a consumer loan?**

![propensity for CL](/images/results/propensity_cl.png)

Clients with more savings and current accounts most likely to have consumer loans, show higher predictive value compared to the count of existing consumer loans (trivial correlation). The number of debit and credit transactions also show some positive relationship to having a consumer loan. The targeting strategy should focus on clients with multiple active accounts and active users with frequent debit and cash transactions.

**Which clients are to be targeted with which offer?**

Refer to [here.](/outputs/sample/logreg-gbr/results/optimized_offers.csv)

**What would be the expected revenue based on your strategy?**

![expected revenue](/images/results/expected_revenue.png)

The total expected revenue is $1,150.50, assuming 100 contacts and cost to contact = 1. 

| Metric                  | Value     | 
| ----------------------- | --------- |
| **Total Revenue**       | 1,150.50 |
| **Revenue per Contact** | 11.50    |
| **Acceptance Rate**     | 63.21%    |
| **ROI**                 | 11.50x    |

**Distribution of Offers by Product**:

* CL: 66%
* CC: 21%
* MF: 13%

**Expected Revenue by Product**:

* CL ≈ 570.47
* MF ≈ 175.47
* CC ≈ 686.78

## Caveats
- Dataset is highly imbalanced with most records showing non-purchasers.
- Though logistic regression show to be the best model for propensity, Hosmer-Lemeshow-test show poor fit in eda. Better models should be experimented on. Plots with the log odds show violation of logistic assumptions (linearity to the log odds). 