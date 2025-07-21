# High-Propensity Client List and Targeting Strategy

## Model Performance

### Propensity Models
*Logistic Regression* shows the highest average F1 test score at **0.3958**.

| Product | Model                      | Train Score | Test Score |
| :------ | :------------------------- | ----------: | ---------: |
| CC      | LogisticRegression         | 0.430523538 | 0.363485246 |
| MF      | LogisticRegression         | 0.377460265 | 0.303618551 |
| CL      | LogisticRegression         | 0.531906699 | 0.520328162 |
| CC      | GradientBoostingClassifier | 0.762154103 | 0.306962082 |
| MF      | GradientBoostingClassifier | 0.732276525 | 0.210947513 |
| CL      | GradientBoostingClassifier | 0.836018230 | 0.369634065 |
| CC      | RandomForestClassifier     | 0.945507166 | 0.273693248 |
| MF      | RandomForestClassifier     | 0.972392705 | 0.104475329 |
| CL      | RandomForestClassifier     | 0.950234000 | 0.344303802 |

### Revenue Models
*Linear Regression* shows the lowest average MSE test score at **9.1962**.

| Product | Model                      | Train Score | Test Score |
| :------ | :------------------------- | ----------: | ---------: |
| CC      | RandomForestRegressor      | 6.764924026 | 13.86897686 |
| MF      | RandomForestRegressor      | 2.627923557 | 6.609437715 |
| CL      | RandomForestRegressor      | 3.013420565 | 7.765028919 |
| CC      | GradientBoostingRegressor  | 3.086129623 | 18.36676641 |
| MF      | GradientBoostingRegressor  | 2.750509515 | 7.511624461 |
| CL      | GradientBoostingRegressor  | 4.074117145 | 8.109091538 |
| CC      | LinearRegression           | 15.89665516 | 13.2932659  |
| MF      | LinearRegression           | 6.671615158 | 6.534309087 |
| CL      | LinearRegression           | 7.641739444 | 7.760934421 |

**Strategy**:
- **Propensity Model**: Logistic Regression 
- **Revenue Model** : Linear Regression

Refer to run name: [`logreg-lr`](outputs/sample/logreg-lr).


## Credit Card:

**Which clients have a higher propensity to buy a credit card?**

![propensity for cc](/images/results/propensity_cc.png)

Top predictors:
* **TransactionsDebCash\_Card**
* **Count_OVD**
* **Count\_SA**
* **Count\_CC**

Active spenders with large transactions in debit cards, product familiarity / existing product penetration with savings and creditcards, who also have overdrafts are most likely to be credit card customers. However, there might be a case of simultaneity bias between OVD and credit card ownership given that people with credit cards are also likely to get overdrafts, so relationships might not be causal.

## Mutual Fund:

**Which clients have a higher propensity to buy mutual funds?**

![propensity for MF](/images/results/propensity_mf.png)

Top predictors:
* **Sex_M**
* **Count_MF**
* **Count_CL**
* **TransactionsCred**

Male customer and with liabilities likely to seek investment vehicles like mutual funds, likely showing a greater risk appetite or to offset debt burdens. Possible MF clientele may also be those who are financially active with their credit cards. Targeting clients for MF products should focus on consumers who have existing debt liabilities looking for returns, an increased risk appetite and also displaying financial familiarity with credit finance. 


## Consumer Loan:

**Which clients have a higher propensity to buy a consumer loan?**

![propensity for CL](/images/results/propensity_cl.png)

Top predictors:
* **TransactionsDebCash_Card**
* **Count_SA**
* **Count_MF**
* **TransactionsCred**

Clients who have active debit card transactions / deposits show positive correlation to the propensity of getting a credit loan. The profile also includes customers who have account diversification in savings account, credit and debit as well as mutual fund holdings.

**Which clients are to be targeted with which offer?**

Refer to [here.](/outputs/sample/logreg-lr/results/optimized_offers.csv)

**What would be the expected revenue based on your strategy?**

![expected revenue](/images/results/expected_revenue.png)

The total expected revenue is $693.62, assuming 100 contacts and cost to contact = 1. 

| Metric                  | Value   | 
| ----------------------- | ------- | 
| **Total Revenue**       | $693.62 |
| **Revenue per Contact** | $6.94   | 
| **Acceptance Rate**     | 79.53%  | 
| **ROI**                 | 6.94x   | 

**Distribution of Offers by Product**:

* **CL**: 64%
* **CC**: 31%
* **MF**: 5%

**Expected Revenue by Product**:

* CL ≈ $420.37
* CC ≈ $228.16
* MF ≈ $35.48

## Caveats
- Dataset is highly imbalanced with most records showing non-purchasers.
- Though logistic regression show to be the best model for propensity, Hosmer-Lemeshow-test show poor fit in eda. Better models should be experimented on. Plots with the log odds show violation of logistic assumptions (linearity to the log odds).