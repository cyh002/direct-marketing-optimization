# **Executive Summary**
For detailed results, please read [`results.md`](results.md).

## 1. Business Objective  
With a limited marketing budget, the organization must decide which subset of its customer base should receive a single targeted offer, either a Consumer Loan (CL), Credit Card (CC), or Mutual Fund (MF) to maximize total net revenue. Contacts are costly (assumed unit cost = 1) and restricted by a campaign limit (e.g. 100 offers).

## 2. Strategy 
For each product (Consumer Loan, Credit Card, Mutual Fund), we develop two predictive models: a propensity model estimating each customer’s likelihood of purchase, and a revenue model forecasting the expected revenue upon conversion. We compute expected value by multiplying purchase probability by predicted revenue, then solve an optimization problem to select the top 100 customer-product offers that maximize total expected revenue under our campaign constraints.

## 3. Key Findings & Impact  
1. **Optimal Offer Mix & Revenue**  
   - Under a 100‐contact campaign, the optimized strategy yields **\$1,150.50** total expected revenue (ROI = 11.5×; acceptance rate ≈63%).  
   - Revenue per contact is **\$11.50**, significantly above contact cost, showing potential value in this strategy. 

2. **Product Prioritization**  
   - **Consumer Loans (CL)** represent **66%** of chosen offers and drive ≈\$570 of revenue. Clients with multiple active deposit accounts and high transaction activity are prime targets.  
   - **Credit Cards (CC)** account for **21%** of offers, contributing ≈\$687 in revenue. High overdraft counts and frequent card transactions characterize the top prospects.  
   - **Mutual Funds (MF)** make up **13%** of offers, delivering ≈\$175. Ideal targets are clients with existing loan exposure and healthy transaction volumes.

## 4. Strategic Recommendations  

- **Allocate Budget by Offer Type**  
  Concentrate ~2/3 of contacts on Consumer Loan prospects, ~1/5 on Credit Card prospects, and remaining on Mutual Fund prospects to mirror expected‐value optimization. Consumer Loans seems like the easiest to sell. 

- **Focus on Behavioral Signals and Financial Activity**  
  Transaction patterns (overdrafts, credit/debit activity) and existing product holdings (accounts, loans) are strong indicators of receptivity. 

## 5. Limitations & Assumptions  
- **Model Accuracy & Data Imbalance**  
  The underlying propensity models achieve modest F1‐scores (~0.29) on test data due to class imbalance. Predictions may misclassify some high-value prospects or overlook others.

- **Causality vs. Correlation**  
  Features such as overdraft counts and existing loans correlate with product purchase but may reflect simultaneity (e.g. existing credit card holders also hold overdrafts). 

- **Cost to Contact & Uniform Acceptance Cost**  
  We assume a unit contact cost of 1 and identical cost across channels. Real‐world costs may vary by channel or segment.

- **One-Offer-Per-Customer Constraint**  
  Each customer is limited to a single offer, which precludes capturing bundling or cross‐sell synergies. Future work should explore multi‐offer strategies and differentiate between new and existing customer segments.

- **Static Contact Limit**  
  We optimize for a fixed 100‐contact limit with fixed costs which may not reflect real world budget flexibility. 