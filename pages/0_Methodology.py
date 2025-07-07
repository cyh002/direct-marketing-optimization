"""Methodology explanation page."""

from __future__ import annotations

import streamlit as st
import plotly.graph_objects as go

st.set_page_config(layout="wide")

st.title("📝 Methodology")

st.markdown(
    """
This page explains the methodology used to develop the optimized targeting strategy for direct marketing campaigns.
A hybrid approach is used to combine machine learning models with mathematical optimization to maximize expected revenue
while respecting business constraints.
"""
)

# Create tabs for different sections of the methodology
tab1, tab2, tab3, tab4, tab5 = st.tabs(
    [
        "Overview",
        "Modeling Approach",
        "Optimization Formulation",
        "Evaluation Metrics",
        "Implementation Details",
    ]
)

with tab1:
    st.header("Hybrid Approach Overview")

    st.markdown(
        """
    ### The Direct Marketing Problem
    
    The company faces a common business problem: **how to allocate limited marketing resources to maximize return on investment**.
    With a large customer base but limited capacity to make contact offers, we need to identify:
    
    1. Which customers are most likely to accept each product offer
    2. How much revenue each customer is likely to generate if they accept
    3. The optimal allocation of marketing efforts across customers and products
    
    ### A Hybrid Approach

    Three-stage hybrid approach is proposed that combines:

    1. **Propensity Modeling**: Machine learning models that predict the probability of a customer purchasing each product
    2. **Revenue Modeling**: Regression models that predict the expected revenue from each customer if they purchase a product
    3. **Mathematical Optimization**: A constrained optimization problem that finds the best customer-product assignments
    """
    )

    # Create a visual diagram of the process flow
    st.subheader("Methodology Flow")

    col1, col2, col3 = st.columns([1, 2, 1])

    with col2:
        fig = go.Figure()

        # Add steps as shapes
        shapes = [
            # Data preprocessing box
            dict(
                type="rect",
                xref="paper",
                yref="paper",
                x0=0.05,
                y0=0.85,
                x1=0.95,
                y1=1.0,
                line=dict(color="RoyalBlue", width=2),
                fillcolor="lightblue",
                opacity=0.5,
            ),
            # Propensity model box
            dict(
                type="rect",
                xref="paper",
                yref="paper",
                x0=0.05,
                y0=0.65,
                x1=0.45,
                y1=0.8,
                line=dict(color="green", width=2),
                fillcolor="lightgreen",
                opacity=0.5,
            ),
            # Revenue model box
            dict(
                type="rect",
                xref="paper",
                yref="paper",
                x0=0.55,
                y0=0.65,
                x1=0.95,
                y1=0.8,
                line=dict(color="orange", width=2),
                fillcolor="lightsalmon",
                opacity=0.5,
            ),
            # Expected value box
            dict(
                type="rect",
                xref="paper",
                yref="paper",
                x0=0.05,
                y0=0.45,
                x1=0.95,
                y1=0.6,
                line=dict(color="purple", width=2),
                fillcolor="lavender",
                opacity=0.5,
            ),
            # Optimization box
            dict(
                type="rect",
                xref="paper",
                yref="paper",
                x0=0.05,
                y0=0.25,
                x1=0.95,
                y1=0.4,
                line=dict(color="crimson", width=2),
                fillcolor="mistyrose",
                opacity=0.5,
            ),
            # Optimized list box
            dict(
                type="rect",
                xref="paper",
                yref="paper",
                x0=0.05,
                y0=0.05,
                x1=0.95,
                y1=0.2,
                line=dict(color="goldenrod", width=2),
                fillcolor="lightyellow",
                opacity=0.5,
            ),
        ]

        # Add arrows connecting boxes
        shapes.extend(
            [
                # Arrow from preprocessing to propensity
                dict(
                    type="line",
                    xref="paper",
                    yref="paper",
                    x0=0.25,
                    y0=0.85,
                    x1=0.25,
                    y1=0.8,
                    line=dict(color="gray", width=2, dash="solid"),
                    layer="below",
                ),
                # Arrow from preprocessing to revenue
                dict(
                    type="line",
                    xref="paper",
                    yref="paper",
                    x0=0.75,
                    y0=0.85,
                    x1=0.75,
                    y1=0.8,
                    line=dict(color="gray", width=2, dash="solid"),
                    layer="below",
                ),
                # Arrow from propensity to expected value
                dict(
                    type="line",
                    xref="paper",
                    yref="paper",
                    x0=0.25,
                    y0=0.65,
                    x1=0.25,
                    y1=0.6,
                    line=dict(color="gray", width=2, dash="solid"),
                    layer="below",
                ),
                # Arrow from revenue to expected value
                dict(
                    type="line",
                    xref="paper",
                    yref="paper",
                    x0=0.75,
                    y0=0.65,
                    x1=0.75,
                    y1=0.6,
                    line=dict(color="gray", width=2, dash="solid"),
                    layer="below",
                ),
                # Arrow from expected value to optimization
                dict(
                    type="line",
                    xref="paper",
                    yref="paper",
                    x0=0.5,
                    y0=0.45,
                    x1=0.5,
                    y1=0.4,
                    line=dict(color="gray", width=2, dash="solid"),
                    layer="below",
                ),
                # Arrow from optimization to optimized list
                dict(
                    type="line",
                    xref="paper",
                    yref="paper",
                    x0=0.5,
                    y0=0.25,
                    x1=0.5,
                    y1=0.2,
                    line=dict(color="gray", width=2, dash="solid"),
                    layer="below",
                ),
            ]
        )

        # Add the shapes to the figure
        fig.update_layout(
            shapes=shapes,
            height=600,
            width=600,
            plot_bgcolor="white",
            margin=dict(l=20, r=20, t=20, b=20),
            showlegend=False,
        )

        # Add text annotations
        fig.add_annotation(
            x=0.5,
            y=0.925,
            xref="paper",
            yref="paper",
            text="1. Data Preprocessing & Feature Engineering",
            showarrow=False,
            font=dict(size=12, color="darkblue"),
        )
        fig.add_annotation(
            x=0.25,
            y=0.725,
            xref="paper",
            yref="paper",
            text="2a. Propensity Models",
            showarrow=False,
            font=dict(size=12, color="darkgreen"),
        )
        fig.add_annotation(
            x=0.75,
            y=0.725,
            xref="paper",
            yref="paper",
            text="2b. Revenue Models",
            showarrow=False,
            font=dict(size=12, color="darkorange"),
        )
        fig.add_annotation(
            x=0.5,
            y=0.525,
            xref="paper",
            yref="paper",
            text="3. Calculate Expected Values",
            showarrow=False,
            font=dict(size=12, color="purple"),
        )
        fig.add_annotation(
            x=0.5,
            y=0.325,
            xref="paper",
            yref="paper",
            text="4. Constrained Optimization",
            showarrow=False,
            font=dict(size=12, color="crimson"),
        )
        fig.add_annotation(
            x=0.5,
            y=0.125,
            xref="paper",
            yref="paper",
            text="5. Optimized Client-Product Assignments",
            showarrow=False,
            font=dict(size=12, color="goldenrod"),
        )

        st.plotly_chart(fig)

with tab2:
    st.header("Modeling Approach")

    st.markdown(
        """
    ### 1. Data Preprocessing
    
    Before modeling, we perform several preprocessing steps:
    - Merging multiple data sources (social-demographic, financial transactions, product holdings, sales and revenue data)
    - Handling missing values through imputation
    - Encoding categorical variables
    - Feature scaling for certain algorithms
    - Train/test splitting for model validation
    - Training with CV folds to ensure robustness
    
    ### 2a. Propensity Modeling
    
    For each product (MF, CC, CL), we train a separate model to predict the probability of purchase:
    
    $$P(Sale_{i,j} = 1 | X_i)$$
    
    Where:
    - $i$ represents a customer
    - $j$ represents a product
    - $X_i$ is the feature vector for customer $i$
    
    **Model Selection**: We evaluate several classification algorithms, including:
    - Logistic Regression
    - Gradient Boosting
    - Random Forest
    
    **Default Evaluation Metric**: F1 score
    
    $$F1 = 2 \\cdot \\frac{precision \\cdot recall}{precision + recall}$$
    
    The F1 score was chosen because:
    - It balances precision and recall
    - Data is highly imbalanced, with many non-purchases. It handles class imbalance better than accuracy
    - It's appropriate for binary classification problems
    
    
    ### 2b. Revenue Modeling
    
    For customers who purchase a product, we need to predict how much revenue they will generate:
    
    $$Revenue_{i,j} | (Sale_{i,j} = 1)$$
    
    **Model Selection**: We evaluate several regression algorithms, including:
    - Linear Regression
    - Gradient Boosting Regressor
    - Random Forest Regressor
    
    **Default Evaluation Metric**: Negative Mean Squared Error (Negative MSE)
    
    $$\\text{Negative MSE} = -\\frac{1}{n} \\sum_{i=1}^{n} (y_i - \\hat{y}_i)^2$$
    
    Negative MSE is used because:
    - Scikit-learn's optimization requires metrics where higher is better
    - It allows for direct comparison between models
    - The negation transforms the traditional MSE (where lower is better) into a metric where higher is better
    
    ### 3. Expected Value Calculation
    
    The expected revenue for each customer-product pair is calculated as:
    
    $$E[Revenue_{i,j}] = P(Sale_{i,j} = 1 | X_i) \\cdot E[Revenue_{i,j} | Sale_{i,j} = 1]$$
    
    This expected value serves as the objective function coefficient in our optimization problem.
    """
    )

with tab3:
    st.header("Optimization Formulation")

    st.markdown(
        """
    ### Mathematical Formulation
    
    The optimization problem is formulated as a binary integer programming problem:
    
    #### Decision Variables:
    
    $$x_{i,j} = 
    \\begin{cases} 
    1 & \\text{if customer } i \\text{ receives offer for product } j \\\\
    0 & \\text{otherwise}
    \\end{cases}
    $$
    
    #### Objective Function:
    
    Maximize the total expected revenue:
    
    $$\\max \\sum_{i=1}^{n} \\sum_{j=1}^{m} E[Revenue_{i,j}] \\cdot x_{i,j}$$
    
    Where:
    - $n$ is the number of customers
    - $m$ is the number of products
    - $E[Revenue_{i,j}]$ is the expected revenue for customer $i$ and product $j$
    
    #### Constraints:
    
    1. **Contact Limit**: The total number of customers contacted must not exceed the specified limit:
    
    $$\\sum_{i=1}^{n} \\sum_{j=1}^{m} x_{i,j} \\leq \\text{contact\\_limit}$$
    
    2. **One Offer Per Customer**: Each customer receives at most one offer:
    
    $$\\sum_{j=1}^{m} x_{i,j} \\leq 1 \\quad \\forall i \\in \\{1, 2, \\ldots, n\\}$$
    
    3. **Product Distribution** (Optional): For each product, there may be a maximum fraction of offers:
    
    $$\\sum_{i=1}^{n} x_{i,j} \\leq \\text{max\\_fraction}_j \\cdot \\text{contact\\_limit} \\quad \\forall j \\in \\{1, 2, \\ldots, m\\}$$
    
    4. **Minimum Expected Revenue** (Optional): The total expected revenue must exceed a threshold:
    
    $$\\sum_{i=1}^{n} \\sum_{j=1}^{m} E[Revenue_{i,j}] \\cdot x_{i,j} \\geq \\text{min\\_expected\\_revenue}$$
    
    5. **Binary Constraint**: Each decision variable is binary:
    
    $$x_{i,j} \\in \\{0, 1\\} \\quad \\forall i,j$$
    
    ### Implementation
    
    This optimization problem is solved using the CVXPY library with the ECOS_BB solver, which is suitable for mixed integer programming problems.
    
    ```python
    # Simplified pseudocode for the optimization
    
    # Decision variables
    x = cp.Variable((n_customers, n_products), boolean=True)
    
    # Objective: maximize expected revenue
    objective = cp.Maximize(cp.sum(cp.multiply(expected_revenues, x)))
    
    # Constraints
    constraints = [
        # Contact limit constraint
        cp.sum(x) <= contact_limit,
        
        # One offer per customer constraint
        cp.sum(x, axis=1) <= 1
    ]
    
    # Optional: product distribution constraint
    if max_fraction_per_product is not None:
        for j in range(n_products):
            max_count = max_fraction_per_product.get(j, 1.0) * contact_limit
            constraints.append(cp.sum(x[:, j]) <= max_count)
    
    # Optional: minimum expected revenue constraint
    if min_expected_revenue > 0:
        constraints.append(
            cp.sum(cp.multiply(expected_revenues, x)) >= min_expected_revenue
        )
    
    # Solve the problem
    problem = cp.Problem(objective, constraints)
    problem.solve(solver=cp.ECOS_BB)
    ```
    """
    )

with tab4:
    st.header("Evaluation Metrics")

    st.markdown(
        """
    ### Model Evaluation Metrics
    
    #### Propensity Models
    
    - **F1 Score (Default)**: Harmonic mean of precision and recall
    - **AUC-ROC**: Area Under the Receiver Operating Characteristic curve
    - **Accuracy**: Proportion of correct predictions
    - **Precision**: Positive predictive value (TP / (TP + FP))
    - **Recall**: Sensitivity or true positive rate (TP / (TP + FN))
    
    #### Revenue Models
    
    - **Negative MSE (Default)**: Negated mean squared error
    - **R²**: Coefficient of determination
    - **MAE**: Mean absolute error
    
    ### Campaign Evaluation Metrics
    
    Once the optimization is complete, we evaluate the effectiveness of the campaign using:
    
    #### 1. Total Expected Revenue
    
    $$\\text{Total Revenue} = \\sum_{i=1}^{n} \\sum_{j=1}^{m} E[Revenue_{i,j}] \\cdot x_{i,j}$$
    
    #### 2. Revenue Per Contact
    
    $$\\text{Revenue Per Contact} = \\frac{\\text{Total Revenue}}{\\text{Number of Contacts}}$$
    
    #### 3. Expected Acceptance Rate
    
    $$\\text{Acceptance Rate} = \\frac{\\sum_{i=1}^{n} \\sum_{j=1}^{m} P(Sale_{i,j} = 1|X_i) \\cdot x_{i,j}}{\\text{Number of Contacts}}$$
    
    #### 4. Return on Investment (ROI)
    
    $$\\text{ROI} = \\frac{\\text{Total Revenue}}{\\text{Total Cost}} - 1$$
    
    Where the total cost is:
    
    $$\\text{Total Cost} = \\text{Number of Contacts} \\cdot \\text{Cost Per Contact}$$
    
    By default, we assume a cost per contact of €1.0, but this can be configured.
    """
    )

with tab5:
    st.header("Implementation Details")

    st.markdown(
        """
    ### Project Components
    
    The solution is implemented using a modular architecture with the following key components:
    
    1. **DataLoader**: Handles reading data from Excel files and initial data preparation
    2. **Preprocessor**: Applies transformations, feature engineering, and encoding
    3. **Trainers**:
       - PropensityTrainer: Fits and evaluates classification models
       - RevenueTrainer: Fits and evaluates regression models
    4. **Inference**: Generates predictions for new data using trained models
    5. **Optimizer**: Solves the constrained optimization problem
    6. **Evaluator**: Calculates performance metrics for the optimized solution
    
    ### Key Technologies Used
    
    - **Scikit-learn**: For building and evaluating machine learning models
    - **CVXPY**: For formulating and solving the optimization problem
    - **Hydra**: For configuration management
    - **MLflow**: For experiment tracking
    - **Streamlit**: For the interactive dashboard
    
    ### Configuration and Hyperparameter Tuning
    
    The project uses Hydra for configuration management, with the main settings in `conf/config.yaml`. 
    Key configurable parameters include:
    
    - Product list (`products`: CL, MF, CC)
    - Feature lists for preprocessing
    - Model types and hyperparameters
    - Optimization constraints (contact_limit, etc.)
    - Evaluation metrics
    
    Hyperparameter tuning is performed using Optuna, which efficiently searches the parameter space to find 
    optimal model configurations for each product.
    
    ### Reproducibility
    
    For reproducibility, all random operations use a fixed seed (default: 42), which can be configured in the 
    main configuration file. MLflow tracking ensures that all experiments are logged and can be compared.
    """
    )

st.markdown("---")

st.markdown(
    """
## Additional Notes

### Cost-Benefit Analysis

In practice, the cost of making contact offers may vary by product or channel. The optimization can be extended to include:

- Product-specific contact costs
- Different marketing channels with varying costs and effectiveness
- Time-dependent propensities and revenues

### Alternative Formulations

While a binary integer programming approach is used, other formulations are possible:

- Multi-objective optimization to balance revenue and customer satisfaction
- Stochastic optimization to account for uncertainty in predictions
- Multi-Armed Bandit (MAB) framework to dynamically learn and adapt the targeting strategy over time by balancing exploration (testing new offers) and exploitation (using the best-performing offers).

"""
)
