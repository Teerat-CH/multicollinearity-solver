# Multicollinearity-Solver

Multicollinearity is the presence of high correlations among predictor variables.

**Multicollinearity-Solver** helps systematically reduce redundancy in feature spaces and improves interpretability of the machine learning model. 

The method operates as follows:  

1. Construct an **undirected graph** where each node corresponds to a feature and edges are drawn between pairs of features whose absolute correlation exceeds a user-specified threshold.  
2. Decompose the graph into **connected components**, representing clusters of mutually correlated variables.  
3. From each component, select a subset of features according to a criterion:  
   - **Feature importance** derived from a supervised model (e.g., Random Forest, Gradient Boosting Tree).  
   - **Variance**, to retain features with greater informational content.

<div style="display: flex; justify-content: center; align-items: center; gap: 20px;">
  <img src="https://github.com/user-attachments/assets/f359feae-fa92-417e-bf80-a52d7e01d763" style="width: 55%;"/>
  <span style="font-size: 2rem;">→</span>
  <img src="https://github.com/user-attachments/assets/42a29eb2-aad1-4fb8-ac7e-42ebc09546b8" style="width: 35%;"/>
</div>

<div style="display: flex; justify-content: center;">
  <img src="https://github.com/user-attachments/assets/0733cb49-b9be-4c29-9fee-cefa779cd19f" style="width: 50%; padding-bottom: 20px;">
</div>
---

## Usage
```bash
git clone https://github.com/Teerat-CH/multicollinearity-solver.git
cd multicollinearity-solver
pip install -r requirements.txt
```

Let we have a feature dataframe where `A, B, C` and `D` are features as follow

| A | B | C | D |
|---|---|---|---|
| 1 | 1 | 4 | 1 |
| 2 | 2 | 5 | 1 |
| 3 | 1 | 6 | 1 |
| 4 | 2 | 8 | 1 |

Let we have the feature importance of the model as 

| Feature | Importance |
|---------|------------|
| A       | 0.5        |
| B       | 0.3        |
| C       | 0.7        |
| D       | 0.2        |

We can construct such feature importance dataframe from model like LightGBM by

```python
feature_importance = pd.DataFrame({
    "Feature": model.feature_name_,
    "Importance": model.feature_importances_
}).set_index('Feature')
```

Then we can get a new set of feature by

```python
from solver import mcl_solver

# Run solver -> list of features to be removed
features_to_remove = mcl_solver(
    data,
    feature_importance=feature_importance,
    by="importance",
    threshold=0.8
)

# Keep only features not flagged for removal
new_features = [feature for feature in data.columns if feature not in features_to_remove]
data = data[new_features]
```
