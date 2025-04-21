import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from patsy import dmatrix
from sklearn.linear_model import LinearRegression
from sklearn.preprocessing import SplineTransformer
from sklearn.pipeline import make_pipeline
from sklearn.metrics import r2_score
import statsmodels.api as sm
from patsy import dmatrix

# Load the Auto dataset from ISLP
from ISLP import load_data
Auto = load_data('Auto')

# Drop the NA values
Auto=Auto.dropna()

print(Auto.head())

print(Auto.dtypes)

#Extract predictor and response variables
X=Auto[['horsepower']].values
y=Auto['mpg'].values

X_flat = X.flatten()

# Sort for plotting
sort_idx = np.argsort(X_flat)
X_sorted = X[sort_idx]
y_sorted = y[sort_idx]


# Create step function basis: group horsepower into bins of width 40
step_basis = dmatrix("C(np.floor(horsepower / 40))", {"horsepower": X.flatten()}, return_type='dataframe')
# Fit OLS model
step_ols = sm.OLS(y, step_basis).fit()

# Print summary
print(step_ols.summary())


# Use cubic spline with 4 knots placed automatically at percentiles
spline_basis = dmatrix("bs(horsepower, df=4, degree=3, include_intercept=True)",
                          {"horsepower": X.flatten()}, return_type='dataframe')

# Fit OLS model
spline_ols = sm.OLS(y, spline_basis).fit()

# Print full regression results
print(spline_ols.summary())

# Predict using existing OLS model objects
step_preds = step_ols.predict(step_basis)
spline_preds = spline_ols.predict(spline_basis)

# Sort predictions
step_preds_sorted = step_preds[sort_idx]
spline_preds_sorted = spline_preds[sort_idx]

# Get R² directly from model objects
step_r2 = step_ols.rsquared
spline_r2 = spline_ols.rsquared

# Plot
plt.figure(figsize=(10, 6))
plt.scatter(X, y, edgecolor='gray', facecolor='lightblue', label='Observed Data', alpha=0.6)

# Step function plot
plt.plot(X_sorted, step_preds_sorted, label=f'Step Function (R² = {step_r2:.3f})', color='green', linewidth=2)

# Spline regression plot
plt.plot(X_sorted, spline_preds_sorted, label=f'Spline (R² = {spline_r2:.3f})', color='red', linewidth=2)

# Add labels and polish
plt.xlabel('Horsepower', fontsize=12)
plt.ylabel('MPG', fontsize=12)
plt.title('MPG vs. Horsepower: Step Function vs. Spline Regression', fontsize=14)
plt.legend()
plt.grid(True, linestyle='--', alpha=0.5)
plt.tight_layout()
plt.show()
