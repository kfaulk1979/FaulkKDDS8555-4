import pandas as pd
import numpy as np
import statsmodels.api as sm
from patsy import dmatrix
import matplotlib.pyplot as plt
from sklearn.metrics import r2_score, mean_squared_error


train = pd.read_csv("/Users/kevinfaulk/Documents/DDS-8555/train.csv")
test = pd.read_csv("/Users/kevinfaulk/Documents/DDS-8555/test.csv")

#Set up features and target
X = train.drop(columns=['id', 'Rings'])
y = train['Rings']

# Define test features
X_test = test.drop(columns=['id'])

# Choose Shell weight feature for step regression
X_feature = train['Shell weight'].values
X_test_feature = test['Shell weight'].values

# Create step function basis (e.g., bin width = 0.1)
step_train_basis = dmatrix("C(np.floor(X_feature / 0.1))", {"X_feature": X_feature}, return_type='dataframe')

# Fit OLS model
step_model = sm.OLS(y, step_train_basis).fit()
print(step_model.summary())

import statsmodels.api as sm
from scipy.stats import shapiro

# Residuals and fitted values
resid = step_model.resid
fitted = step_model.fittedvalues

# Residuals vs Fitted (Linearity & Homoscedasticity)
plt.figure(figsize=(6, 4))
plt.scatter(fitted, resid, alpha=0.5)
plt.axhline(0, color='black', linestyle='--')
plt.xlabel("Fitted values")
plt.ylabel("Residuals")
plt.title("Residuals vs. Fitted Values")
plt.grid(True)
plt.tight_layout()
plt.show()

# 2. Q-Q Plot (Normality of Errors)
sm.qqplot(resid, line='45')
plt.title("Normal Q-Q Plot of Residuals")
plt.tight_layout()
plt.show()

# 3. Shapiro-Wilk Test (Normality test)
stat, p = shapiro(resid)
print(f"Shapiro-Wilk Test: W = {stat:.4f}, p = {p:.4f}")
if p < 0.05:
    print("Residuals deviate significantly from normality.")
else:
    print("Residuals are approximately normal.")


# Create step function basis for test set
step_test_basis = dmatrix("C(np.floor(X_test_feature / 0.1))", {"X_test_feature": X_test_feature}, return_type='dataframe')

# Predict
y_test_pred = step_model.predict(step_test_basis)

# Print performance (only if true y_test is available)
if 'Rings' in test.columns:
    y_test_true = test['Rings']
    r2 = r2_score(y_test_true, y_test_pred)
    rmse = np.sqrt(mean_squared_error(y_test_true, y_test_pred))
    print(f"Test R²: {r2:.3f}")
    print(f"Test RMSE: {rmse:.3f}")

# Only for visual exploration
import matplotlib.pyplot as plt

# Sort for smooth plotting
sort_idx = np.argsort(X_feature)
X_sorted = X_feature[sort_idx]
pred_sorted = step_model.predict(step_train_basis)[sort_idx]

plt.figure(figsize=(10, 6))
plt.scatter(X_feature, y, alpha=0.5, label="Train Data", color="skyblue", edgecolor='gray')
plt.plot(X_sorted, pred_sorted, color='green', linewidth=2, label="Step Function Fit")
plt.xlabel("Shell weight")
plt.ylabel("Rings (Age)")
plt.title("Step Function Regression on Abalone (Train)")
plt.legend()
plt.grid(True)
plt.tight_layout()
plt.show()


# Round predictions if required by assignment or competition
rounded_preds = np.round(y_test_pred).astype(int)

# Create StepFunction DataFrame
StepFunction = pd.DataFrame({
    'id': test['id'],
    'Rings': rounded_preds
})

# Save to CSV
submission_path = "/Users/kevinfaulk/Documents/DDS-8555/abalone_submission_step.csv"
StepFunction.to_csv(submission_path, index=False)
print(f"Rounded submission file saved: {submission_path}")
