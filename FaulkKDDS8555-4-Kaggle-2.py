import pandas as pd
import numpy as np
import statsmodels.api as sm
from patsy import dmatrix
import matplotlib.pyplot as plt
from sklearn.metrics import r2_score, mean_squared_error
from scipy.stats import shapiro



train = pd.read_csv("/Users/kevinfaulk/Documents/DDS-8555/train.csv")
test = pd.read_csv("/Users/kevinfaulk/Documents/DDS-8555/test.csv")

# Define features and target variable
X_feature = train['Shell weight'].values  # predictor
y = train['Rings'].values                 # target
X_test_feature = test['Shell weight'].values  # test predictor

# Create Spline Basis for Train
spline_train_basis = dmatrix(
    "bs(X_feature, df=4, degree=3, include_intercept=True)",
    {"X_feature": X_feature},
    return_type='dataframe'
)

# Fit the model
spline_model = sm.OLS(y, spline_train_basis).fit()
print(spline_model.summary())

# Visualize the fit
y_spline_train_pred = spline_model.predict(spline_train_basis)

# Sort for smooth plotting
sort_idx = np.argsort(X_feature)
X_sorted = X_feature[sort_idx]
y_sorted_pred = y_spline_train_pred[sort_idx]

plt.figure(figsize=(10, 6))
plt.scatter(X_feature, y, alpha=0.5, label='Training Data', color='skyblue', edgecolor='gray')
plt.plot(X_sorted, y_sorted_pred, color='red', linewidth=2, label='Spline Fit (Train)')
plt.xlabel('Shell weight')
plt.ylabel('Rings (Age)')
plt.title('Spline Regression Fit to Training Data')
plt.legend()
plt.grid(True)
plt.tight_layout()
plt.show()

# Check Assumptions
resid = spline_model.resid
fitted = spline_model.fittedvalues

# Residuals vs Fitted
plt.figure(figsize=(6, 4))
plt.scatter(fitted, resid, alpha=0.5)
plt.axhline(0, linestyle='--', color='black')
plt.xlabel("Fitted values")
plt.ylabel("Residuals")
plt.title("Spline: Residuals vs Fitted")
plt.grid(True)
plt.tight_layout()
plt.show()

# Q-Q Plot
sm.qqplot(resid, line='45')
plt.title("Spline: Q-Q Plot of Residuals")
plt.tight_layout()
plt.show()

# Shapiro-Wilk Test for Normality
stat, p = shapiro(resid)
print(f"Shapiro-Wilk Test: W = {stat:.4f}, p = {p:.4f}")
if p < 0.05:
    print("Residuals deviate significantly from normality.")
else:
    print("Residuals are approximately normal.")

# Create Spline Basis for Test set
spline_test_basis = dmatrix(
    "bs(X_test_feature, df=4, degree=3, include_intercept=True)",
    {"X_test_feature": X_test_feature},
    return_type='dataframe'
)

#Predict on Test set
y_spline_pred = spline_model.predict(spline_test_basis)
rounded_spline_preds = np.round(y_spline_pred).astype(int)


# Save the predictions to a CSV file
Spline = pd.DataFrame({
    'id': test['id'],
    'Rings': rounded_spline_preds
})

submission_path = "/Users/kevinfaulk/Documents/DDS-8555/abalone_submission_spline.csv"
Spline.to_csv(submission_path, index=False)
print(f"Rounded spline submission file saved: {submission_path}")