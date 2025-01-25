import pandas as pd
import joblib
import shap
import matplotlib.pyplot as plt

# Load the dataset
print("Loading dataset for explainability...")
data_path = "/Users/aryan/Desktop/PrognosticEngine/data/train_FD001_features_enhanced.csv"
data = pd.read_csv(data_path)

# Prepare the features and target
X = data.drop(columns=["RUL", "engine_id", "cycle"])
y = data["RUL"]

# Load the trained Linear Regression model
print("Loading Linear Regression model...")
model_path = "/Users/aryan/Desktop/PrognosticEngine/src/models/LinearRegression_model.pkl"
linear_model = joblib.load(model_path)

# Use SHAP for explainability
print("Running SHAP explainability...")
explainer = shap.Explainer(linear_model, X)
shap_values = explainer(X)

# Summary plot
print("Generating SHAP summary plot...")
shap.summary_plot(shap_values, X)

# Feature importance bar plot
print("Generating SHAP feature importance plot...")
shap.summary_plot(shap_values, X, plot_type="bar")

# Save the plots (optional)
plt.savefig("/Users/aryan/Desktop/PrognosticEngine/visuals/shap_summary_plot.png")
print("SHAP summary plot saved.")
