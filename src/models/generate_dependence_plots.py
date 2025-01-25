import os
import pandas as pd
import shap
import joblib
import matplotlib.pyplot as plt

# 1. Load the dataset
print("Loading dataset and trained model...")
data_path = "/Users/aryan/Desktop/PrognosticEngine/data/train_FD001_features_enhanced.csv"
data = pd.read_csv(data_path)

# Prepare the features (X)
X = data.drop(columns=["RUL", "engine_id", "cycle"])  # Adjust columns as needed

# 2. Load the trained model
model_path = "/Users/aryan/Desktop/PrognosticEngine/src/models/LinearRegression_model.pkl"  # Update with your model file path
model = joblib.load(model_path)
print("Model loaded successfully!")

# 3. Generate SHAP values
print("Generating SHAP values...")
explainer = shap.Explainer(model, X)
shap_values = explainer(X)  # New-style Explanation object

# Convert SHAP values to a NumPy array
shap_values_array = shap_values.values  # Extract the SHAP values as a NumPy array

# 4. Define features for dependence plots
features_to_plot = [
    "sensor_measurement_13_roll3",
    "sensor_measurement_8_roll3",
    "sensor_measurement_18_lag2",
    "sensor_measurement_8_lag"
]

# Ensure output directory exists
output_dir = "/Users/aryan/Desktop/PrognosticEngine/outputs/dependence_plots"
os.makedirs(output_dir, exist_ok=True)

# 5. Generate and save dependence plots

print("Creating dependence plots...")
for feature_name in features_to_plot:
    if feature_name not in X.columns:
        print(f"Warning: Feature '{feature_name}' not found in the dataset. Skipping...")
        continue
    print(f"Generating dependence plot for: {feature_name}")
    shap.dependence_plot(feature_name, shap_values_array, X, show=False)
    plt.title(f"Dependence Plot for {feature_name}")
    plot_path = os.path.join(output_dir, f"{feature_name}_dependence_plot.png")
    plt.savefig(plot_path)
    plt.close()
    print(f"Plot saved to: {plot_path}")


# 6. Generate summary report
summary_path = os.path.join(output_dir, "summary_report.txt")
print("Generating summary report...")
with open(summary_path, "w") as report:
    report.write("### SHAP Dependence Plot Analysis ###\n\n")
    report.write("Generated Dependence Plots for the following features:\n")
    for feature_name in features_to_plot:
        plot_path = os.path.join(output_dir, f"{feature_name}_dependence_plot.png")
        report.write(f"- Feature: {feature_name}\n")
        report.write(f"  Plot Path: {plot_path}\n")
        report.write("  Insights:\n")
        report.write("    - Replace this with your observations for this feature.\n")
        report.write("\n")
    report.write("Analysis complete. Add your insights above.\n")

print(f"Summary report saved to: {summary_path}")
print("All dependence plots generated and saved successfully!")
