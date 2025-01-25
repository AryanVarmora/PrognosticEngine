import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.metrics import mean_squared_error, mean_absolute_error, r2_score
import xgboost as xgb
import joblib

# 1. Load the dataset
print("Loading dataset...")
data_path = "/Users/aryan/Desktop/PrognosticEngine/data/train_FD001_features_enhanced.csv"  # Update path if needed
data = pd.read_csv(data_path)

# 2. Prepare the data
print("Preparing features and target...")
X = data.drop(columns=["RUL", "engine_id", "cycle"])  # Features
y = data["RUL"]  # Target variable

# 3. Split the data into training and validation sets
print("Splitting data into train and validation sets...")
X_train, X_val, y_train, y_val = train_test_split(X, y, test_size=0.2, random_state=42)

# 4. Define the XGBoost model
print("Defining XGBoost model...")
xgb_model = xgb.XGBRegressor(
    n_estimators=200,  # Number of trees
    max_depth=10,      # Maximum tree depth
    learning_rate=0.1, # Learning rate
    random_state=42,   # For reproducibility
    verbosity=1
)

# 5. Train the model
print("Training the XGBoost model...")
xgb_model.fit(X_train, y_train)

# 6. Save the model
print("Saving the trained model...")
model_path = "/Users/aryan/Desktop/PrognosticEngine/src/models/XGBoost_model.pkl"  # Update path if needed
joblib.dump(xgb_model, model_path)
print(f"Model saved to {model_path}")

# 7. Evaluate the model
print("Evaluating the model on validation data...")
val_predictions = xgb_model.predict(X_val)
rmse = mean_squared_error(y_val, val_predictions, squared=False)
mae = mean_absolute_error(y_val, val_predictions)
r2 = r2_score(y_val, val_predictions)

print(f"Validation RMSE: {rmse:.4f}")
print(f"Validation MAE: {mae:.4f}")
print(f"Validation R^2 Score: {r2:.4f}")
