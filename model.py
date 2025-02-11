import pandas as pd
import pickle
import joblib
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LinearRegression
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score

# Load cleaned dataset
file_path = "dataset/cleaned_car_data.csv"  # Update with your correct path
df = pd.read_csv(file_path)

# Define features (X) and target variable (y)
X = df.drop(columns=['selling_price'])
y = df['selling_price']

# Split data into training (80%) and testing (20%) sets
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# Initialize and train the Linear Regression model
model = LinearRegression()
model.fit(X_train, y_train)

# Make predictions
y_pred = model.predict(X_test)

# Model Evaluation
mae = mean_absolute_error(y_test, y_pred)
mse = mean_squared_error(y_test, y_pred)
r2 = r2_score(y_test, y_pred)

# Print evaluation metrics
print(f"✅ Model Training Completed!")
print(f"🔹 Mean Absolute Error (MAE): {mae:.2f}")
print(f"🔹 Mean Squared Error (MSE): {mse:.2f}")
print(f"🔹 R² Score: {r2:.4f}")

# Save the trained model
with open("linear_regression_model.pkl", "wb") as f:
    pickle.dump(model, f)

print("✅ Model saved as 'models/linear_regression_model.pkl'.")