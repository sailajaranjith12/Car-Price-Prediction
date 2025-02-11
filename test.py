import joblib
import pandas as pd

# Load the trained model
model = joblib.load("linear_regression_model.pkl")

# Load the dataset to check feature names
df = pd.read_csv("dataset/cleaned_car_data.csv")
X = df.drop(columns=['selling_price'])

print("Features used for training:")
print(X.columns.tolist())