import pandas as pd

# Load dataset
file_path = "dataset/Car details v3.csv"  # Update with your dataset path
df = pd.read_csv(file_path)

# Remove units and convert columns to numerical
df['mileage'] = df['mileage'].str.extract('(\d+\.\d+)').astype(float)
df['engine'] = df['engine'].str.extract('(\d+)').astype(float)
df['max_power'] = df['max_power'].str.extract('(\d+\.\d+)').astype(float)

# Handle missing values
df['mileage'].fillna(df['mileage'].median(), inplace=True)
df['engine'].fillna(df['engine'].median(), inplace=True)
df['max_power'].fillna(df['max_power'].median(), inplace=True)
df['seats'].fillna(df['seats'].mode()[0], inplace=True)

# Convert categorical columns using One-Hot Encoding
df = pd.get_dummies(df, columns=['fuel', 'seller_type', 'transmission', 'owner'], drop_first=True)

# Drop irrelevant columns
df.drop(columns=['name', 'torque'], inplace=True)

# Save cleaned dataset
df.to_csv("dataset/cleaned_car_data.csv", index=False)

print("✅ Data preprocessing completed. Cleaned dataset saved as 'cleaned_car_data.csv'.")
