import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns

# Load cleaned dataset
file_path = "dataset/cleaned_car_data.csv"  # Update this with your correct path
df = pd.read_csv(file_path)

# Set plot style
plt.style.use("ggplot")

# 1️⃣ Distribution of Selling Price
plt.figure(figsize=(8, 5))
sns.histplot(df['selling_price'], bins=50, kde=True)
plt.title("Distribution of Car Selling Prices")
plt.xlabel("Selling Price")
plt.ylabel("Frequency")
plt.show()

# 2️⃣ Correlation Heatmap
plt.figure(figsize=(12, 8))
sns.heatmap(df.corr(), annot=True, cmap="coolwarm", fmt=".2f", linewidths=0.5)
plt.title("Correlation Heatmap")
plt.show()

# 3️⃣ Scatter plot: Year vs. Selling Price
plt.figure(figsize=(8, 5))
sns.scatterplot(x=df['year'], y=df['selling_price'])
plt.title("Car Year vs. Selling Price")
plt.xlabel("Year")
plt.ylabel("Selling Price")
plt.show()

# 4️⃣ Boxplot: Fuel Type vs. Selling Price
plt.figure(figsize=(8, 5))
sns.boxplot(x="fuel_Diesel", y="selling_price", data=df)
plt.title("Impact of Fuel Type on Selling Price (Diesel vs Non-Diesel)")
plt.xlabel("Diesel (1 = Yes, 0 = No)")
plt.ylabel("Selling Price")
plt.show()

print("✅ Data analysis completed. Check the generated visualizations.")
