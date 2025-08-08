import pandas as pd
from sklearn.linear_model import LinearRegression

# Load dataset
df = pd.read_csv("dataset.csv")

# Train model
X = df[["sqft", "bedrooms", "bathrooms", "location_index"]]
y = df["price"]
model = LinearRegression()
model.fit(X, y)

print("🏠 House Price Prediction Model Ready!\n")

# Predict based on user input
try:
    sqft = float(input("Enter house size in sqft: "))
    bedrooms = int(input("Enter number of bedrooms: "))
    bathrooms = int(input("Enter number of bathrooms: "))
    location_index = int(input("Enter location index (1-10): "))

    predicted_price = model.predict([[sqft, bedrooms, bathrooms, location_index]])[0]
    print(f"💰 Predicted Price: ₹{predicted_price:,.2f}")

except Exception as e:
    print("❌ Error:", e)
