import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LinearRegression
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score
import matplotlib.pyplot as plt
import numpy as np

# 1. Import and preprocess the dataset
# Load the dataset (replace 'house_price.csv' with the actual path if needed)
try:
    df = pd.read_csv('house_price.csv')
except FileNotFoundError:
    print("Error: 'house_price.csv' not found.  Make sure the file is in the same directory or provide the correct path.")
    exit()  # Stop execution if the file isn't found

# Display the first few rows to understand the data
print("First 5 rows of the dataset:")
print(df.head())

# Get information about the columns and their data types
print("\nDataset information:")
print(df.info())

# Check for missing values
print("\nMissing values per column:")
print(df.isnull().sum())

# Handle missing values (impute with the mean for numerical columns)
for col in df.select_dtypes(include=np.number).columns:
    df[col] = df[col].fillna(df[col].mean())

# Verify that there are no more missing values
print("\nMissing values after handling:")
print(df.isnull().sum())

# 2. Split data into train-test sets
# Select features and target
#  We'll start with simple linear regression using 'GrLivArea' (Above ground living area)
X = df[['GrLivArea']]
y = df['SalePrice']

# Split the data into training and testing sets (80% train, 20% test)
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

print("\nShape of training data (X_train, y_train):", X_train.shape, y_train.shape)
print("Shape of testing data (X_test, y_test):", X_test.shape, y_test.shape)


# 3. Fit a Linear Regression model
# Create a Linear Regression model object
model = LinearRegression()

# Train the model using the training data
model.fit(X_train, y_train)

print("\nLinear Regression model trained.")

# 4. Evaluate the model
# Make predictions on the test set
y_pred = model.predict(X_test)

# Calculate evaluation metrics
mae = mean_absolute_error(y_test, y_pred)
mse = mean_squared_error(y_test, y_pred)
r2 = r2_score(y_test, y_pred)

print("\nModel Evaluation:")
print(f"Mean Absolute Error (MAE): {mae:.2f}")
print(f"Mean Squared Error (MSE): {mse:.2f}")
print(f"R-squared (R²): {r2:.2f}")

# 5. Plot regression line and interpret coefficients (for simple linear regression)
plt.figure(figsize=(10, 6))  # Make the plot a bit bigger
plt.scatter(X_test, y_test, color='blue', label='Actual Data')
plt.plot(X_test, y_pred, color='red', linewidth=2, label='Regression Line')
plt.xlabel('Above Ground Living Area (GrLivArea)')
plt.ylabel('Sale Price')
plt.title('Simple Linear Regression: GrLivArea vs. SalePrice')
plt.legend()
plt.grid(True)  # Add a grid for better readability
plt.show()

# Interpret the coefficients (for simple linear regression)
print("\nInterpretation of Simple Linear Regression:")
print(f"Coefficient (Slope): {model.coef_[0]:.2f}")
print(f"Intercept: {model.intercept_:.2f}")
print(f"For every 1 square foot increase in above ground living area, the predicted sale price increases by ${model.coef_[0]:.2f}.")
print(f"The intercept (${model.intercept_:.2f}) is the predicted sale price when the living area is 0.  This might not have a realistic interpretation.")


# 6. Multiple Linear Regression (Example)
print("\n\n----------------------------------------------------------")
print("Multiple Linear Regression Example (using GrLivArea and LotArea):")
# Select multiple features
X = df[['GrLivArea', 'LotArea']]
y = df['SalePrice']

# Split the data
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# Create and train the model
model = LinearRegression()
model.fit(X_train, y_train)

# Make predictions
y_pred = model.predict(X_test)

# Evaluate the model
mae = mean_absolute_error(y_test, y_pred)
mse = mean_squared_error(y_test, y_pred)
r2 = r2_score(y_test, y_pred)
print(f"Multiple Regression - MAE: {mae:.2f}, MSE: {mse:.2f}, R2: {r2:.2f}")

# Interpret coefficients
print("\nInterpretation of Multiple Linear Regression:")
print(f"Coefficient for GrLivArea: {model.coef_[0]:.2f}")
print(f"Coefficient for LotArea: {model.coef_[1]:.2f}")
print(f"Intercept: {model.intercept_:.2f}")

print("\nInterpretation (Multiple Regression):")
print(
    "Holding LotArea constant, for every 1 square foot increase in GrLivArea, the"
    f" predicted sale price increases by ${model.coef_[0]:.2f}."
)
print(
    "Holding GrLivArea constant, for every 1 square foot increase in LotArea, the"
    f" predicted sale price increases by ${model.coef_[1]:.2f}."
)
print(f"The intercept is ${model.intercept_:.2f}.")
