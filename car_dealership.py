import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from sklearn.linear_model import LinearRegression
from sklearn.metrics import mean_squared_error

# Load the data
data = pd.read_csv('/Users/sahibkazimli/Programmering - Projekt och Uppgifter/Car_data.csv')

# Display the first few rows of the data ghghghghgh
print(data.head())

# Calculate the correlation coefficient
correlation = data['Price'].corr(data['Mileage'])
print(f"Correlation Coefficient: {correlation}")

###################################################
# Try to compute the correlation here by yourself #
# Check the slides for the formula                #
# #### TODO ####                                  #
###################################################

mean_mileage = np.mean(data['Mileage'].values)
mean_price = np.mean(data['Price'].values)
numerator = np.sum((data['Mileage'].values - mean_mileage)*(data['Price'].values - mean_price))
sum_mileage_squared = np.sum((data['Mileage'].values - mean_mileage)**2)
sum_price_squared = np.sum((data['Price'].values - mean_price)**2)
denominator = np.sqrt(np.sum((data['Mileage'].values - mean_mileage) ** 2)) * np.sqrt(np.sum((data['Price'].values - mean_price) ** 2))



correlation_from_scratch = (numerator/denominator)

print(f"Correlation Coefficient (from scratch): {correlation_from_scratch} Difference: {correlation - correlation_from_scratch}")

# Now that we know there is a correlation between Price and Mileage, let's build a regression model

# Build a simple linear regression model
X = data['Mileage'].values.reshape(-1, 1)
y = data['Price'].values

###################################
# Build the regression model here #
# #### TODO ####                  #
###################################

# We can do it from scratch



# Define the Linear Regression class
class LinearRegressionFromScratch:
    def __init__(self):
        self.slope = None
        self.intercept = None

    def fit(self, X, y):
        X = X.flatten()  # Convert from (n,1) shape to (n,)
        
        # Step 1: Calculate means
        mean_X = np.mean(X)
        mean_y = np.mean(y)

        # Step 2: Compute slope (B1)
        numerator = np.sum((X - mean_X) * (y - mean_y))
        denominator = np.sum((X - mean_X) ** 2)
        self.slope = numerator / denominator

        # Step 3: Compute intercept (B0)
        self.intercept = mean_y - self.slope * mean_X

    def predict(self, X):
        return self.slope * X + self.intercept

# Train the model
model = LinearRegressionFromScratch()
model.fit(X, y)
predicted_prices = model.predict(X)

# Calculate the Mean Squared Error (MSE)
mse = mean_squared_error(y, predicted_prices)
print(f"Mean Squared Error: {mse}")

# Plot the data and the regression line
plt.scatter(X, y, color='blue', label='Actual Prices')
plt.plot(X, predicted_prices, color='red', label='Predicted Prices')
plt.xlabel("Mileage")
plt.ylabel("Price")
plt.title("Car Price Prediction using Linear Regression (scikit-learn)")
plt.legend()
plt.savefig('car_price_prediction.png')
plt.show()


"""Since the correlation coefficient is negative, near -1, price and mileage are negatively correlated.
This makes sense since the value of a car should decrease with increased mileage. However, my mean squared error is quite massive,
around 4044839, which make the predictions quite inaccurate and indicates a weak model. I would in short, not recommend this model."""