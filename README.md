# Implementation-of-Linear-Regression-Using-Gradient-Descent

## AIM:
To write a program to predict the profit of a city using the linear regression model with gradient descent.

## Equipments Required:
1. Hardware – PCs
2. Anaconda – Python 3.7 Installation / Jupyter notebook

## Algorithm
1..Load the dataset from a CSV file and separate the features and target variable, encoding any categorical variables as needed.
2.Scale the features using a standard scaler to normalize the data.
3.Initialize model parameters (theta) and add an intercept term to the feature set.
4.Train the linear regression model using gradient descent by iterating through a specified number of iterations to minimize the cost function.
5.Make predictions on new data by transforming it using the same scaling and encoding applied to the training data. 
 

## Program:
```
/*
Program to implement the linear regression using gradient descent.
Developed by:  Yogalakshmi M 
RegisterNumber:  212224020061
*/
``
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from sklearn.preprocessing import StandardScaler

# Load dataset
data = pd.read_csv("exp_3_50_Startups.csv")

print("First 5 rows of the dataset:")
print(data.head())

# Encode categorical State column
data_encoded = pd.get_dummies(
    data, columns=["State"], drop_first=True
)

print("Encoded dataset:")
print(data_encoded.head())

# Separate features and target
X_raw = data_encoded.drop("Profit", axis=1).values
y_raw = data_encoded["Profit"].values.reshape(-1, 1)

print("Shape of features:", X_raw.shape)
print("Shape of target:", y_raw.shape)

# Scale features and target
X_scaler = StandardScaler()
y_scaler = StandardScaler()

X = X_scaler.fit_transform(X_raw)
y = y_scaler.fit_transform(y_raw)

# Add bias term
m = X.shape[0]
X = np.hstack((np.ones((m, 1)), X))

print("Shape of X after adding bias:", X.shape)

# Define cost function
def compute_cost(X, y, theta):
    m = len(y)
    predictions = X.dot(theta)
    cost = (1 / (2 * m)) * np.sum((predictions - y) ** 2)
    return cost

# Implement gradient descent
def gradient_descent(
    X, y, learning_rate=0.01,
    num_iters=2000, tol=1e-8, verbose=False
):
    m, n = X.shape
    theta = np.zeros((n, 1))
    J_history = []

    prev_cost = compute_cost(X, y, theta)

    for i in range(num_iters):
        predictions = X.dot(theta)
        errors = predictions - y
        gradient = (1 / m) * X.T.dot(errors)

        theta -= learning_rate * gradient

        cost = compute_cost(X, y, theta)
        J_history.append(cost)

        if abs(prev_cost - cost) < tol:
            if verbose:
                print("Converged at iteration", i)
            break

        prev_cost = cost

        if verbose and (i % 500 == 0 or i < 5):
            print("Iteration", i, "Cost:", round(cost, 6))

    return theta, J_history

# Train the model
alpha = 0.01

theta, J_hist = gradient_descent(
    X, y,
    learning_rate=alpha,
    num_iters=5000,
    tol=1e-9,
    verbose=True
)

print("Learned Parameters (Theta):")
print(theta.flatten())

# Plot cost vs iterations
plt.figure(figsize=(7, 4))
plt.plot(J_hist)
plt.xlabel("Iterations")
plt.ylabel("Cost (MSE/2)")
plt.title("Cost Function Convergence")
plt.grid(True)
plt.show()

# Predict profit for a new startup
new_sample = pd.DataFrame([{
    "R&D Spend": 165349.2,
    "Administration": 136897.8,
    "Marketing Spend": 471784.1,
    "State": "New York"
}])

# Encode the new sample
new_encoded = pd.get_dummies(
    new_sample, columns=["State"], drop_first=True
)

# Match training columns
new_encoded = new_encoded.reindex(
    columns=data_encoded.drop("Profit", axis=1).columns,
    fill_value=0
)

# Scale new data and add bias
new_scaled = X_scaler.transform(new_encoded)
new_design = np.hstack(
    (np.ones((new_scaled.shape[0], 1)), new_scaled)
)

# Predict and convert to original units
scaled_pred = new_design.dot(theta)
pred_original = y_scaler.inverse_transform(scaled_pred)

print("Predicted Profit for the new startup:",
      round(pred_original[0][0], 2))
```

## Output:

<img width="725" height="493" alt="image" src="https://github.com/user-attachments/assets/552f2c9b-9d3f-48fc-a7bf-ca33d2c7d89a" />



## Result:
Thus the program to implement the linear regression using gradient descent is written and verified using python programming.
