#import necessary libraries
import numpy as np
import pandas as pd
from sklearn.model_selection import train_test_split

#dataset:
column_names = [
    "CRIM",    # per capita crime rate by town
    "ZN",      # proportion of residential land zoned for lots over 25,000 sq.ft.
    "INDUS",   # proportion of non-retail business acres per town
    "CHAS",    # Charles River dummy variable (= 1 if tract bounds river; 0 otherwise)
    "NOX",     # nitric oxides concentration (parts per 10 million)
    "RM",      # average number of rooms per dwelling
    "AGE",     # proportion of owner-occupied units built before 1940
    "DIS",     # weighted distances to five Boston employment centers
    "RAD",     # index of accessibility to radial highways
    "TAX",     # full-value property-tax rate per $10,000
    "PTRATIO", # pupil-teacher ratio by town
    "B",       # 1000(Bk - 0.63)^2 where Bk is proportion of Black population
    "LSTAT",   # % lower status of the population
    "MEDV"     # Median value of owner-occupied homes in $1000s (target variable)
]
df = pd.read_csv(
    'https://drive.google.com/uc?id=1LEngxxFuJyhNp7Gf7ubE6ow14f7_42vT',
    header=None,
    names=column_names,
    sep=r'\s+'
)
print(df.head())
#prepare the feature and target variable 
from sklearn.preprocessing import StandardScaler
X_raw = df.iloc[:, :-1].values  # all columns except MEDV
y = df.iloc[:, -1].values.reshape(-1, 1)  # MEDV column

# Initialize scaler and fit-transform the features
scaler = StandardScaler()
X_scaled = scaler.fit_transform(X_raw)
X = np.hstack((np.ones((X_scaled.shape[0], 1)), X_scaled))
# Instead of printing the whole X, print its shape and a small sample
print("X shape:", X.shape)
print("First 5 rows of X:\n", X[:5])

# separate the dataset into training and testing sets
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.25, random_state=42)

#Set the training algorithm
iterations=5000
learning_rate=0.005
# initial weight vector
w= np.zeros((X_train.shape[1], 1))
#Training algorithm
for i in range(iterations):
    y_pred = X_train @ w
    error = y_pred - y_train  # use y_train here
    gradient = (1 / X_train.shape[0]) * (X_train).T @ error
    w = w - learning_rate * gradient
#print w
print("Final weights (first 5):", w.flatten()[:5])

#test the model
y_test_pred=X_test@w
error_test= y_test_pred-y_test
#MSE:
mse_test = np.mean(error_test**2)
print("Mean Squared Error on Test Set:", mse_test)
print("Before R-squared calculation")
# R-squared score
from sklearn.metrics import r2_score
r2_test = r2_score(y_test, y_test_pred)
print("After R-squared calculation")
print("R-squared on Test Set:", r2_test)

#R^2 score interpretation:
print("R-squared interpretation:")
if r2_test>0.8:
    print("The model is very strong :)!!")
elif r2_test>0.6:
    print("The model is strong :)")
elif r2_test>0.4:
    print("The model is moderate :|")
elif r2_test>0.2:
    print("The model is weak :(")
else:
    print("The model is very weak :(")