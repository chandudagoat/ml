import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LinearRegression

df = pd.read_csv("Salary_Data.csv")

# feature df
X = df.iloc[:, :-1].values
# dependant var df
y = df.iloc[:, -1].values

X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.2, train_size=0.8, random_state=0
)

regression = LinearRegression()
regression.fit(X_train, y_train)

y_pred = regression.predict(X_test)


def training_set_plot():
    plt.scatter(X_train, y_train, color="red")
    plt.plot(X_train, regression.predict(X_train), color="blue")
    plt.title("salary - experience (training set)")
    plt.xlabel("Exp.")
    plt.ylabel("Salary")


def test_set_plot():
    plt.scatter(X_test, y_test, color="red")
    plt.plot(X_test, y_pred)
    plt.title("salary - exp (test)")
    plt.xlabel("Exp.")
    plt.ylabel("Salary")


test_set_plot()
plt.show()
