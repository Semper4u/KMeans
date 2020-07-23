import numpy as np
from sklearn.datasets import load_boston
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LinearRegression
from sklearn import metrics
from matplotlib import pyplot as plt


data = load_boston()
print(data.data.shape)
x_train, x_test, y_train, y_test = train_test_split(data.data, data.target, test_size=0.1)
LR = LinearRegression()
LR.fit(x_train, y_train)
print(LR.score(x_train, y_train))
y_pre = LR.predict(x_test)
print(metrics.mean_squared_error(y_test, y_pre))

plt.figure(figsize=(10, 6))
plt.plot(y_test, linewidth=3, label="truth")
plt.plot(y_pre, linewidth=3, label='predict')
plt.legend(loc="best")
plt.show()
