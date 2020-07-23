from sklearn.datasets import load_iris
from sklearn.model_selection import train_test_split
from sklearn.ensemble import GradientBoostingClassifier
import pandas as pd
import numpy as np

np.set_printoptions(suppress=True, threshold=np.inf)
iris = load_iris()
x_train, x_test, y_train, y_test = train_test_split(iris.data, iris.target)
print(type(x_test), y_train)

gdbt = GradientBoostingClassifier(random_state=10)
gdbt.fit(x_train, y_train)
print(gdbt.train_score_)
y_pre = gdbt.predict(x_test)
print(y_pre)
score = gdbt.score(x_test, y_test)
print(score)