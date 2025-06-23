import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D
from scipy import stats
import seaborn as sns
from sklearn import preprocessing, svm
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LinearRegression
from pydataset import data

df = pd.read_csv("C:/Users/zeebu/OneDrive/Documents/Python/Multiple-Regression-Models/startups.csv")

X = df[['Administration', 'R&D Spend']]
x = df[['Administration']]
y = df[['R&D Spend']]
z = df[['Profit']]

MR_model = LinearRegression()
MR_model.fit(X,z)
predictions = MR_model.predict(X)

X1, Y1 = np.meshgrid(x,y)
ax: Axes3D = plt.figure().add_subplot(projection="3d")
surf = ax.plot_surface(X1, Y1, z, cmap="coolwarm", rstride=1, cstride=1)
ax.view_init(20, -120)
ax.set_xlabel("Admin")
ax.set_ylabel("R&D")
ax.set_zlabel("Profit")

plt.show()