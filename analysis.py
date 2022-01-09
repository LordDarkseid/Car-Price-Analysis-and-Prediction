!pip install pandas==1.3.3
!pip install numpy==1.21.2
!pip install scikit-learn==0.20.1

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import os
import seaborn as sns
%matplotlib inline
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler
from sklearn.preprocessing import PolynomialFeatures
from sklearn.metrics import mean_squared_error
from sklearn.metrics import r2_score

os.getcwd()
df = pd.read_csv('automobileEDA.csv')
df.head()

# Simple Linear Regression
from sklearn.linear_model import LinearRegression
lm_hmpg = LinearRegression()
lm_hmpg

# Using highway miles per gallon and price
X = df[['highway-mpg']]
Y = df['price']
lm_hmpg.fit(X,Y)
Yhat_hmpg=lm_hmpg.predict(X)
Yhat_hmpg

lm_hmpg.intercept_
lm_hmpg.coef_
# Price = 38423.31 - 821.73 x highway-mpg

# Using Engine Size and Car Price
lm_es = LinearRegression()
lm_es.fit(df[['engine-size']], df[['price']])
lm_es

lm_es.coef_
lm_es.intercept_
# Price=-7963.34 + 166.86*engine-size
Yhat_es=lm_es.predict(X)
Yhat_es

df[["peak-rpm","highway-mpg","horsepower", "curb-weight", "engine-size","price"]].corr()

# Using Multiple Linear Regression (using factors with high correlation)
# Factors used->
# Horsepower
# Curb Weight
# Engine Size
# Highway mpg 

Z = df[['horsepower', 'curb-weight', 'engine-size', 'highway-mpg']]
lm_mlr = LinearRegression()
lm_mlr.fit(Z, df['price'])

lm_mlr.intercept_
lm_mlr.coef_

# Price = -15806.62 + 53.49 x horsepower + 4.70 x curb-weight + 81.53 x engine-size + 36.05 x highway-mpg

# Regression Plots
sns.regplot(x="highway-mpg", y="price", data=df)
plt.ylim(0,)

sns.regplot(x="curb-weight", y="price", data=df)
plt.ylim(0,)

sns.regplot(x="horsepower", y="price", data=df)
plt.ylim(0,)

sns.regplot(x="engine-size", y="price", data=df)
plt.ylim(0,)

# Residual Plots
sns.residplot(df['highway-mpg'], df['price'])
plt.show()

# Checking Fit
Y_hat = lm_mlr.predict(Z)
ax1 = sns.distplot(df['price'], hist=False, color="r", label="Actual Value")
sns.distplot(Y_hat, hist=False, color="b", label="Fitted Values" , ax=ax1)

plt.title('Actual vs Fitted Values for Price')
plt.xlabel('Price (in dollars)')
plt.ylabel('Proportion of Cars')

plt.show()
plt.close()

# Polynomial Regression
# Plotting Function
def PlotPoly(model, independent_variable, dependent_variable, Name):
    x_new = np.linspace(15, 55, 100)
    y_new = model(x_new)

    plt.plot(independent_variable, dependent_variable, '.', x_new, y_new, '-')
    plt.title('Polynomial Fit')
    ax = plt.gca()
    ax.set_facecolor((0.898, 0.898, 0.898))
    fig = plt.gcf()
    plt.xlabel(Name)
    plt.ylabel('Price of Cars')

    plt.show()
    plt.close()
    
x = df['highway-mpg']
y = df['price']
f = np.polyfit(x, y, 3)
p = np.poly1d(f)
print(p)
PlotPoly(p, x, y, 'highway-mpg')
np.polyfit(x, y, 3)

# Pipeline
Input=[('scale',StandardScaler()), ('polynomial', PolynomialFeatures(include_bias=False)), ('model',LinearRegression())]
pipe=Pipeline(Input)
pipe
Z = Z.astype(float)
pipe.fit(Z,y)
ypipe=pipe.predict(Z)
ypipe

# Evaluating MSE and R^2 errors
# For Highway Mile per Gallon and Price model
print('The R-square is: ', lm_hmpg.score(X, Y))
mse_hmpg = mean_squared_error(df['price'], Yhat_hmpg)
print('The mean square error of price and predicted value is: ', mse_hmpg)

# For Multiple Linear Regression Model
print('The R-square is: ', lm_mlr.score(Z, df['price']))
Y_predict_multifit = lm_mlr.predict(Z)
Y_predict_multifit = lm_mlr.predict(Z)
print('The mean square error of price and predicted value using multifit is: ', mean_squared_error(df['price'], Y_predict_multifit))

# For Polynomial Fit
r_squared = r2_score(y, p(x))
print('The R-square value is: ', r_squared)
mean_squared_error(df['price'], p(x))
