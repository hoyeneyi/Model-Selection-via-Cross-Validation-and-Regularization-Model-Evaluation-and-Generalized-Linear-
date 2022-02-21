'''Libraries'''
from sklearn.pipeline import make_pipeline
from sklearn.linear_model import LinearRegression
from sklearn.model_selection import KFold
from sklearn.model_selection import cross_val_score
import pandas as pd
import numpy as np
from sklearn.metrics import mean_squared_error
import matplotlib.pyplot as plt
from sklearn.preprocessing import PolynomialFeatures, StandardScaler
import math
##########################################################################################################
'''Variables'''
deg = 0
rmseMin = float('inf')
Lambda1 =  [0, math.exp(-25), math.exp(-20), math.exp(-14), math.exp(-7), math.exp(-3), 1, math.exp(3), math.exp(7)]

'''Splitting into 6 folds'''
folds = KFold(n_splits=6, random_state=1, shuffle=True)
##########################################################################################################
'''RMSE'''
def rmse(orig, predicted):
    mse = mean_squared_error(orig, predicted)
    rmse = math.sqrt(mse)
    return rmse

'''Pipeline '''
def polyreg(degree):
    return make_pipeline(StandardScaler(), PolynomialFeatures(degree),LinearRegression())

##########################################################################################################

'''Loading data and splitting into x,y train/test'''
deficit_train = pd.read_fwf('deficit_train.dat', header= None)
deficit_train.columns = ["Year", "Deficit"]
deficit_train.info()
print(deficit_train)
xtrain = deficit_train.Year.values.reshape(-1, 1)
ytrain = deficit_train.Deficit.values.reshape(-1, 1)

deficit_test = pd.read_fwf('deficit_test.dat', header= None)
deficit_test.columns = ["Year", "Deficit"]
deficit_test.info()
print(deficit_test)
xtest = deficit_train.Year.values.reshape(-1, 1)
ytest = deficit_train.Deficit.values.reshape(-1, 1)
##########################################################################################################

for degree in range(13):
    pl = polyreg(degree).fit(xtrain, ytrain)
    scores = -1 * cross_val_score(pl, xtrain, ytrain, cv = folds, scoring='neg_root_mean_squared_error')
    print("Degree: ", degree,"\nRSME scores:", scores)
    rmseAvg = np.mean(scores)
    if rmseAvg < rmseMin:
      rmseMin = rmseAvg
      deg = degree
    print("Avg: ", rmseAvg, "\n")
##########################################################################################################

''''Train/Test RMSE'''
print("Min RMSE Average = ", rmseMin, " ; Degree = ", deg)
d = polyreg(9).fit(xtrain, ytrain)
trainRMSE = rmse(ytrain, d.predict(xtrain))
testRMSE = rmse(ytest, d.predict(xtest))
print("train_rmse: ", trainRMSE, "\ntest_RMSE: ", testRMSE)

##########################################################################################################

'''Plotting and Labeling'''
Year = np.arange(1940, 2005, 1, dtype=int).reshape(-1, 1)
plt.title('Federal Budget Deficit')
plt.xlabel('Year')
plt.ylabel('U.S.A. federal budget deficit (in billions of dollars)')
plt.plot(Year, d.predict(Year))
plt.scatter(xtrain, ytrain)