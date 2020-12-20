import pickle
import pandas as pd
import matplotlib.pyplot as plt
import numpy as np
import xlrd , xdrlib
from math import radians, cos, sin, asin, sqrt
from sklearn.linear_model import LinearRegression
from sklearn.preprocessing import MinMaxScaler, StandardScaler, Normalizer
from sklearn.model_selection import train_test_split
import sklearn

df1 = pd.read_excel('DataTesttechniquePFE8.xlsx',engine='openpyxl')

df2 = pd.read_csv('mer-positions.csv')

lon2 = df1['longitude'];
lat2 =df1['latitude'];

lon1 = df2['longitude'];
lat1 =df2['latitude'];

def haversine_vectorize(lon1, lat1, lon2, lat2):

    lon1, lat1, lon2, lat2 = map(np.radians, [lon1, lat1, lon2, lat2])

    newlon = lon2 - lon1
    newlat = lat2 - lat1

    haver_formula = np.sin(newlat/2.0)**2 + np.cos(lat1) * np.cos(lat2) * np.sin(newlon/2.0)**2

    dist = 2 * np.arcsin(np.sqrt(haver_formula ))
    km = 6367 * dist
    return km


haversine_vectorize(lon1 = df2['longitude'],lat1 =df2['latitude'],lon2 = df1['longitude'],lat2 =df1['latitude'])

df1['distance_mer'] = haversine_vectorize(lon1 = df2['longitude'],lat1 =df2['latitude'],lon2 = df1['longitude'],
                                          lat2 =df1['latitude'])

X=df1.iloc[:, 1:6].values
Y=df1.iloc[:, 7].values


#spliting the data into test set and train set
from sklearn.model_selection import train_test_split
xtrain,xtest,ytrain,ytest=train_test_split(X,Y,test_size=0.2,random_state=0)

#model creation
from sklearn.linear_model import LinearRegression
regressior=LinearRegression()
regressior.fit(xtrain, ytrain)

#predcution based on the test
ypred=regressior.predict(xtest)
from sklearn.model_selection import cross_val_score
accuracies=cross_val_score(estimator=regressior, X=xtrain,y=ytrain,cv=10)
mean_accuracy=accuracies.mean()

pickle.dump(regressior,open('model.pkl','wb'))
model=pickle.load(open('model.pkl','rb'))

