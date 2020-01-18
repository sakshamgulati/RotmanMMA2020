# -*- coding: utf-8 -*-
"""
Created on Fri Nov 15 20:17:55 2019

@author: saksh
"""
#https://machinelearningmastery.com/handle-missing-data-python/
#creating a ANN with dropping na values and ordinal variables

import pandas as pd
from sklearn.ensemble import RandomForestRegressor
import numpy
import pandas as pd
from sklearn import preprocessing
from keras.models import Sequential
from keras.layers import Dense
from sklearn.ensemble import BaggingClassifier
df = pd.read_csv("C:/Users/saksh/Desktop/Rotman/Fall/Big data analytics/Group Assignment/ANN/finalfinaldata.csv")

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
churn =pd.read_csv('C:/Users/saksh/Desktop/Rotman/Fall/Big data analytics/Group Assignment/ANN/USCensusTraining.csv')
churn.shape

#show first 10 rows
churn.head(10)

#to see the diff types of values
churn.dtypes

#to see if any object is null or not null
churn.info()

#to analyze the outcome data
churn["income"]
#to summarize the content
churn["income"].value_counts()
#to summarize as %
churn["income"].value_counts(normalize=True)*100
#to visualize the outcome variable
sns.countplot(churn["income"])
#see the relationship of predictor with the outcome variable
sns.countplot(x="education",hue="income",data=churn)

#how to provide a crosstab of the relationshup without and with percentages
pd.crosstab(churn["income"],churn[("education")])
pd.crosstab(churn["income"],churn[("education")]).apply(lambda r: r/r.sum(), axis=0)

#exploring numeric variables
#provides a mean, std and quartile distribution
churn.describe()
#provide a histogram for all
churn.hist()
#provide a boxplot for all
churn.boxplot()
#rpivde a density plot of all the plots
churn.plot(kind="density", subplots=True, layout=(4,4),sharex=False)

churn.plot(kind="box", subplots=True, layout=(4,4),sharex=True)

#draw a particular box plot
sns.boxplot(x="Churn?",y="CustServ Calls",data=churn)

#multivariate relationships
from pandas.plotting import  scatter_matrix
scatter_matrix(churn)

#perform a basic scatterplot 
sns.scatterplot(x="Day Mins",y="Eve Mins",hue="Churn?",data=churn)

sns.scatterplot(x="Day Mins",y="CustServ Calls",hue="Churn?",data=churn)
#using correlations
correlations= churn.corr()
correlations




df=df.dropna()
df.head()



scaler = preprocessing.MinMaxScaler()
scaled_dataset = scaler.fit_transform(df)
pd.DataFrame(scaled_dataset).describe()
X = scaled_dataset[:,0:12]
Y = scaled_dataset[:,12]

model = Sequential()
model.add(Dense(16, input_dim=12, activation='relu'))

model.add(Dense(13, activation='relu'))

model.add(Dense(6, activation='relu'))

model.add(Dense(1, activation='sigmoid'))
model.compile(loss='binary_crossentropy', optimizer='adam', metrics=['mean_squared_error'])
model.fit(X, Y, epochs=20, batch_size=10)

accuracy = model.evaluate(X, Y)

predictions = model.predict(X)
print(predictions)

from sklearn.model_selection import GridSearchCV
from keras.wrappers.scikit_learn import KerasClassifier
from keras.utils.vis_utils import plot_model
import keras
import pydot as pyd
from IPython.display import SVG
from keras.utils.vis_utils import model_to_dot

keras.utils.vis_utils.pydot = pyd
import keras
import pydotplus
from keras.utils.vis_utils import model_to_dot
keras.utils.vis_utils.pydot = pydot

#create your model
#then call the function on your model
def create_model(neurons=1):
    model = Sequential()
    model.add(Dense(neurons, input_dim=12, activation="relu"))
    model.add(Dense(neurons, activation="relu"))
    model.add(Dense(1, activation="sigmoid"))
    model.compile(loss="binary_crossentropy", optimizer="adam", metrics=['accuracy'])
    print(model.summary())
    plot_model(model, to_file='model.png')
    return model

model = KerasClassifier(build_fn=create_model, epochs=5, batch_size=250, verbose=1)

neurons = [46,64,128]
param_grid = dict(neurons = neurons)
grid = GridSearchCV(estimator = model, param_grid= param_grid)
grid_result = grid.fit(X,Y)
stds = grid_result.cv_results_["std_test_score"]
parms = grid_result.cv_results_["params"]
means = grid_result.cv_results_["mean_test_score"]

for mean, stdev, param in zip(means, stds, parms):
    print("%f (%f) with: %r" % (mean, stdev, param))


X_train=df.drop('income',axis=1)
X_test=df['income']
import eli5
from eli5.sklearn import PermutationImportance
model.fit(X_train,X_test)

perm=PermutationImportance(model).fit(X_train,X_test)

test=eli5.show_weights(perm, feature_names = X_train.columns.tolist())


from ann_visualizer.visualize import ann_viz

ann_viz(model, title="My first neural network")