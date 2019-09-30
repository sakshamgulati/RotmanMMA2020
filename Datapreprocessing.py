# -*- coding: utf-8 -*-
"""
Created on Mon Sep 30 12:54:21 2019

@author: saksh
"""


import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
churn =pd.read_csv('C:/Users/saksh/Desktop/Rotman/Fall/Big data analytics/churn.csv')

#rows and columns
churn.shape

#show first 10 rows
churn.head(10)

#to see the diff types of values
churn.dtypes

#to see if any object is null or not null
churn.info()

#to analyze the outcome data
churn["Churn?"]
#to summarize the content
churn["Churn?"].value_counts()
#to summarize as %
churn["Churn?"].value_counts(normalize=True)*100
#to visualize the outcome variable
sns.countplot(churn["Churn?"])
#see the relationship of predictor with the outcome variable
sns.countplot(x="Int'l Plan",hue="Churn?",data=churn)

#how to provide a crosstab of the relationshup without and with percentages
pd.crosstab(churn["Churn?"],churn[("Int'l Plan")])
pd.crosstab(churn["Churn?"],churn[("Int'l Plan")]).apply(lambda r: r/r.sum(), axis=0)

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

