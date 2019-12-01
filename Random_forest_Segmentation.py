# -*- coding: utf-8 -*-
"""
Created on Sat Nov 30 21:30:10 2019

@author: saksh
"""

# -*- coding: utf-8 -*-
"""
Created on Wed Nov 27 18:58:21 2019

@author: saksh
"""

from sklearn.cluster import KMeans
from sklearn.metrics import silhouette_samples, silhouette_score

import pandas as pd
import matplotlib.pyplot as plt
import matplotlib.cm as cm
import numpy as np

print(__doc__)


data = "C:/Users/saksh/Desktop/Rotman/Fall/Big data analytics/Group Assignment/Last Assignment/data.csv"
data=pd.read_csv(data)
data.head()

X = np.array(data.astype(float))

print(X)
range_n_clusters = [2, 3, 4, 5, 6]

#kmeans

for n_clusters in range_n_clusters:
    # Create a subplot with 1 row and 2 columns
    fig, (ax1, ax2) = plt.subplots(1, 2)
    fig.set_size_inches(18, 7)

    # The 1st subplot is the silhouette plot
    # The silhouette coefficient can range from -1, 1 but in this example all
    # lie within [-0.1, 1]
    ax1.set_xlim([-0.1, 1])
    # The (n_clusters+1)*10 is for inserting blank space between silhouette
    # plots of individual clusters, to demarcate them clearly.
    ax1.set_ylim([0, len(X) + (n_clusters + 1) * 10])

    # Initialize the clusterer with n_clusters value and a random generator
    # seed of 10 for reproducibility.
    clusterer = KMeans(n_clusters=n_clusters, random_state=10)
    cluster_labels = clusterer.fit_predict(X)

    # The silhouette_score gives the average value for all the samples.
    # This gives a perspective into the density and separation of the formed
    # clusters
    silhouette_avg = silhouette_score(X, cluster_labels)
    print("For n_clusters =", n_clusters,
          "The average silhouette_score is :", silhouette_avg)

    # Compute the silhouette scores for each sample
    sample_silhouette_values = silhouette_samples(X, cluster_labels)

    y_lower = 10
    for i in range(n_clusters):
        # Aggregate the silhouette scores for samples belonging to
        # cluster i, and sort them
        ith_cluster_silhouette_values = \
            sample_silhouette_values[cluster_labels == i]

        ith_cluster_silhouette_values.sort()

        size_cluster_i = ith_cluster_silhouette_values.shape[0]
        y_upper = y_lower + size_cluster_i

        color = cm.nipy_spectral(float(i) / n_clusters)
        ax1.fill_betweenx(np.arange(y_lower, y_upper),
                          0, ith_cluster_silhouette_values,
                          facecolor=color, edgecolor=color, alpha=0.7)

        # Label the silhouette plots with their cluster numbers at the middle
        ax1.text(-0.05, y_lower + 0.5 * size_cluster_i, str(i))

        # Compute the new y_lower for next plot
        y_lower = y_upper + 10  # 10 for the 0 samples

    ax1.set_title("The silhouette plot for the various clusters.")
    ax1.set_xlabel("The silhouette coefficient values")
    ax1.set_ylabel("Cluster label")

    # The vertical line for average silhouette score of all the values
    ax1.axvline(x=silhouette_avg, color="red", linestyle="--")

    ax1.set_yticks([])  # Clear the yaxis labels / ticks
    ax1.set_xticks([-0.1, 0, 0.2, 0.4, 0.6, 0.8, 1])

    # 2nd Plot showing the actual clusters formed
    colors = cm.nipy_spectral(cluster_labels.astype(float) / n_clusters)
    ax2.scatter(X[:, 0], X[:, 1], marker='.', s=30, lw=0, alpha=0.7,
                c=colors, edgecolor='k')

    # Labeling the clusters
    centers = clusterer.cluster_centers_
    # Draw white circles at cluster centers
    ax2.scatter(centers[:, 0], centers[:, 1], marker='o',
                c="white", alpha=1, s=200, edgecolor='k')

    for i, c in enumerate(centers):
        ax2.scatter(c[0], c[1], marker='$%d$' % i, alpha=1,
                    s=50, edgecolor='k')

    ax2.set_title("The visualization of the clustered data.")
    ax2.set_xlabel("Feature space for the 1st feature")
    ax2.set_ylabel("Feature space for the 2nd feature")

    plt.suptitle(("Silhouette analysis for KMeans clustering on sample data "
                  "with n_clusters = %d" % n_clusters),
                 fontsize=14, fontweight='bold')

plt.show()

kmeans = KMeans(n_clusters=2)
kmeans.fit(data)
labels=kmeans.predict(data)
print(labels)
type(labels)

cluster_result = labels.tolist()
data["cluster number"] = cluster_result
#data.to_csv(r'C:\Users\saksh\Desktop\Rotman\Fall\Big data analytics\Group Assignment\Last Assignment\File Name.csv')






import pandas as pd  
import numpy as np  
import matplotlib.pyplot as plt  
import seaborn as seabornInstance 
from sklearn.model_selection import train_test_split 
from sklearn.linear_model import LinearRegression
from sklearn import metrics

import pandas as pd

# Using Skicit-learn to split data into training and testing sets
from sklearn.model_selection import train_test_split

x = data.iloc[:, 0:12].values  
y = data.iloc[:, 12].values  


X_train, X_test, y_train, y_test = train_test_split(x, y, test_size = 0.25, random_state = 42)

from sklearn.ensemble import RandomForestRegressor
#nestimators is the number of trees in the forest
rf = RandomForestRegressor(n_estimators = 100, random_state = 42)
rf.fit(X_train, y_train);
import numpy as np

np.seterr(divide='ignore', invalid='ignore')

predictions = rf.predict(X_test)

errors = abs(predictions - y_test)
print('Metrics for Random Forest Trained on Original Data')
print('Average absolute error:', round(np.mean(errors), 2), 'degrees.')
# Print out the mean absolute error (mae)

data2 = "C:/Users/saksh/Desktop/Rotman/Fall/Big data analytics/Group Assignment/Last Assignment/test.csv"
data2=pd.read_csv(data2)
data2.head()
len(data2)


X = np.array(data2.astype(float))

kmeans = KMeans(n_clusters=2)

kmeans.fit(data2)
labels=kmeans.predict(data2)
print(labels)
type(labels)

cluster_result = labels.tolist()
data2["cluster number"] = cluster_result
len(data2)
#data.to_csv(r'C:\Users\saksh\Desktop\Rotman\Fall\Big data analytics\Group Assignment\Last Assignment\File Name.csv')


import numpy
import pandas as pd
from sklearn import preprocessing
from keras.models import Sequential
from keras.layers import Dense
from sklearn.model_selection import cross_val_score
from sklearn.model_selection import KFold
from keras.wrappers.scikit_learn import KerasRegressor

seed = 7
numpy.random.seed(seed)
julie=data2[data2["cluster number"]==0]
len(julie)
julie.to_csv(r'C:\Users\saksh\Desktop\Rotman\Fall\Big data analytics\Group Assignment\Last Assignment\Julie.csv')

data3=data2[data2["cluster number"]==1]
data3=data3.drop(["cluster number"], 1)
len(data3)


data3.head()
#julie.to_csv(r'C:\Users\saksh\Desktop\Rotman\Fall\Big data analytics\Group Assignment\Last Assignment\julie.csv')



pred = rf.predict(data3)

data3['Prediction'] = pred
data3.to_csv(r'C:\Users\saksh\Desktop\Rotman\Fall\Big data analytics\Group Assignment\Last Assignment\RF.csv')
