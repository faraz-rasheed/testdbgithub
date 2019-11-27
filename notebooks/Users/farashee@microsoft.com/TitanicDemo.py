# Databricks notebook source
# MAGIC %md
# MAGIC 
# MAGIC ## Predicting surviving Titanic passengers - Databricks Spark

# COMMAND ----------

#%matplotlib inline

import pickle
import sys
import os
import pandas as pd

import numpy as np
from sklearn.metrics import confusion_matrix

from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import train_test_split
from sklearn.metrics import precision_recall_curve
from sklearn.metrics import roc_curve, auc, roc_auc_score
from sklearn.neighbors import KNeighborsClassifier

from matplotlib import pyplot

# COMMAND ----------

df = spark.table("titanic")
display(df)

# COMMAND ----------

df = df.toPandas()

# COMMAND ----------

df.dtypes

# COMMAND ----------

df.head(3)

# COMMAND ----------

#import pandas_profiling
#rpt = pandas_profiling.ProfileReport(df)

# COMMAND ----------

df = df[['pclass', 'survived', 'sex', 'age', 'sibsp', 'parch', 'fare', 'embarked']]
df.head(4)

# COMMAND ----------

dft=pd.concat([df,pd.get_dummies(df[['pclass', 'sex', 'embarked']],prefix=['pclass', 'sex', 'embarked'])],axis=1).drop(['pclass', 'sex', 'embarked'], axis=1)
dftp = dft.drop_duplicates().fillna(0)
dftp.head(4)

# COMMAND ----------

# load features and labels
X, Y = dftp.drop('survived', axis=1).values, dftp['survived'].values

# COMMAND ----------

# split data 65%-35% into training set and test set
X_train, X_test, Y_train, Y_test = train_test_split(X, Y, test_size=0.35, random_state=0)
#print(X_train)

# COMMAND ----------

# change regularization rate and you will likely get a different accuracy.
reg = 0.01
print("Regularization rate is {}".format(reg))

# train a logistic regression model on the training set
clf1 = LogisticRegression(C=1.0/reg).fit(X_train, Y_train)
print (clf1)

# COMMAND ----------

# evaluate the test set
accuracy = clf1.score(X_test, Y_test)
print ("Accuracy is {}".format(accuracy))

# COMMAND ----------

scd = pd.read_csv("/dbfs/mnt/frzstg/titanic_scoring.csv")
y_scores = clf1.predict(scd)

np.savetxt("/dbfs/mnt/frzstg/scoring_results.csv", y_scores, delimiter=",")

#type(y_scores)
#print(y_scores)
#np.set_printoptions(edgeitems=10)
#print(X_test)

# COMMAND ----------

# calculate and log precesion, recall, and thresholds, which are list of numerical values
y_scores = clf1.predict_proba(X_test)
precision, recall, thresholds = precision_recall_curve(Y_test, y_scores[:,1],pos_label=1.0)

print("Precision {}".format(precision))
print("Recall {}".format(recall))
print("Thresholds {}".format(thresholds))

# COMMAND ----------

# calculate AUC
auc = roc_auc_score(Y_test, y_scores[:,1])
print('AUC: %.3f' % auc)



# COMMAND ----------

fpr, tpr, thresholds = roc_curve(Y_test, y_scores[:,1])

fig = pyplot.gcf()
# plot no skill
pyplot.plot([0, 1], [0, 1], linestyle='--')
# plot the roc curve for the model
pyplot.plot(fpr, tpr, marker='.')
# show the plot
display(fig)

# COMMAND ----------

