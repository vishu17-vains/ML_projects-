#importing the liabraries 

import numpy as np 
import pandas as pd 
import matplotlib.pyplot as plt 

# importing the dataset 

dataset = pd.read_csv(r"C:\Users\Administrator\Desktop\FSDS\2.LOGISTIC REGRESSION CODE\logit classification.csv")


X = dataset.iloc[:, [2, 3]].values
y = dataset.iloc[:, -1].values

# Splitting the dataset into the Training set and Test set
from sklearn.model_selection import train_test_split

#for this observation let me selcted as 100 observaion for test set
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size = 0.20,random_state=0)

# Feature Scaling
from sklearn.preprocessing import Normalizer
sc = Normalizer() 
X_train = sc.fit_transform(X_train)
X_test = sc.transform(X_test) 

# Training the Logistic Regression model on the Training set

from sklearn.linear_model import LogisticRegression
classifier = LogisticRegression()
classifier.fit(X_train, y_train)


# Predicting the Test set results
y_pred = classifier.predict(X_test)


# Making the Confusion Matrix
from sklearn.metrics import confusion_matrix
cm = confusion_matrix(y_test, y_pred)
print(cm)

# This is to get the Models Accuracy 
from sklearn.metrics import accuracy_score 
ac = accuracy_score(y_test, y_pred)
print(ac) 

# This is to get the Classification Report
from sklearn.metrics import classification_report
cr = classification_report(y_test, y_pred)
cr


bias = classifier.score(X_train, y_train)
bias

variance = classifier.score(X_test, y_test)
variance
#**********************************************************************************
#                               FUTURE PREDICTION
#**********************************************************************************

dataset1 = pd.read_csv(r"C:\Users\Administrator\Desktop\FSDS\15. Logistic regression with future prediction\Future prediction1.csv")
d2 = dataset1.copy()

dataset1 = dataset1.iloc[:, [2, 3]].values

# performing feature scalling (using standardization)
from sklearn.preprocessing import StandardScaler
sc = StandardScaler()
M = sc.fit_transform(dataset1)

y_pred1 = pd.DataFrame()


d2 ['y_pred1'] = classifier.predict(M)

d2.to_csv('pred_model.csv')

# To get the path 
import os
os.getcwd()

                       