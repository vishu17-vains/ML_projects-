# importing the liabraries

import numpy as np 
import pandas as pd
import matplotlib.pyplot as plt 

# importing the dataset 
dataset = pd.read_csv(r"C:\Users\Administrator\Desktop\FSDS\11th - SVM\Social_Network_Ads.csv")

# slicing the dependent and independent variable from the dataset 

X  = dataset.iloc[:,[2,3]]
y = dataset.iloc[:,4]

# spliting the dataset into training and testing dataset 
from sklearn.model_selection import train_test_split 
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size = 0.20, random_state = 0)

# feature scaling 
from sklearn.preprocessing import StandardScaler
sc = StandardScaler()
sc.fit_transform(X_train, X_test)

# training the svm model on training set 
from sklearn.svm import SVC
classifier = SVC()
classifier.fit(X_train, y_train)

# predicting the test set result 
y_pred = classifier.predict(X_test)

# Confusion matrix 

from sklearn.metrics import confusion_matrix
cm = confusion_matrix(y_test,y_pred)
print(cm)

# checking the accuracy 

from sklearn.metrics import accuracy_score
ac = accuracy_score(y_test, y_pred)
print(ac)

# bias 
bias = classifier.score(X_train, y_train)
print(bias)

# variance 
varience = classifier.score(X_test, y_test)
print(varience)

#=======================================================================================================
#                                  FUTURE PREDICTION 
#=======================================================================================================

# IMPORTING THE DATASET ON WHICH WE HAVE TO PREDICT THE FUTURE 

dataset1 = pd.read_csv(r"C:\Users\Administrator\Desktop\FSDS\11th - SVM\Future prediction1.csv")
print(dataset1)  # as we can see purchase feature in dataset1 is missing , so we have to predict it 


# copy the dataset1
d2 = dataset1.copy()

# slicing the dataset on which model have to make prediction 
dataset1 = dataset1.iloc[:,[2,3]].values 
dataset1

# performing feature scaling on dataset1 

from sklearn.preprocessing import StandardScaler
sc = StandardScaler()
M = sc.fit_transform(dataset1)

y_pred1 = pd.DataFrame()


d2 ['y_pred1'] = classifier.predict(M)

d2.to_csv('pred_model_svm.csv')

# To get the path 
import os
os.getcwd()


















