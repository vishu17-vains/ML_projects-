# importing the liabraries 
import numpy as np 
import pandas as pd 
import matplotlib.pyplot as plt 

# importing the dataset 
dataset = pd.read_csv(r"C:\Users\Administrator\Desktop\FSDS\13th - NAIVE BAYES\Social_Network_Ads.csv")
 
# spliting the dataset 
X = dataset.iloc[:,[2,3]]
y = dataset.iloc[:,4]

# spliting the data into training and test set 
from sklearn.model_selection import train_test_split 
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size = 0.20, random_state = 0)

# feature scaling 
from sklearn.preprocessing import Normalizer
sc = Normalizer()
X_train = sc.fit_transform(X_train)
X_test = sc.fit_transform(X_test)

# training the naive bayes model on the training set 
from sklearn.naive_bayes import MultinomialNB
classifier = MultinomialNB()
classifier.fit(X_train, y_train)

# predicting the test set result 
y_pred = classifier.predict(X_test)

# confusion matrix 
from sklearn.metrics import confusion_matrix
cm = confusion_matrix(y_test, y_pred)
print(cm)

# accuracy score 
from sklearn.metrics import accuracy_score
ac = accuracy_score(y_test, y_pred)
print(ac)

# Bias 
bias = classifier.score(X_train, y_train)
print(bias)

# varience 
varience = classifier.score(X_test, y_test)
print(varience)