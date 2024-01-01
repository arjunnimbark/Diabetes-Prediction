# -*- coding: utf-8 -*-
"""Diabetes.ipynb

Automatically generated by Colaboratory.

Original file is located at
    https://colab.research.google.com/drive/1s2eUZGPLaYp4jqD4Jlf3ErKgYDbo1xlW
"""

import numpy as np
import pandas as pd
from sklearn.preprocessing import StandardScaler
from sklearn.model_selection import train_test_split
from sklearn import svm
from sklearn.metrics import accuracy_score

data=pd.read_csv("/content/diabetes.csv")
data.head()

data["Outcome"].value_counts()

data.groupby("Outcome").mean()

#separating data and labels
X=data.drop(columns="Outcome",axis=1)
Y=data["Outcome"]

#STANDARDIZING DATA
scaler=StandardScaler()
scaler.fit(X)

standardized=scaler.transform(X)
print(standardized)

X=standardized
Y=data["Outcome"]
print(X)
print(Y)

#TRAIN TEST SPLIT
X_train,X_test,Y_train,Y_test=train_test_split(X,Y,test_size=0.2,stratify=Y,random_state=2)
print(X.shape,X_train.shape,X_test.shape)

#TRAINING MODEL

classifier=svm.SVC(kernel="linear")
classifier.fit(X_train,Y_train)

#MODEL EVALUATION for TRAINING
X_train_pred=classifier.predict(X_train)
training_acc=accuracy_score(X_train_pred,Y_train)
print(training_acc)

X_test_pred=classifier.predict(X_test)
test_acc=accuracy_score(X_test_pred,Y_test)
print(test_acc)

#MAKING PREDICTIVE SYStem

input_data=(5,166,72,19,175,25.0,0.507,51)
asnumpy=np.asarray(input_data)
reshaped=asnumpy.reshape(1,-1)

#print(std_data)
prediction=classifier.predict(reshaped)
print(prediction)
if(prediction[0]==0):
  print("the patient is non diabetic")
else:
  print("diabetic")



import pickle
filename="trained_model.sav"
pickle.dump(classifier,open(filename,"wb"))

#LOADING THE SAVED MODEL
loaded_model=pickle.load(open("trained_model.sav","rb"))

#MAKING PREDICTIVE SYStem

input_data=(5,166,72,19,175,25.0,0.507,51)
asnumpy=np.asarray(input_data)
reshaped=asnumpy.reshape(1,-1)

#print(std_data)
prediction=loaded_model.predict(reshaped)
print(prediction)
if(prediction[0]==0):
  print("the patient is non diabetic")
else:
  print("diabetic")