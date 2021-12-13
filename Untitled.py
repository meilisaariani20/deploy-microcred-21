#!/usr/bin/env python
# coding: utf-8

# In[1]:


import pandas as pd
import numpy as np


# In[2]:


df = pd.read_csv("INDONESIA_COVID19.csv", sep=";")


# In[4]:


#Library membuat model
from sklearn.metrics import confusion_matrix
from sklearn.metrics import accuracy_score,precision_score,recall_score,f1_score,r2_score
from sklearn.model_selection import train_test_split


# In[5]:


#Train and Test split
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size = 0.3, random_state = 0)


# In[6]:


X=np.asanyarray(df[['Positif Baru (1 hari)']])
Y= np.asanyarray(df[['Sembuh Baru (1 hari)']])


# In[13]:


#library
import numpy as np
import joblib
from sklearn.linear_model import LinearRegression
#dataset 
X=np.asanyarray(df[['Positif Baru (1 hari)']]).reshape((-1, 1))
Y= np.asanyarray(df[['Sembuh Baru (1 hari)']])
#call model regression
model = LinearRegression().fit(X,Y)


# In[14]:


#save model
filename = 'model.sav'
joblib.dump(model, filename)


# In[18]:


#load model
loaded_model = joblib.load(filename)
#prediction model
loaded_model.predict(np.array([10]).reshape(1, 1))

