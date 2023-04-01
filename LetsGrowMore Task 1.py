#!/usr/bin/env python
# coding: utf-8

# ### LetsGrowMore : Data Science

# #### Task 1 : Iris Flower Classification ML Project  

# #### Name : Dhukate Harshala Gajendra

# ### Step 1 : Importing Libraries

# In[1]:


import pandas as pd
import numpy as np
import os
import matplotlib.pyplot as plt
import seaborn as sns


# ### step 2 : Load The Dataset 

# In[2]:


df = pd.read_csv("C:/Users/a/Downloads/Iris (1).csv")
df.head()


# In[3]:


df.shape


# In[4]:


#To display stats about data
df.describe()


# In[5]:


df.info()


# In[6]:


#check null values
df.isnull().sum()


# In[7]:


df1=df[["SepalLengthCm" , "SepalWidthCm" , "PetalLengthCm" , "PetalWidthCm" , "Species"]]
print(df1.head())


# ###  Step 3 : Apply EDA

# In[8]:


df['SepalLengthCm'].hist()


# In[9]:


df['SepalWidthCm'].hist()


# In[10]:


df['PetalLengthCm'].hist()


# In[11]:


df['PetalWidthCm'].hist()


# In[12]:


df['Species'].hist()


# In[13]:


#scraterplot
colors =['red','orange','blue']
Species = ['Iris-setosa','Iris-versicolor','Iris-virginica']


# In[14]:


for i in range(3):
    x = df[df['Species'] == Species[i]]
    plt.scatter(x['SepalLengthCm'], x['SepalWidthCm'], c = colors[i],label=Species[i])
    plt.xlabel("SepalLengthCm")
    plt.ylabel("SepalWidthCm")


# In[15]:


for i in range(3):
    x = df[df['Species'] == Species[i]]
    plt.scatter(x['PetalLengthCm'], x['PetalWidthCm'], c = colors[i],label=Species[i])
    plt.xlabel("PetalLengthCm")
    plt.ylabel("PetalWidthCm")


# In[16]:


for i in range(3):
    x = df[df['Species'] == Species[i]]
    plt.scatter(x['SepalLengthCm'], x['PetalLengthCm'], c = colors[i],label=Species[i])
    plt.xlabel("SepalLengthCm'")
    plt.ylabel("PetalLengthCm")


# In[17]:


for i in range(3):
    x = df[df['Species'] == Species[i]]
    plt.scatter(x['SepalWidthCm'], x['PetalWidthCm'], c = colors[i],label=Species[i])
    plt.xlabel("SepalWidthCm")
    plt.ylabel("PetalWidthCm")


# In[18]:


df.corr()


# ### show in matrics form

# In[19]:


corr =df.corr()
fig,ax = plt.subplots(figsize=(5,4))
sns.heatmap(corr, annot=True, ax=ax)


# In[20]:


from sklearn.preprocessing import LabelEncoder
le = LabelEncoder()


# In[21]:


df['Species'] = le.fit_transform(df['Species'])
df.head()


# ### Step 4 : Model Training

# In[22]:


from sklearn.model_selection import train_test_split
X = df.drop(columns=['Species'])
Y = df['Species']
x_train, x_test, y_train, y_test = train_test_split(X, Y, test_size=0.30)


# In[23]:


#decesion tree
from sklearn.tree import DecisionTreeClassifier
model = DecisionTreeClassifier()


# In[24]:


model.fit(x_train, y_train)


# In[25]:


#print metrics
print("Accuracy:",model.score(x_test,y_test)*100)


#  ### Thank You
