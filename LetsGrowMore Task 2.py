#!/usr/bin/env python
# coding: utf-8

# ### LetsGrowMore : Data Science

# #### Task 2 :Prediction using Decision tree Algorithm

# #### Name of intern : Dhukate Harshala Gajendra

# ### Import Libraries

# In[1]:


import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
get_ipython().run_line_magic('matplotlib', 'inline')
import seaborn as sns
from sklearn.model_selection import train_test_split
from sklearn.tree import DecisionTreeClassifier
from sklearn import metrics
from sklearn.metrics import accuracy_score, plot_confusion_matrix, classification_report,confusion_matrix


# In[5]:


data=pd.read_csv("C:/Users/a/Downloads/Iris (1).csv")
data


# In[7]:


data.head(10)


# In[8]:


data.info()


# In[9]:


data.describe()


# In[10]:


data.shape


# In[11]:


data.isnull().sum()


# In[12]:


data['Species'].unique()


# In[13]:


data['Species'].value_counts()


# ### Explorotory data analysis

# In[14]:


sns.pairplot(data,hue='Species')


# In[16]:


sns.violinplot(y='Species', x='SepalLengthCm', data=data, inner='quartile')
sns.violinplot(y='Species', x='SepalWidthCm', data=data, inner='quartile')
sns.violinplot(y='Species', x='PetalLengthCm', data=data, inner='quartile')
sns.violinplot(y='Species', x='PetalWidthCm', data=data, inner='quartile')
plt.show()


# In[17]:


fig,(ax1,ax2)=plt.subplots(ncols=2,figsize=(16,5))
sns.scatterplot(x='SepalLengthCm',y='PetalLengthCm',data=data,hue='Species',ax=ax1,s=300,marker='o')
sns.scatterplot(x='SepalWidthCm',y='PetalWidthCm',data=data,hue='Species',ax=ax2,s=300,marker='o')


# ### Fitting  decision Tree Classifier

# In[18]:


features = ['SepalLengthCm','SepalWidthCm','PetalLengthCm','PetalWidthCm']
X = data.loc[:, features].values   #defining the feature matrix
y = data.Species
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size = 0.33,random_state=0)


# In[19]:


dtree = DecisionTreeClassifier()
dtree.fit(X_train,y_train)


# In[20]:


from sklearn import tree
feature_name =  ['sepal length(cm)','sepal width(cm)','petal length(cm)','petal width(cm)']
class_name= data.Species.unique()
plt.figure(figsize=(15,10))
tree.plot_tree(dtree, filled = True, feature_names = feature_name, class_names= class_name)


# In[21]:


y_pred = dtree.predict(X_test)
y_pred
score=accuracy_score(y_test,y_pred)
print("Accuracy:",score)


# In[22]:


def report(model):
    preds=model.predict(X_test)
    print(classification_report(preds,y_test))
    plot_confusion_matrix(model,X_test,y_test,cmap='nipy_spectral',colorbar=True)
print('Decision Tree Classifier')
report(dtree)
print(f'Accuracy: {round(score*100,2)}%')
confusion_matrix(y_test, y_pred)
dtree.predict([[5, 3.6, 1.4 , 0.2]])
dtree.predict([[9, 3.1, 5, 1.5]])
dtree.predict([[4.1, 3.0, 5.1, 1.8]])


# ### Thank You

# In[ ]:




