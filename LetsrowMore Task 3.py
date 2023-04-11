#!/usr/bin/env python
# coding: utf-8

# ### LetsGrowMore : Data Science

# #### Task 2:Intermediate Level Task) Exploratory Data Analysis on Terrorism Dataset

# #### Name of intern : Dhukate Harshala Gajendra

# ### Step 1 : Import Libraries

# In[1]:


import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns


#  ### Step 2 : Loading the dataset

# In[2]:


data = pd.read_csv("C:/Users/a/Downloads/globalterrorismdb_0718dist.csv",encoding = "Latin", low_memory=False)
data.head()


# In[3]:


data=data[["iyear","imonth","iday","country_txt","region_txt","provstate","city",
           "latitude","longitude","location","summary","attacktype1_txt","targtype1_txt",
           "gname","motive","weaptype1_txt","nkill","nwound","addnotes"]]
data.head()


# In[4]:


data.rename(columns={"iyear":"Year","imonth":"Month","iday":"Day","country_txt":"Country",
                    "region_txt":"Region","provstate":"Province/State","city":"City",
                    "latitude":"Latitude","longitude":"Longitude","location":"Location",
                    "summary":"Summary","attacktype1_txt":"Attack Type","targtype1_txt":"Target Type",
                    "gname":"Group Name","motive":"Motive","weaptype1_txt":"Weapon Type",
                    "nkill":"Killed","nwound":"Wounded","addnotes":"Add Notes"},inplace=True)


# In[5]:


data.head()


# In[6]:


data.info()


# In[7]:


data.shape


# In[8]:


data.isnull().sum()


# In[9]:


data["Killed"]=data["Killed"].fillna(0)
data["Wounded"]=data["Wounded"].fillna(0)
data["Casualty"]=data["Killed"]+data["Wounded"]


# In[10]:


data.describe()


# ### Step 3 : Performing EDA

# In[11]:


data.Region.value_counts()


# In[12]:


plt.figure(figsize=(22,10))
plt.bar(x=data.Region.value_counts().index,height=data.Region.value_counts().values)
plt.title('Regionwise terrorist attack')
plt.xlabel('Region')
plt.ylabel('No of terrorist attack')


# In[13]:


killed=data["Region"].value_counts(dropna=False).sort_index().to_frame().reset_index().rename(columns={"index":"Region","Region":"Killed"}).set_index("Region")
killed.head()


# In[14]:


killed.plot(kind="bar",color="blue",figsize=(15,6),fontsize=15)
plt.title('Terrorist attack',fontsize=12)
plt.xlabel('Region',fontsize=12)
plt.ylabel('No of peoples killed',fontsize=12)
plt.show()


# In[15]:


attacks=data["Year"].value_counts(dropna=False).sort_index().to_frame().reset_index().rename(columns={"index":"Year","Year":"Attacks"}).set_index("Year")
attacks.head()


# In[16]:


attacks.plot(kind="bar",color="blue",figsize=(15,6),fontsize=15)
plt.title('Terrorist attack',fontsize=12)
plt.xlabel('Year',fontsize=12)
plt.ylabel('No of terrorist attack',fontsize=12)
plt.show()


# In[17]:


country=data["Country"].value_counts().head(10)
country


# In[18]:


country.plot(kind="bar",color="blue",figsize=(15,6),fontsize=15)
plt.title('Terrorist attack',fontsize=12)
plt.xlabel('Country',fontsize=12)
plt.ylabel('No of terrorist attack',fontsize=12)
plt.show()


# In[19]:


count=data[["Country","Casualty"]].groupby("Country").sum().sort_values(by="Casualty",ascending=False)
count.head(10)


# In[20]:


count[:10].plot(kind="bar",color="blue",figsize=(15,6),fontsize=15)
plt.title('Terrorist attack',fontsize=12)
plt.xlabel('Country',fontsize=12)
plt.ylabel('No of Casualties',fontsize=12)
plt.show()


# In[21]:


countk=data[["Country","Killed"]].groupby("Country").sum().sort_values(by="Killed",ascending=False)
countk.head(10)


# In[22]:


countw=data[["Country","Wounded"]].groupby("Country").sum().sort_values(by="Wounded",ascending=False)
countw.head(10)


# In[23]:


fig=plt.figure()
ax0=fig.add_subplot(1,2,1)
ax1=fig.add_subplot(1,2,2)

#Killed
countk[:10].plot(kind="bar",color="blue",figsize=(15,6),ax=ax0)
ax0.set_title("People Killed in each country")
ax0.set_xlabel("Country")
ax0.set_ylabel("No of people killed")

#Wounded
countw[:10].plot(kind="bar",color="blue",figsize=(15,6),ax=ax1)
ax1.set_title("People wounded in each country")
ax1.set_xlabel("Country")
ax1.set_ylabel("No of people wounded")

plt.show()


# In[24]:


at=data["Attack Type"].value_counts()
at


# In[25]:


at.plot(kind="bar",color="blue",figsize=(15,6))
plt.title("Types of attacks",fontsize=12)
plt.xlabel("Attack Types",fontsize=12)
plt.xticks(fontsize=13)
plt.ylabel("Number of Attacks",fontsize=12)
plt.show()


# In[26]:


tt=data["Target Type"].value_counts()
tt


# In[27]:


tt.plot(kind="bar",color="blue",figsize=(15,6))
plt.title("Types of target",fontsize=12)
plt.xlabel("Target Types",fontsize=12)
plt.xticks(fontsize=13)
plt.ylabel("Number of Targets",fontsize=12)
plt.show()


# ### Thank you

# In[ ]:




