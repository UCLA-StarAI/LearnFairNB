#!/usr/bin/env python
# coding: utf-8

# In[1]:


import pandas as pd
import numpy as np


# In[2]:



df = pd.read_csv('../data/german.data',delimiter=r"\s+")
df.columns = [
    "Attribute1", "Attribute2", "Attribute3", "Attribute4", "Attribute5",
    "Attribute6", "Attribute7", "Attribute8", "Attribute9", "Attribute10",
    "Attribute11", "Attribute12", "Attribute13", "Attribute14", "Attribute15",
    "Attribute16", "Attribute17", "Attribute18", "Attribute19", "Attribute20",
    "Decision"
]


# In[3]:


df.Attribute1 = pd.Categorical(df.Attribute1)
df['Attribute1'] = df.Attribute1.cat.codes

df.Attribute3 = pd.Categorical(df.Attribute3)
df['Attribute3'] = df.Attribute3.cat.codes


df.Attribute4 = pd.Categorical(df.Attribute4)
df['Attribute4'] = df.Attribute4.cat.codes

df.Attribute6 = pd.Categorical(df.Attribute6)
df['Attribute6'] = df.Attribute6.cat.codes

df.Attribute7 = pd.Categorical(df.Attribute7)
df['Attribute7'] = df.Attribute7.cat.codes

df.Attribute9 = pd.Categorical(df.Attribute9)
df['Attribute9'] = df.Attribute9.cat.codes

df.Attribute10 = pd.Categorical(df.Attribute10)
df['Attribute10'] = df.Attribute10.cat.codes

df.Attribute12 = pd.Categorical(df.Attribute12)
df['Attribute12'] = df.Attribute12.cat.codes

df.Attribute14 = pd.Categorical(df.Attribute14)
df['Attribute14'] = df.Attribute14.cat.codes

df.Attribute15 = pd.Categorical(df.Attribute15)
df['Attribute15'] = df.Attribute15.cat.codes

df.Attribute17 = pd.Categorical(df.Attribute17)
df['Attribute17'] = df.Attribute17.cat.codes

df.Attribute19 = pd.Categorical(df.Attribute19)
df['Attribute19'] = df.Attribute19.cat.codes

df.Attribute20 = pd.Categorical(df.Attribute20)
df['Attribute20'] = df.Attribute20.cat.codes

df.Attribute8 = pd.Categorical(df.Attribute8)
df['Attribute8'] = df.Attribute8.cat.codes

df.Attribute11 = pd.Categorical(df.Attribute11)
df['Attribute11'] = df.Attribute11.cat.codes

df.Attribute16 = pd.Categorical(df.Attribute16)
df['Attribute16'] = df.Attribute16.cat.codes

df.Attribute18 = pd.Categorical(df.Attribute18)
df['Attribute18'] = df.Attribute18.cat.codes


df["Decisione"] = df["Decision"].map({ "1": 0 , "2": 1 })


# In[4]:


bins = [0, 10, 20, 30, 40, 50, 60, 70, 80, 90, 100]
labels = [1,2,3,4,5,6,7,8,9,10]
df['Attribute13'] = pd.cut(df['Attribute13'], bins=bins, labels=labels, include_lowest=True)
df['Attribute2'] = pd.cut(df['Attribute2'], bins=bins, labels=labels,  include_lowest=True)

bins = [0, 1000, 2000, 3000, 4000, 5000, 6000, 7000, 8000, 9000, 10000, 11000, 12000, 13000, 14000, 15000, 16000, 17000, 18000, 19000]
labels = [1,2,3,4,5,6,7,8,9,10, 11,12,13,14,15,16,17,18,19]
df['Attribute5'] = pd.cut(df['Attribute5'], bins=bins, labels=labels, include_lowest=True)


# In[5]:


df


# In[6]:


df.to_csv('./german_categorized.data')


# In[ ]:




