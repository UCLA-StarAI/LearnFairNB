#!/usr/bin/env python
# coding: utf-8

# In[1]:


import pandas as pd
import numpy as np


# In[2]:


df = pd.read_csv('../data/adult.data', header=None, delimiter=r",\s+",)
df.columns = [
    "Age", "WorkClass", "fnlwgt", "Education", "EducationNum",
    "MaritalStatus", "Occupation", "Relationship", "Race", "Gender",
    "CapitalGain", "CapitalLoss", "HoursPerWeek", "NativeCountry", "Income"
]


df["Income"] = df["Income"].map({ "<=50K": 0 , ">50K": 1 })


df.Age = df.Age.astype(float)
df.fnlwgt = df.fnlwgt.astype(float)
df.EducationNum = df.EducationNum.astype(float)
df.HoursPerWeek = df.HoursPerWeek.astype(float)

df.drop("CapitalGain", axis=1, inplace=True,)
df.drop("CapitalLoss", axis=1, inplace=True,)

df.Age = pd.Categorical(df.Age)
df['Age'] = df.Age.cat.codes
df.HoursPerWeek = pd.Categorical(df.HoursPerWeek)
df['HoursPerWeek'] = df.HoursPerWeek.cat.codes
df.EducationNum = pd.Categorical(df.EducationNum)
df['EducationNum'] = df.EducationNum.cat.codes
df.Race = pd.Categorical(df.Race)
df['Race'] = df.Race.cat.codes
df.Gender = pd.Categorical(df.Gender)
df['Gender'] = df.Gender.cat.codes
df.Relationship = pd.Categorical(df.Relationship)
df['Relationship'] = df.Relationship.cat.codes
df.MaritalStatus = pd.Categorical(df.MaritalStatus)
df['MaritalStatus'] = df.MaritalStatus.cat.codes
df.Occupation = pd.Categorical(df.Occupation)
df['Occupation'] = df.Occupation.cat.codes
df.WorkClass = pd.Categorical(df.WorkClass)
df['WorkClass'] = df.WorkClass.cat.codes
df.Education = pd.Categorical(df.Education)
df['Education'] = df.Education.cat.codes
df.NativeCountry = pd.Categorical(df.NativeCountry)
df['NativeCountry'] = df.NativeCountry.cat.codes


# In[3]:


df.dropna()


# In[4]:


df['Age'].unique()


# In[5]:


bins = [0, 100000, 200000, 300000, 400000, 500000, 600000, 700000, 800000, 900000, 1000000, 2000000]
labels = [1,2,3,4,5,6,7,8,9,10,11]
df['fnlwgt'] = pd.cut(df['fnlwgt'], bins=bins, labels=labels, include_lowest=True)


# In[6]:


bins = [0, 10, 20, 30, 40, 50, 60, 70, 80, 90, 100]
labels = [1 , 2, 3, 4, 5, 6, 7 , 8, 9, 10]
df['Age'] = pd.cut(df['Age'], bins=bins, labels=labels, include_lowest=True)

df['HoursPerWeek'] = pd.cut(df['HoursPerWeek'], bins=bins, labels=labels, include_lowest=True)


# In[7]:


df['HoursPerWeek'].unique()


# In[8]:


df.to_csv('./adult_categorized_3.data')


# In[ ]:




