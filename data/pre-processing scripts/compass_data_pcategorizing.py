#!/usr/bin/env python
# coding: utf-8

# In[155]:


import pandas as pd
import numpy as np


# In[156]:


df = pd.read_csv('../data/compas.data', 
                 parse_dates = ['DateOfBirth'])


# In[157]:


removed_columns = [
    'Person_ID',
    'AssessmentID',
    'Case_ID',
    'LastName',
    'FirstName',
    'MiddleName',
    'Screening_Date',
    'RecSupervisionLevelText',
    'RawScore',
    'DecileScore',
    'IsCompleted',
    'IsDeleted'
]
df.drop(removed_columns, axis=1, inplace=True)


# In[158]:


import datetime
import numpy as np
age = (datetime.datetime.now() - df.DateOfBirth).astype('timedelta64[Y]')
age = age.astype('int')
age[age<0] = np.nan
df['age_'] = age


# In[159]:


df.drop(df[df['ScoreText'].isnull()].index, inplace=True)
df.drop(df[df['age_'].isnull()].index, inplace=True)
df.drop(df[df['MaritalStatus']=='Unknown'].index, inplace=True)


# In[160]:


age_bins = [0, 30, 100]
age_groups = pd.cut(df['age_'], bins=age_bins)
df['Age'] = age_groups
num_groups = len(df['Age'].cat.categories)
df['Age'] = df['Age'].cat.rename_categories(range(num_groups))


# In[161]:


df.Sex_Code_Text = pd.Categorical(df.Sex_Code_Text)
df['Sex_Code_Text'] = df.Sex_Code_Text.cat.codes

df.Ethnic_Code_Text = pd.Categorical(df.Ethnic_Code_Text)
df['Ethnic_Code_Text'] = df.Ethnic_Code_Text.cat.codes

df.MaritalStatus = pd.Categorical(df.MaritalStatus)
df['MaritalStatus'] = df.MaritalStatus.cat.codes

df.CustodyStatus = pd.Categorical(df.CustodyStatus)
df['CustodyStatus'] = df.CustodyStatus.cat.codes

df.LegalStatus = pd.Categorical(df.LegalStatus)
df['LegalStatus'] = df.LegalStatus.cat.codes

df['ScoreText_'] = -1
mask = df['ScoreText']=='High'
df.loc[mask, 'ScoreText_'] = 0
df.loc[~mask, 'ScoreText_'] = 1


# In[162]:


df.drop(['DisplayText','ScoreText','Agency_Text', 'AssessmentType', 'ScaleSet_ID', 'ScaleSet', 'AssessmentReason', 'Language'], axis=1, inplace=True)
df.drop(['DateOfBirth', 'age_', 'Scale_ID'], axis=1, inplace=True)


# In[163]:


df


# In[164]:


df.to_csv('./compass_categorized.data')


# In[ ]:




