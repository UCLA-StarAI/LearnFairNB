
# coding: utf-8

# In[1]:

import pandas as pd


# In[2]:

df = pd.read_csv('../data/compas.data', 
                 parse_dates = ['DateOfBirth'])


# Drop these case-specific columns:
# - Person_ID
# - AssessmentID
# - Case_ID
# - LastName
# - FirstName
# - MiddleName
# 
# Drop these columns:
# - `Screening_Date`: We don't know how to categorize this
# - `RecSupervisionLevelText`: same as `RecSupervisionLevel`
# - `RawScore`: many values. seems to be used for computation of target value
# - `DecileScore`: seems to be used for computation of target value
# - `IsCompleted`: same value for everyone
# - `IsDeleted`: same value for everyone

# In[3]:

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


# In[4]:

import datetime
import numpy as np
age = (datetime.datetime.now() - df.DateOfBirth).astype('timedelta64[Y]')
age = age.astype('int')
age[age<0] = np.nan
df['age_'] = age


# In[5]:

# dealing with missing values
df.drop(df[df['ScoreText'].isnull()].index, inplace=True)
df.drop(df[df['age_'].isnull()].index, inplace=True)
df.drop(df[df['MaritalStatus']=='Unknown'].index, inplace=True)


# In[6]:

get_ipython().magic('matplotlib inline')
hist = df['age_'].hist()
df['age_'].mean()


# In[7]:

age_bins = [0, 22, 100]
age_groups = pd.cut(df['age_'], bins=age_bins)


# In[8]:

df.columns


# In[9]:

df.groupby('Ethnic_Code_Text').count()


# In[10]:

df['CustodyStatus_'] = -1
mask = df['CustodyStatus']=='Jail Inmate'
df.loc[mask, 'CustodyStatus_'] = 0
df.loc[~mask, 'CustodyStatus_'] = 1


# In[11]:

df['LegalStatus_'] = -1
mask = df['LegalStatus']=='Post Sentence'
df.loc[mask, 'LegalStatus_'] = 0
df.loc[~mask, 'LegalStatus_'] = 1


# In[12]:

df['RecSupervisionLevel_'] = -1
mask = df['RecSupervisionLevel'].isin((1, 2))
df.loc[mask, 'RecSupervisionLevel_'] = 0
df.loc[~mask, 'RecSupervisionLevel_'] = 1


# In[13]:

df['Ethnic_Code_Text_'] = -1
mask = df['Ethnic_Code_Text']=='Caucasian'
df.loc[mask, 'Ethnic_Code_Text_'] = 0
df.loc[~mask, 'Ethnic_Code_Text_'] = 1


# In[14]:

df['MaritalStatus_'] = -1
mask = df['MaritalStatus'].isin(('Married', 'Significant Other'))
df.loc[mask, 'MaritalStatus_'] = 0
df.loc[~mask, 'MaritalStatus_'] = 1


# In[15]:

# https://www.bop.gov/about/statistics/statistics_inmate_age.jsp

age_bins = [0, 30, 100]
age_groups = pd.cut(df['age_'], bins=age_bins)
df['Age'] = age_groups
num_groups = len(df['Age'].cat.categories)
df['Age'] = df['Age'].cat.rename_categories(range(num_groups))
get_ipython().magic('matplotlib inline')
hist = df['Age'].hist()


# In[16]:

df['ScoreText_'] = -1
mask = df['ScoreText']=='High'
df.loc[mask, 'ScoreText_'] = 0
df.loc[~mask, 'ScoreText_'] = 1


# In[17]:

df['Sex_'] = -1
mask = df['Sex_Code_Text']=='Male'
df.loc[mask, 'Sex_'] = 0
df.loc[~mask, 'Sex_'] = 1


# In[18]:


df.drop(['DisplayText', 'Sex_Code_Text', 'ScoreText','Agency_Text', 'AssessmentType', 'ScaleSet_ID', 'ScaleSet', 'AssessmentReason', 'Language'], axis=1, inplace=True)
df.drop(['DateOfBirth', 'age_', 'MaritalStatus', 'Ethnic_Code_Text', 'RecSupervisionLevel','CustodyStatus','LegalStatus', 'Scale_ID'], axis=1, inplace=True)



# In[20]:

df.to_csv('compas_binerized.csv', index=False)


# In[ ]:



