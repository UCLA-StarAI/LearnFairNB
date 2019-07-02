#!/usr/bin/env python
# coding: utf-8

# In[1]:


import pandas as pd


# In[2]:


df = pd.read_csv('../data/german.data')


# In[3]:


df.columns


# In[4]:


df['checking_status'].value_counts()


# In[5]:


df['checking_status_'] = -1
mask = df['checking_status'].isin(('A13', 'A12'))
df.loc[mask, 'checking_status_'] = 1
df.loc[~mask, 'checking_status_'] = 0


# In[6]:


df['checking_status_'].value_counts()


# In[7]:


duration_bins = [0, 12, 100]
duration_groups = pd.cut(df['duration'], bins=duration_bins)
df['duration'] = duration_groups
num_groups = len(df['duration'].cat.categories)
df['duration'] = df['duration'].cat.rename_categories(range(num_groups))


# In[8]:


df['duration'].value_counts()


# In[9]:


df['credit_history'].value_counts()


# In[10]:


df['credit_history_'] = -1
mask = df['credit_history'].isin(('A32', 'A31','A30'))
df.loc[mask, 'credit_history_'] = 0
df.loc[~mask, 'credit_history_'] = 1


# In[11]:


df['credit_history_'].value_counts()


# In[12]:


df['purpose'].value_counts()


# In[13]:


df['purpose_'] = -1
mask = df['purpose'].isin(('A45', 'A49','A40', 'A41', 'A46'))
df.loc[mask, 'purpose_'] = 1
df.loc[~mask, 'purpose_'] = 0


# In[14]:


df['purpose_'].value_counts()


# In[15]:


duration_bins = [0, 2800, 200000]
duration_groups = pd.cut(df['credit_amount'], bins=duration_bins)
df['credit_amount'] = duration_groups
num_groups = len(df['credit_amount'].cat.categories)
df['credit_amount'] = df['credit_amount'].cat.rename_categories(range(num_groups))


# In[16]:


df['credit_amount'].value_counts()


# In[17]:


df['saving_account'].value_counts()


# In[18]:


df['saving_account_'] = -1
mask = df['saving_account'].isin(('A63', 'A64'))
df.loc[mask, 'saving_account_'] = 1
df.loc[~mask, 'saving_account_'] = 0


# In[19]:


df['saving_account_'].value_counts()


# In[20]:


df['employment'].value_counts()


# In[21]:


df['employment_'] = -1
mask = df['employment'].isin(('A74', 'A75'))
df.loc[mask, 'employment_'] = 1
df.loc[~mask, 'employment_'] = 0


# In[22]:


df['employment_'].value_counts()


# In[23]:


df['income'].value_counts()


# In[24]:


df['income_'] = -1
mask = df['income'].isin(('3', '4'))
df.loc[mask, 'income_'] = 1
df.loc[~mask, 'income_'] = 0


# In[25]:


df['income_'].value_counts()


# In[26]:


df['sex_status'].value_counts()


# In[27]:


df['sex'] = -1
mask = df['sex_status'].isin(('A92', 'A95'))
df.loc[mask, 'sex'] = 1
df.loc[~mask, 'sex'] = 0


# In[28]:


df['sex'].value_counts()


# In[29]:


df['marital_status'] = -1
mask = df['sex_status'].isin(('A93', 'A95'))
df.loc[mask, 'marital_status'] = 1
df.loc[~mask, 'marital_status'] = 0


# In[30]:


df['marital_status'].value_counts()


# In[31]:


df['guarantor'].value_counts()


# In[32]:


df['guarantor_'] = -1
mask = df['guarantor'].isin(('A101', 'A102'))
df.loc[mask, 'guarantor_'] = 0
df.loc[~mask, 'guarantor_'] = 1


# In[33]:


df['guarantor_'].value_counts()


# In[34]:


df['residence'].value_counts()


# In[35]:


df['residence_'] = -1
mask = df['residence'].isin(('3', '4'))
df.loc[mask, 'residence_'] = 1
df.loc[~mask, 'residence_'] = 0


# In[36]:


df['residence_'].value_counts()


# In[37]:


df['property'].value_counts()


# In[38]:


df['property_'] = -1
mask = df['property'].isin(('A123', 'A124'))
df.loc[mask, 'property_'] = 0
df.loc[~mask, 'property_'] = 1


# In[39]:


df['property_'].value_counts()


# In[40]:


duration_bins = [0, 30, 100]
duration_groups = pd.cut(df['age'], bins=duration_bins)
df['age'] = duration_groups
num_groups = len(df['age'].cat.categories)
df['age'] = df['age'].cat.rename_categories(range(num_groups))


# In[41]:


df['age'].value_counts()


# In[42]:


df['installation_plan'].value_counts()


# In[43]:


df['installation_plan_'] = -1
mask = df['installation_plan'].isin(('A141', 'A142'))
df.loc[mask, 'installation_plan_'] = 0
df.loc[~mask, 'installation_plan_'] = 1


# In[44]:


df['installation_plan_'].value_counts()


# In[45]:


df['housing'].value_counts()


# In[46]:


df['housing_'] = -1
mask = df['housing'].isin(('A151', 'A153'))
df.loc[mask, 'housing_'] = 0
df.loc[~mask, 'housing_'] = 1


# In[47]:


df['housing_'].value_counts()


# In[48]:


df['existing_credit'].value_counts()


# In[49]:


df['existing_credit_'] = -1
mask = df['existing_credit'].isin(('2', '3','4'))
df.loc[mask, 'existing_credit_'] = 1
df.loc[~mask, 'existing_credit_'] = 0


# In[50]:


df['existing_credit_'].value_counts()


# In[51]:


df['job'].value_counts()


# In[52]:


df['job_'] = -1
mask = df['job'].isin(('A173', 'A174'))
df.loc[mask, 'job_'] = 1
df.loc[~mask, 'job_'] = 0


# In[53]:


df['job_'].value_counts()


# In[54]:


df['maintenance'].value_counts()


# In[55]:


df['maintenance'][df['maintenance'] == 1]=1
df['maintenance'][df['maintenance'] == 2]=0


# In[56]:


df['maintenance'].value_counts()


# In[57]:


df['telephone'].value_counts()


# In[58]:


df['telephone'][df['telephone'] == 'A191']=0
df['telephone'][df['telephone'] == 'A192']=1


# In[59]:


df['telephone'].value_counts()


# In[60]:


df['foreign_worker'].value_counts()


# In[61]:


df['foreign_worker'][df['foreign_worker'] == 'A201']=1
df['foreign_worker'][df['foreign_worker'] == 'A202']=0


# In[62]:


df['foreign_worker'].value_counts()


# In[63]:


df['loan_status'].value_counts()


# In[64]:


df['loan_status'][df['loan_status'] == 1]=1
df['loan_status'][df['loan_status'] == 2]=0


# In[65]:


df['loan_status'].value_counts()


# In[66]:


df.columns


# In[67]:


df.drop(['checking_status', 'credit_history', 'purpose', 'saving_account','employment','income','sex_status'], axis=1, inplace=True)
df.drop(['guarantor', 'residence', 'property', 'installation_plan','housing','existing_credit','job'], axis=1, inplace=True)


# In[68]:


df.columns


# In[69]:


df.drop(['duration','residence_', 'property_', 'installation_plan_','telephone', 'property_'], axis=1, inplace=True)


# In[70]:


df.to_csv('../data/german_binerized.csv', index=False)






