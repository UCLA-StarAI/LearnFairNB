#!/usr/bin/env python
# coding: utf-8

# In[1]:


import pandas as pd


# In[2]:


df = pd.read_csv('../data/adult.data')


# In[3]:


df


# In[4]:


get_ipython().run_line_magic('matplotlib', 'inline')
hist = df['age'].hist()
df['age'].mean()


# In[5]:


bins = [0, 40, 100]
groups = pd.cut(df['age'], bins=bins)
df['age'] = groups
num_groups = len(df['age'].cat.categories)
df['age'] = df['age'].cat.rename_categories(range(num_groups))
get_ipython().run_line_magic('matplotlib', 'inline')
hist = df['age'].hist()


# In[6]:


removed_columns = ['age']


# In[7]:


df['workclass'] = -1
mask = df[' workclass'].isin((' Self-emp-not-inc',' Self-emp-inc',' Without-pay',' Never-worked'))
df.loc[mask, 'workclass'] = 0
df.loc[~mask, 'workclass'] = 1


# In[8]:


hist = df['workclass'].hist()


# In[9]:


removed_columns.append(' workclass')


# In[10]:


get_ipython().run_line_magic('matplotlib', 'inline')
hist = df[' fnlwgt'].hist()
df[' fnlwgt'].mean()


# In[11]:


fnlwgt_bins = [0, 189778, 15000000]
fnlwgt_groups = pd.cut(df[' fnlwgt'], bins=fnlwgt_bins)
df['fnlwgt'] = fnlwgt_groups
num_groups = len(df['fnlwgt'].cat.categories)
df['fnlwgt'] = df['fnlwgt'].cat.rename_categories(range(num_groups))
get_ipython().run_line_magic('matplotlib', 'inline')
hist = df['fnlwgt'].hist()


# In[12]:


removed_columns.append(' fnlwgt')


# In[13]:


df['education'] = -1
mask = df[' education'].isin((' Bachelors',' Some-college',' Prof-school',' Assoc-acdm',' Assoc-voc', ' Masters', ' Doctorate'))
df.loc[mask, 'education'] = 1
df.loc[~mask, 'education'] = 0


# In[14]:


hist = df['education'].hist()


# In[15]:


removed_columns.append(' education')


# In[16]:


get_ipython().run_line_magic('matplotlib', 'inline')
hist = df[' education-num'].hist()
df[' education-num'].mean()


# In[17]:


bins = [0, 10, 20]
groups = pd.cut(df[' education-num'], bins=bins)
df['education-num'] = groups
num_groups = len(df['education-num'].cat.categories)
df['education-num'] = df['education-num'].cat.rename_categories(range(num_groups))
get_ipython().run_line_magic('matplotlib', 'inline')
hist = df['education-num'].hist()


# In[18]:


removed_columns.append(' education-num')


# In[19]:


bins = [0, 39, 1000]
groups = pd.cut(df[' hours-per-week'], bins=bins)
df['hours-per-week'] = groups
num_groups = len(df['hours-per-week'].cat.categories)
df['hours-per-week'] = df['hours-per-week'].cat.rename_categories(range(num_groups))
get_ipython().run_line_magic('matplotlib', 'inline')
hist = df['hours-per-week'].hist()


# In[20]:


removed_columns.append(' hours-per-week')


# In[21]:


df['marital-status'] = -1
mask = df[' marital-status'].isin((' Married-civ-spouse', ' Married-spouse-absent', ' Married-AF-spouse'))
df.loc[mask, 'marital-status'] = 0
df.loc[~mask, 'marital-status'] = 1


# In[22]:


hist = df['marital-status'].hist()


# In[23]:


removed_columns.append(' marital-status')


# In[24]:


df['occupation'] = -1
mask = df[' occupation'].isin((' Tech-support',' Sales',' Exec-managerial',' Prof-specialty',' Protective-serv',' Armed-Forces'))
df.loc[mask, 'occupation'] = 1
df.loc[~mask, 'occupation'] = 0


# In[25]:


hist = df['occupation'].hist()


# In[26]:


removed_columns.append(' occupation')


# In[27]:


df['relationship'] = -1
mask = df[' relationship'].isin((' Wife',' Husband'))
df.loc[mask, 'relationship'] = 1
df.loc[~mask, 'relationship'] = 0


# In[28]:


hist = df['relationship'].hist()


# In[29]:


removed_columns.append(' relationship')


# In[30]:


df['race'] = -1
mask = df[' race']==' White'
df.loc[mask, 'race'] = 1
df.loc[~mask, 'race'] = 0


# In[31]:


hist = df['race'].hist()


# In[32]:


removed_columns.append(' race')


# In[33]:


df['sex'] = -1
mask = df[' sex']==' Female'
df.loc[mask, 'sex'] = 1
df.loc[~mask, 'sex'] = 0


# In[34]:


hist = df['sex'].hist()


# In[35]:


removed_columns.append(' sex')


# In[36]:


df['native-country'] = -1
mask = df[' native-country']==' United-States'
df.loc[mask, 'native-country'] = 1
df.loc[~mask, 'native-country'] = 0


# In[37]:


hist = df['native-country'].hist()


# In[38]:


removed_columns.append(' native-country')


# In[39]:


get_ipython().run_line_magic('matplotlib', 'inline')
hist = df[' capital-gain'].hist()
df[' capital-gain'].mean()


# In[40]:


bins = [-1, 1, 200000]
groups = pd.cut(df[' capital-gain'], bins=bins)
df['capital-gain'] = groups
num_groups = len(df['capital-gain'].cat.categories)
df['capital-gain'] = df['capital-gain'].cat.rename_categories(range(num_groups))
get_ipython().run_line_magic('matplotlib', 'inline')
hist = df['capital-gain'].hist()


# In[41]:


removed_columns.append(' capital-gain')


# In[42]:


get_ipython().run_line_magic('matplotlib', 'inline')
hist = df[' capital-loss'].hist()
df[' capital-loss'].mean()


# In[43]:


bins = [-1, 1, 5000]
groups = pd.cut(df[' capital-loss'], bins=bins)
df['capital-loss'] = groups
num_groups = len(df['capital-loss'].cat.categories)
df['capital-loss'] = df['capital-loss'].cat.rename_categories(range(num_groups))
get_ipython().run_line_magic('matplotlib', 'inline')
hist = df['capital-loss'].hist()


# In[44]:


removed_columns.append(' capital-loss')


# In[45]:


df['target'] = -1
mask = df[' target']==' <=50K'
df.loc[mask, 'target'] = 0
df.loc[~mask, 'target'] = 1


# In[46]:


hist = df['target'].hist()


# In[47]:


removed_columns.append(' target')


# In[48]:


df.drop(removed_columns, axis=1, inplace=True)


# In[49]:


df.to_csv('../data/adult_binerized.csv', index=False)


# In[50]:




