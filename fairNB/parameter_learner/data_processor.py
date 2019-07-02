#!/usr/bin/env python
# coding: utf-8

# In[1]:


import pandas as pd
import math
import numpy as np
import json
import gpkit
from gpkit import Variable, Model


# In[2]:


def extract_parameters(info_file):
    bn_dict = {}
    sensitive_var_ids = []
    target_value = 0
    feature_names = []
    with open(info_file) as f:
        lines = f.readlines()
    i=1
    while i<len(lines):
        info = lines[i].replace('\n','').split(' ')
        if len(info)>1:
            feature_name = unicode(info[0], "utf-8")
            feature_label = info[1]
            feature_names.append(feature_name)
            #target (root) feature
            if i==1:
                target_value= int(feature_label)
                target_name = feature_name
            else:    
                bn_dict[i-2] = feature_name
                if feature_label=='1':
                    sensitive_var_ids.append(i-2)
        i+=1
    return bn_dict,sensitive_var_ids,target_value,target_name, feature_names


# In[3]:


def get_params_dict(df, feature_names, target_name):
    feature_names.remove(target_name)
    columns_dict = {}
    for c in df.columns:
        if c.endswith('_'):
            v = c[:-1]
        else:
            v = c
        columns_dict[c] = v
    df.rename(columns=columns_dict, inplace = True)  

    params_dict = {}
    # Decision variable
    for value in [0,1]:
        variable_name = (target_name, value)
        params_dict[variable_name] = (Variable(str(variable_name)),0.0)
    for feature in feature_names:
        for v1 in [0,1]:
            for v2 in [0,1]:
                variable_name = (feature, v1, v2)
                params_dict[variable_name] = (Variable(str(variable_name)),0.0)
    feature_names.append(target_name)
    df_selected = df.filter(items=feature_names)  
    for i, row in df_selected.iterrows():
        v2 = row[target_name]
        variable_name = (target_name, v2)
        expo_info = params_dict[variable_name]
        params_dict[variable_name] =  (expo_info[0], expo_info[1]+1.0)
        for j, column in row.iteritems():
            if (j!=target_name):
                variable_name = (j, column, v2)
                expo_info = params_dict[variable_name]
                params_dict[variable_name] =  (expo_info[0], expo_info[1]+1.0)
    return params_dict


# In[4]:


def maximum_likelihood_from_data(params_dict, target_name):
    #This function is calculating the probabilities that is equivalent to the results of the max-likelihood
    prob_dict = {}
    score_1 = params_dict[(target_name, 1)][1]
    score_0 = params_dict[(target_name, 0)][1]
    sum_scores = score_0 + score_1
    prob_dict[(target_name, 0)] = float(score_1)/float(sum_scores)
    prob_dict[(target_name, 1)] = float(score_0)/float(sum_scores)
    for key,value in params_dict.items():
        if key[0]!=target_name:
            v_score = key[2]
            if v_score==0:
                prob_dict[key] = float(value[1])/float(score_0)
            elif v_score==1:
                prob_dict[key] = float(value[1])/float(score_1)
    return prob_dict


# In[5]:


def convert_result_to_parameters(prob_dict,sensitive_var_ids,bn_dict, target_name):
    root_params = [prob_dict[(target_name, 0)], prob_dict[(target_name, 1)]]
    leaf_params = []
    for i in range(len(bn_dict.keys())):
        leaf_params.append([])
    for key,value in bn_dict.items():
        score = []
        for i in range(2):
            for j in range(2):
                score_value = prob_dict[(value,j,i)]
                score.append(score_value)
        leaf_params[key] = score    
    return root_params, leaf_params


# In[ ]:


def get_feature_names(feature_ids, bn_dict):
    feature_names = []
    for feature_id in feature_ids:
        feature_names.append(bn_dict[feature_id])
    return feature_names


# In[6]:


def init_learning(db_file,info_file):
    df = pd.read_csv(db_file)
    bn_dict,sensitive_var_ids,target_value,target_name,feature_names = extract_parameters(info_file)

    params_dict = get_params_dict(df, feature_names,target_name)
    prob_dict = maximum_likelihood_from_data(params_dict,target_name)
    
    root_params, leaf_params = convert_result_to_parameters(prob_dict,sensitive_var_ids,bn_dict,target_name)
    return root_params, leaf_params, target_value, target_name, sensitive_var_ids, feature_names, bn_dict, params_dict, df


# In[7]:


def test():
    db_file = 'data/compas_binerized.csv'
    info_file = 'data/compass_binary.net.txt'
    init_learning(db_file,info_file)
#test()


# In[ ]:




