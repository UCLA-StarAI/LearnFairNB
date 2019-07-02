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


def get_feature_names(patterns):
    feature_names = set()
    sensitive_names = set()
    for f1, f2, _ in patterns:
        for sattr, _ in f1:
            sensitive_names.add(sattr)
        for f in (f1, f2):
            for attr, _ in f:
                feature_names.add(attr)
    feature_names = list(feature_names)
    sensitive_names = list(sensitive_names)
    return feature_names,sensitive_names


# In[3]:


def get_params_dict(df, feature_names, target_name):
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


def get_objective(params_dict):
    objective = 1.0
    for key, value in params_dict.items():
        variable = value[0]
        exponent = value[1]*-1.0
        objective *= variable**exponent
    return objective


# In[5]:


def get_parity_constraints(params_dict, feature_names, target_name):
    constraints = []
    for feature in feature_names:
        if (feature==target_name):
            var1 = (feature,0)
            variable1 = params_dict[var1][0]
            var2 = (feature,1)
            variable2 = params_dict[var2][0]
            constraints+=[variable1+variable2<=1]
        else:
            for v in [0,1]:
                var1 = (feature,0,v)
                variable1 = params_dict[var1][0]
                var2 = (feature,1,v)
                variable2 = params_dict[var2][0]
                constraints+=[variable1+variable2<=1]
    for key,value in params_dict.items():
        variable = value[0]
    return constraints


# In[6]:


def get_feasibility_parity_constraints(params_dict, s, feature_names, target_name):
    constraints = []
    for feature in feature_names:
        if (feature==target_name):
            var1 = (feature,0)
            variable1 = params_dict[var1][0]
            var2 = (feature,1)
            variable2 = params_dict[var2][0]
            constraints+=[variable1+variable2<=s]
        else:
            for v in [0,1]:
                var1 = (feature,0,v)
                variable1 = params_dict[var1][0]
                var2 = (feature,1,v)
                variable2 = params_dict[var2][0]
                constraints+=[variable1+variable2<=s]
    for key,value in params_dict.items():
        variable = value[0]
        constraints+=[variable>=0.00000000000000000000001]
        constraints+=[variable<=1]
    return constraints


# In[7]:


def get_pattern_constraints(X, Y, var_dict=None, D_=None, D=None, d_attr=None, c1=None, c2=None):
    def get_r(assignments):
        r = 1
        for attr, val in assignments:
            key = (attr, val, D_)
            r *= var_dict[key]
            key = (attr, val, D)
            r *= (1 / var_dict[key])
        return r

    rx = get_r(X)
    ry = var_dict[(d_attr, D_)] / var_dict[(d_attr, D)]
    ry *= get_r(Y)
    
    constraints = []
    with gpkit.SignomialsEnabled():
        constraints.append(c1 * rx * ry - c2 * ry - rx * (ry ** 2)<=1)
        constraints.append(-c2 * rx * ry  + c1 * ry - rx * (ry ** 2) <= 1)
    return constraints


# In[8]:


def get_fairness_constraints(delta, patterns, params_dict, target_name):
    var_dict = {k: v[0] for k,v in params_dict.items()}
    D_ = 1
    D = 0
    d_attr = target_name
    c1 = (1-delta)/delta
    c2 = (1+delta)/delta

    args = dict(var_dict=var_dict, D_=D_, D=D, d_attr=d_attr, c1=c1, c2=c2)
    fairness_constraints = []
    for X, Y, _ in patterns:
        fairness_constraints += get_pattern_constraints(X, Y, **args)
    return fairness_constraints


# In[9]:


def get_value_pattern_constraints(X, Y, var_dict=None, D_=None, D=None, d_attr=None, c1=None, c2=None):
    def get_r(assignments):
        r = 1
        for attr, val in assignments:
            key = (attr, val, D_)
            r *= var_dict[key]
            key = (attr, val, D)
            r *= (1 / var_dict[key])
        return r

    rx = get_r(X)
    ry = var_dict[(d_attr, D_)] / var_dict[(d_attr, D)]
    ry *= get_r(Y)
    
    result = 0
  
    if (c1 * rx * ry - c2 * ry - rx * (ry ** 2)<=1):
        result+=1
    if (-c2 * rx * ry  + c1 * ry - rx * (ry ** 2) <= 1):
        result+=1
    return result


# In[10]:


def check_constraints(delta, patterns, var_dict, target_name):
    D_ = 1
    D = 0
    d_attr = target_name
    c1 = (1-delta)/delta
    c2 = (1+delta)/delta
    args = dict(var_dict=var_dict, D_=D_, D=D, d_attr=d_attr, c1=c1, c2=c2)
    
    result = 0
    for X, Y, _  in patterns:
        result+=get_value_pattern_constraints(X, Y, **args)
    return float(result)/(len(patterns)*2)


# In[11]:


def maximum_likelihood(delta, params_dict, feature_names, target_name, patterns=None, result_dict=None):
        is_fair = (patterns is not None)
        is_initialize = (result_dict is not None)
        objective = get_objective(params_dict)
        constraints = get_parity_constraints(params_dict, feature_names, target_name)
        if is_fair:
            fair_constraints = get_fairness_constraints(delta, patterns, params_dict, target_name)
            constraints+=fair_constraints
        
        m = Model(objective, constraints)
        
        if is_fair:
            if is_initialize:
                #print("Local solver with initialization")
                sol = m.localsolve(solver='mosek_cli', x0=result_dict)
            else:
                sol = m.localsolve(solver='mosek_cli')
        else:
            sol = m.solve(solver='mosek_cli')
        
        #print("Optimal cost:  %.4g" % sol["cost"])
        #print(sol.summary())
        result_dict_2 = {}
        for key, value in params_dict.items():
            result_dict_2[key] = sol["variables"][value[0]]
        return result_dict_2


# In[12]:


def infeasibility_check(delta, params_dict, feature_names, target_name, patterns=None, result_dict=None):
        is_fair = (patterns is not None)
        is_initialize = (result_dict is not None)
        s = Variable(str("s"))
        objective = s
        s_constraint = [s>=1]
        
        constraints = get_feasibility_parity_constraints(params_dict, s, feature_names)
        constraints += s_constraint
        
        if is_fair:
            fair_constraints = get_fairness_constraints(delta, patterns, params_dict, target_name)
            #print(fair_constraints)
            constraints+=fair_constraints
        
        m = Model(objective, constraints)
        
        if is_fair:
            if is_initialize:
                #print("Local solver with initialization")
                sol = m.localsolve(solver='mosek_cli', x0=result_dict)
            else:
                sol = m.localsolve(solver='mosek_cli')
        else:
            sol = m.solve(solver='mosek_cli')
        
        #print("Optimal cost:  %.4g" % sol["cost"])
        #print(sol.summary())
        result_dict_2 = {}
        for key, value in params_dict.items():
            result_dict_2[key] = sol["variables"][value[0]]
        return result_dict_2


# In[13]:


def independence_process(params_dict, sentitive_names, result_dict):
    for sensitive_feature in sentitive_names:
        for v in [0,1]:
            key_0 = (sensitive_feature, v ,0)
            value_0 = result_dict[key_0]
            key_1 = (sensitive_feature, v ,1)
            value_1 = result_dict[key_1]
            independent_value = (value_1 + value_0)/2.0
            result_dict[key_0] = independent_value
            result_dict[key_1] = independent_value
    return result_dict


# In[ ]:


def check_validity(result_dict,  sentitive_names):
    valid = 0
    threshold  = 1e-6
    for sensitive_feature in sentitive_names:
        for v in [0,1]:
            key_0 = (sensitive_feature, 0, v)
            value_0 = result_dict[key_0]
            key_1 = (sensitive_feature, 1 ,v)
            value_1 = result_dict[key_1]
            validity_value = (value_1 + value_0)
            if (abs(validity_value -1 )<= threshold):
                valid+=1
    if (valid == (len(sentitive_names)*2)):
        return 1
    else:
        return 0


# In[14]:


def calculate_log_likelihood(result_dict, df, feature_names, target_name):
    log_value = 0.0
    df_selected = df.filter(items=feature_names)  
    for i, row in df_selected.iterrows():
        v2 = row[target_name]
        log_value += math.log(result_dict[(target_name, v2)])
        for j, column in row.iteritems():
            if (j!=target_name):
                feature_name = (j, column, v2)
                log_value += math.log(result_dict[feature_name])
    return log_value


# In[ ]:




