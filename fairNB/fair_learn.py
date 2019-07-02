from __future__ import print_function
import pandas as pd
import math
import numpy as np
import json, sys, os, inspect
import argparse
import gpkit
import time
import timeout_decorator
from gpkit import Variable, Model
from parameter_learner.data_processor import init_learning,convert_result_to_parameters,get_feature_names
from parameter_learner.maximum_likelihood_calculator import calculate_log_likelihood, maximum_likelihood, check_constraints, independence_process, check_validity
from pattern_finder.pattern_finder import PatternFinder

def initialize(db_file, info_file):
    root_params, leaf_params, target_value, target_name, sensitive_var_ids, feature_names, bn_dict, params_dict, df = init_learning(db_file,info_file)
    return  root_params, leaf_params, target_value, target_name, sensitive_var_ids, feature_names, bn_dict, params_dict, df


@timeout_decorator.timeout(1800, use_signals=False)
def _get_sorted_patterns(selector, delta, k, root_params, leaf_params, target_value, sensitive_var_ids, bn_dict, previous_patterns):
    #this function gets discrimination patterns from the pattern finder
    remaining_patterns_size = k
    pf = PatternFinder(root_params, leaf_params, target_value, sensitive_var_ids)
    if (selector=='KLD'):
        raw = pf.get_divergent_patterns(delta , remaining_patterns_size)#k)
        patterns, remaining_patterns = process_patterns(raw, bn_dict, k, selector, delta)
        num_node_visited = pf.num_visited
    elif (selector=='Diff'):
        raw = pf.get_discriminating_patterns(delta , remaining_patterns_size)#k)
        patterns, remaining_patterns = process_patterns(raw, bn_dict, k, selector, delta)
        num_node_visited = pf.num_visited
    return patterns, num_node_visited, remaining_patterns


def get_sorted_patterns(selector, delta, k, root_params, leaf_params, target_value, sensitive_var_ids, bn_dict, previous_patterns, output_file):
    try:
        patterns, num_node_visited, remaining_patterns = _get_sorted_patterns(selector, delta, k, root_params, leaf_params, target_value, sensitive_var_ids, bn_dict, previous_patterns)
    except:
        with open(output_file, 'a') as f:
            print('TIMEOUT: !!! extracting patterns took too long !!!', file=f)
        exit(1)
    return patterns, num_node_visited, remaining_patterns

def process_patterns(raw_patterns, bn_dict, k, selector, delta):
    patterns = []
    for pattern in raw_patterns:
        base = pattern.base
        for i in range(len(base)):
            feature = base[i]
            feature_name = bn_dict[feature[0]]
            base[i] = (feature_name, feature[1])
        sens = pattern.sens
        for i in range(len(sens)):  
            feature = sens[i]
            feature_name = bn_dict[feature[0]]
            sens[i] = (feature_name, feature[1])
        score = pattern.score
        if (selector == 'Diff' and score > delta) or (selector == 'KLD' and score > 0):
            if (len(pattern.sens)>0 or len(pattern.base)>0):
                patterns.append((sens, base, score))
    sorted_by_score = sorted(patterns, key=lambda tup: tup[2], reverse=True)
    return sorted_by_score, len(sorted_by_score)


def learn_parameters(output_file, selector, delta, k, patterns, num_node_visited, remaining_patterns, target_value, target_name, sensitive_var_ids, feature_names, bn_dict, params_dict, df):
    iteration = 1
    prob_dict = maximum_likelihood(delta, params_dict, feature_names, target_name)
    log_value = calculate_log_likelihood(prob_dict, df, feature_names, target_name)
    
    sensitive_features = get_feature_names(sensitive_var_ids, bn_dict)
    base_dict = independence_process(params_dict, sensitive_features, prob_dict)
    base_log = calculate_log_likelihood(base_dict, df, feature_names, target_name)
    
    with open(output_file, 'a') as the_file:
        the_file.write('baseline (independent):'+'\t'+str(base_log)+'\n')
            
    output_line= str(iteration-1)+'\t'+str(log_value)+'\n'
    with open(output_file, 'a') as the_file:
            the_file.write(output_line)
    fairness_patterns = []
    temp = 0
    while len(patterns)>0 and iteration<30:
        print("Start iteration %s"%iteration)
        degree = check_constraints(delta, patterns, prob_dict,target_name)
        fairness_patterns += patterns
        prob_dict = maximum_likelihood(delta, params_dict, feature_names, target_name, fairness_patterns)
        temp = calculate_log_likelihood(prob_dict, df, feature_names, target_name)
        if (abs(temp-log_value)>0):
            log_value = temp
            temp = 0
        validity = check_validity(prob_dict,  sensitive_features)
        output_line= str(iteration)+'\t'+str(log_value)+'\t'+str(validity)+'\t'+ str(num_node_visited)+'\t'+ str(remaining_patterns)+'\n'
        with open(output_file, 'a') as the_file:
            the_file.write(output_line)
        root_params, leaf_params  = convert_result_to_parameters(prob_dict,sensitive_var_ids,bn_dict,target_name)
        patterns, num_node_visited, remaining_patterns = get_sorted_patterns(selector, delta, k, 
                                                                             root_params, leaf_params, target_value, 
                                                                             sensitive_var_ids, bn_dict, fairness_patterns,
                                                                             output_file)
        iteration+=1
    
    with open(output_file, 'a') as f:
        if (len(patterns)>0):
            print('TIMEOUT: !!! Early termination !!!', file=f)
        else:
            print("Optimal results after %s iterations"%(iteration-1), file=f)
    return prob_dict

def run(args):
    selector = args.heuristic
    data_source = args.data
    delta = args.delta
    k = args.k
    outdir = args.outdir
    data_dir = args.data_dir

    log_file = os.path.join(outdir, 'runtime.%s.%s.%g.%d.txt'%(selector, data_source, delta, k))
    db_file = os.path.join(data_dir, '%s_binerized.csv'%data_source)
    info_file = os.path.join(data_dir, '%s_binary.net.txt'%data_source)
    root_params, leaf_params, target_value, target_name, sensitive_var_ids, feature_names, bn_dict, params_dict, df  = initialize(db_file, info_file)
    output_base = '%s/%s_%f'%(outdir,data_source,delta)
    start_time  = time.time()
    output_file = output_base+'result_%s_%d.txt'%(selector,k)
    print('datasource:'+ str(data_source)+', using '+str(selector)+' with k ='+str(k)+' and delta='+str(delta))
    sorted_patterns, num_node_visited, remaining_patterns = get_sorted_patterns(selector, delta, k, 
                                                                                root_params, leaf_params, target_value, 
                                                                                sensitive_var_ids, bn_dict, [],
                                                                                output_file)
    prob_dict = learn_parameters(output_file, selector, delta, k, sorted_patterns, num_node_visited, remaining_patterns, target_value, target_name, sensitive_var_ids, feature_names, bn_dict, params_dict, df)
    elapsed_time = time.time() - start_time 
    output_line= 'datasource:'+ str(data_source)+', using '+str(selector)+' with k ='+str(k)+' and delta='+str(delta)+' has running time='+str(elapsed_time)
    with open(log_file, 'a') as the_file:
        the_file.write(output_line+'\n')


if __name__ == '__main__':
    
    parser = argparse.ArgumentParser()
    parser.add_argument('data', type=str, choices=['compas', 'german', 'adult'],
                        help='name of the dataset')
    parser.add_argument('heuristic', type=str, choices=['KLD', 'Diff'],
                        help='pattern selection heuristic')
    parser.add_argument('delta', type=float, choices=[0.05, 0.01, 0.1, 0.5],
                        help='fairness threshold')
    parser.add_argument('k', type=int, choices=[1, 10, 100],
                        help='number of patterns')
    parser.add_argument('--outdir', type=str, default=None)
    parser.add_argument('--data_dir', type=str, default=None)
    args = parser.parse_args()

    cwdir = os.path.dirname(os.path.abspath(inspect.getfile(inspect.currentframe())))
    if not args.outdir: 
        args.outdir = cwdir
    if not args.data_dir:
        args.data_dir = os.path.abspath(os.path.join(cwdir, '..', 'data'))
    
    run(args)





