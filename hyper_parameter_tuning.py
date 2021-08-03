#!/usr/bin/env python3
# -*- coding: utf-8 -*-

import numpy as np

import gc
from time import time
from skopt import dump, load
import joblib
from time import time
import numpy as np
import matplotlib.pyplot as plt
from sklearn.model_selection import GridSearchCV, RandomizedSearchCV
from sklearn.metrics import make_scorer, log_loss
import lightgbm as lgb
from skopt.utils import use_named_args
from skopt import gp_minimize
from skopt import callbacks
from skopt.callbacks import CheckpointSaver

def mask_dict(dictionary, mask):
        return_dict = {}
        for var in dictionary:
            return_dict[var] = dictionary[var][mask]
            
        return return_dict 

def logloss_eval(preds, train_data):
    logloss_value = log_loss(train_data.get_label(), preds, sample_weight = train_data.get_weight())

    return 'logloss_eval', logloss_value, False

def hyper_parameter_tuning_main(data, labels_train, labels_valid, vars, space, param_dir="./",):
    t_all = time()
    #print("labels_train, labels_valid:\n",labels_train, labels_valid)
    # Add special labels if needed
    # data[settings.train_string][settings.truth] = labels_train
    # data[settings.val_string]['Truth'] = labels_valid
    
    # data[settings.train_string]['logits'] = labels_train
    # data[settings.val_string]['Truth'] = labels_valid

    
    # Get train-sample mask

    valid_subsample = {"labels": np.array([])} # @Lau BC when use in while before defined
    train_subsample = {"labels": np.array([])} # @Lau BC when use in while before defined
    iter = 0
    while not valid_subsample['labels'].any() or not train_subsample['labels'].any(): # Make sure that at least one of the samples are truth @Lau
        iter +=1 
        print("iter:", iter)
        N = data['train']['labels'].shape[0]  # Number of samples

        idx_train            = np.random.choice(N, int(N * 0.1), replace=False)
        msk_train            = np.zeros((N,)).astype(bool)
        msk_train[idx_train] = True

        train_subsample      = mask_dict(data['train'], msk_train)
        N                    = data['val']['labels'].shape[0]  # Number of samples
        # Get valid-sample mask
        idx_valid            = np.random.choice(N, int(N * 0.1), replace=False)
        msk_valid            = np.zeros((N,)).astype(bool) # [False, False, False...]
        msk_valid[idx_valid] = True # make it a int(N * settings.HP_subsample_frac)(=n) long list

        valid_subsample      = mask_dict(data['val'], msk_valid) # Make all lists in data n long     
    del msk_train, msk_valid
    gc.collect()

    print(f"Tuning parameters ...\n")
    # print(f"valid_subsample[{'Truth'}]:", valid_subsample['Truth'])
    # print(f"train_subsample[{settings.truth}]:", train_subsample[settings.truth])

    t = time()

    results = hyper_parameter_tuning('gbdt', space, train_subsample, train_subsample['labels'],
                                    valid_subsample, valid_subsample['labels'],
                                    training_vars          = vars,
                                    obj                    = 'binary',
                                    eval_func              = logloss_eval,
                                    early_stopping_rounds  = 100,
                                    num_boost_round        = 10000)
                                    # train_weight           = train_subsample[settings.weight],
                                    # valid_weight           = valid_subsample['weight'],
                                    
    path = f'./results_gbdt.pkl'
    dump(results, path, store_objective=False)
    print("Dump path:", path)
    print(f"Time spent on Hyper Parameter tuning : {int((time() - t)/60):.2f} minutes.")
    print(f"Time spent on Hyper Parameter tuning: {int((time() - t_all)/60):.2f} minutes.")

def hyper_parameter_tuning(boosting, space, train_data, train_label,
                           valid_data, valid_label,
                           training_vars,
                           obj,
                           eval_func,
                           train_weight          = None,
                           valid_weight          = None,
                           early_stopping_rounds = 1000,
                           num_boost_round       = 10000):#,
                          # init_score = 0 ):

    if training_vars == [None]:
        train = lgb.Dataset(train_data,
                        label                    = train_label,
                        weight                   = train_weight,
                        free_raw_data            = False)#,
                        #init_score = init_score * np.ones(len(train_label)))
    elif 'p_eta_cat' in training_vars:
        train = lgb.Dataset(np.array([train_data[var] for var in training_vars]).T,
                        label                    = train_label,
                        weight                   = train_weight,
                        feature_name             = training_vars,
                        free_raw_data            = False
                        )# categorical_feature      = ['p_eta_cat', 'p_et_calo_cat', 'Nvtx_cat'],        
    else:
        train = lgb.Dataset(np.array([train_data[var] for var in training_vars]).T,
                        label                    = train_label,
                        weight                   = train_weight,
                        feature_name             = training_vars,
                        free_raw_data            = False)#,
    
    if training_vars == [None]:
        valid = train.create_valid(valid_data,
                           label = valid_label, weight = valid_weight)#, init_score = 0)
    else:
        valid = train.create_valid(np.array([valid_data[var] for var in training_vars]).T,
                           label = valid_label, weight = valid_weight)#, init_score = 0)

    @use_named_args(space)
    def objective(**params):
        results             = {}
        
        params['boosting']  = boosting
        params['n_jobs']    = 2
        
        if callable(obj):
            fobj                = obj
        else:
            params['objective'] = obj
            fobj                = None
        
        params['metrics']   = 'None'
        params['verbose']   = -1

        lgb.train(params                = params,                train_set      = train,
                  valid_sets            = [valid],               valid_names    = ['valid'],
                  evals_result          = results,               verbose_eval   = early_stopping_rounds,
                  early_stopping_rounds = early_stopping_rounds, fobj           = fobj,
                  num_boost_round       = num_boost_round,       feval          = eval_func)
        
        best_result         = min(results['valid'][[*results['valid']][0]])

        return best_result

    checkpoint_saver = CheckpointSaver("checkpoint.pkl", store_objective=False)
    results          = gp_minimize(objective,                   space, 
                                   n_calls=100,                 n_random_starts = 10, 
                                   callback=[checkpoint_saver], random_state=0, 
                                   verbose = True)
                                   #n_points = 10000 if settings.numDataPoints>10000 else int(settings.numDataPoints), # @Lau
                                   
    print(f'Best score: {results.fun}')
    print(f'Best Parameters: {results.x}')

    return results

    