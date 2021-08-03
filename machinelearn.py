import lightgbm as lgb
import numpy as np
import joblib
from skopt.space import Real, Integer
from hyper_parameter_tuning import hyper_parameter_tuning_main
from sklearn.metrics import make_scorer, log_loss
from skopt import gp_minimize, dump, load

from datetime import datetime
# datetime object containing current date and time
now = datetime.now()
# dd/mm/YY H:M:S
dt_string = now.strftime("%Y-%m-%d-%H-%M-%S")

# Data dict:
"""{"guesses_min": guesses_min, "guesses_max": guesses_max, "sellingpoint": sellingpoint, 
            "price_dif_data": price_dif_data, "price_diff":price_diff, "accountHolding": accountHolding,
            "accountDevellopment": accountDevellopment, "price_dif_data_pct": price_dif_data_pct,
            "slopeData": slopeData, "slope5_data": slope5_data, "slope30_data": slope30_data, "decc_coeff": decc_coeff, 
            "p1_data":p1_data,"p2_data":p2_data,"p3_data":p3_data,"p4_data":p4_data,"p5_data":p5_data,"p6_data":p6_data,"p7_data":p7_data,
            "labels": labels,  'correct_guesses': correct_guesses, 'correct_guess_freq': correct_guess_freq, 
            }"""

def create_data(varsList):
    data = joblib.load('closingDataDict|2015-01-01|2020-04-25', mmap_mode='r')

    comps = list(data.keys())
    # print("comps", comps)
    vars = data[comps[0]].keys()
    # print(vars)

    print(data["PENN"].keys())

    npoints = len(data["AAPL"]["labels"])
    # print("\nnpoints:", npoints)
    # print("\nlen(comps):", len(comps))
    numDict = {"train": slice(0, int(np.floor(0.80*npoints))), "val": slice(int(np.floor(0.80*npoints)), int(np.floor(0.90*npoints))), "test": slice(int(np.floor(0.90*npoints)), npoints)}
    dataDict = {}
    
    for Dset in ["train", "val","test"]:
        dataDict[Dset] = {}
        for biz in comps:
            for var in varsList:
                # Make dict not empty before concatenating
                if var not in dataDict[Dset].keys():    dataDict[Dset][var] = np.array([])
                
                # if var == "labels":
                #     print("labels", "len(np.array(data[biz][var])):", len(np.array(data[biz][var])))
                # else: print(var, "len(np.array(data[biz][var])):",len(np.array(data[biz][var])))
                data[biz][var] = np.array(data[biz][var])
                # if var == "labels": 
                #     # Error: size of labels is 1334, but the others are 1338
                #     print(var, "data[biz][var].shape:", data[biz][var].shape)
                #     # data[biz][var] = np.array(data[biz][var][0])
                #     # print(var, "data[biz][var].shape:", data[biz][var].shape)
                #     data[biz][var] = data[biz][var][:]

                dataDict[Dset][var] = np.concatenate([dataDict[Dset][var], data[biz][var][numDict[Dset]]])
    
    return dataDict

def create_lgb_dataset(data, mlSet):
    labels = data[mlSet]["labels"]
    # print("data[mlSet].items():", data[mlSet].items())
    vars_array = np.zeros([len(data[mlSet].keys()), len(data[mlSet]["labels"])])
    print( f"len(labels), len(data[{mlSet}][p1]):", len(labels), len(data[mlSet]["labels"]))
    
    k = 0
    for var in sorted(data[mlSet]): # Sort the values to put them in the right order when predicting
        if var == "labels": continue
        print("k, var:",k, var)
        vars_array[k] = np.array(data[mlSet][var])
        k += 1

    # vars_array = np.array([[np.array(item).T for name, item in data[mlSet].items() if "label" not in item]])
    # vars_array.reshape(len(data[mlSet]["labels"]), len(data[mlSet].keys()))    
    # vars_array = np.array([data[mlSet]["labels"], data[mlSet]["labels"]]).T
    
    # print("vars_array:", vars_array[0])
    print( mlSet, "vars_array.shape:", vars_array.shape)
    
    return lgb.Dataset(vars_array.T, label=labels)

def logloss_eval(preds, train_data):
    logloss_value = log_loss(train_data.get_label(), preds, sample_weight = train_data.get_weight())

    return 'logloss_eval', logloss_value, False

def machLearn(vars):
    mlData = create_data(vars)
    # for dset in mlData.keys():
        # print(dset, "dset")
        # for var in mlData[dset].keys():
            # print(f"{var}, len({var})", var, len(mlData[dset][var]))
    lgbData_train = create_lgb_dataset(mlData, "train")
    lgbData_valid = create_lgb_dataset(mlData, "val")
    
    predict_data_test = np.array([np.array(item).T  for name, item in mlData["test"].items() if "label" not in item])
    predict_labels = mlData["test"]["labels"]
    
    ###### Hyper parameter tuning:
    
    boosting_types  = ['gbdt', 'rf']
    early_stopping_rounds = 100
    objective = 'binary'
    eval_func = logloss_eval
    init_score = 0


    HP_subsample_frac = 0.1

    space = [
              Real(  0.01,    0.1,                name='learning_rate'   ),
            #   Real(10**-4, 10**-1, "log-uniform", name='min_child_weight'),
            #   Real(   0.0,    0.4,                name='min_split_gain'  ),
            #   Real( 0.001, 0.9999,                name='bagging_fraction'),
            #   Integer(  1,     10,                name='bagging_freq'    ),
            #   Real(   0.1,    0.9,                name='feature_fraction'),
              Integer( 30,    100,                name='num_leaves'      )
              ]   
    # hyper_parameter_tuning_main(data, labels_train,              labels_valid,            vars, space, param_dir="./",)
    # print("mlData['train']['labels'][:100]:", mlData['train']['labels'][:100])
    # print("mlData['val']['labels'][:100]:", mlData['val']['labels'][:100])
    if 0:
        hyper_parameter_tuning_main(mlData, mlData['train']['labels'], mlData['val']['labels'], vars, space)
    
    ######
    # print("mlDataTrain.shape:",lgbData_train.shape)
    
    
    param = {'num_leaves': 50, 'objective': 'binary', 'learning_rate': 0.1}
    param['metric'] = ['binary_logloss']

    num_round = 100
    res = load("results_gbdt.pkl")

    # names = [space[var].name for var in range(len(space))]
    # param = {names[i]:res.x[i] for i in range(len(names))}
    # param['boosting'] = 'gbdt'
    # param['metric'] = ['auc', 'binary_logloss']
    
    # test_data  = np.array([np.array(mlData["train"]["p2_data"][:]), np.array(mlData["train"]["p1_data"][:])]).T
    # test_labels = mlData["train"]["labels"][:]
    # print("\ntest_labels.shape, test_data.shape:", test_labels.shape, test_data.shape )
    # print("\ntest_data:",test_data[:10])
    # test_val_data   = np.array([np.array(mlData["val"]["p2_data"][:]), np.array(mlData["val"]["p1_data"][:])]).T
    # test_val_labels = mlData["val"]["labels"][:]
    # print("\ntest_val_data[:10]:",test_val_data[:10])
    # print("\ntest_val_labels.shape, test_val_data.shape:", test_val_labels.shape, test_val_data.shape )

    # model_test = lgb.train(param, lgb.Dataset(test_data, label=test_labels), num_round, valid_sets=lgb.Dataset(test_val_data, label=test_val_labels), early_stopping_rounds=100)
    model = lgb.train(param, lgbData_train, num_round, valid_sets=lgbData_valid, early_stopping_rounds=200)
    if False:
        model.save_model("lgb_model_50leaves.txt")
    # ypred_test = model_test.predict(predict_data_test, num_iteration=model_test.best_iteration)
    ypred = model.predict(predict_data_test.T, num_iteration=model.best_iteration)
    # print("ypred:", ypred)
    pred_truth = ypred>0.71
    # print("len(pred_truth[pred_truth]):", 
        #   len(pred_truth[pred_truth]))
    # print("sum(np.logical_and(pred_truth[pred_truth], predict_labels[pred_truth]))/float(len(pred_truth))", 
    #        sum(np.logical_and(pred_truth[pred_truth], predict_labels[pred_truth]))/float(len(pred_truth[pred_truth])))
    print("sum(predict_labels[pred_truth])/float(len(pred_truth))", 
           sum(predict_labels[pred_truth])/float(len(predict_labels[pred_truth])))
    # print("predict_labels[pred_truth]):", predict_labels[pred_truth], sum(predict_labels[pred_truth])/len( predict_labels[pred_truth]))
    # print("sum(np.logical_and(ypred>0.5, predict_labels))/float(len(predict_labels)):", sum(np.logical_and(ypred>0.6, predict_labels))/float(len(predict_labels)))
    # print("ypred[:10]:",     ypred[:10])

    # PRint the guesses algorithms correctness
    guesses = mlData["train"]['guesses_min']
    guesses[guesses!=0] = 1
    guesses_float = np.array(guesses, dtype=np.float64)
    print("sum(mlData['train']['labels'][guesses>0])/float(len(guesses))", 
           sum(mlData["train"]['labels'][guesses_float>0])/float(len(guesses_float[guesses_float>0])))
    
# varList = ["p1_data", "p2_data","p3_data","p4_data", "p5_data", "p6_data", "p7_data", "slope5_data", "slope30_data",
#                "decc_coeff", "guesses_min", "labels"] #6% accuracy # Gives 71% result. With hyperparameter tuning, 68%
# varList = ["slope5_data", "slope30_data",
#                "decc_coeff", "guesses_min", "labels"] # This gave 68% result

if __name__=="__main__":
    varList = ["p1_data", "p2_data",
            "decc_coeff", "guesses_min", "labels", 'correct_guess_freq' ]
    machLearn(varList)
