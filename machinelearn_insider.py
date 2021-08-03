import lightgbm as lgb
import numpy as np, pandas as pd
import joblib, datetime, time, os
from skopt.space import Real, Integer
from hyper_parameter_tuning import hyper_parameter_tuning_main
from sklearn.metrics import make_scorer, log_loss
from skopt import gp_minimize, dump, load
from BS4_get_stockprice import change_in_sec_vs_buy_date, make_insider_labels
import matplotlib.pyplot as plt

def load_data():
    dir_path = os.path.dirname(os.path.realpath(__file__))+ "/data/"
    data = joblib.load(dir_path + 'insider_scraped_data')
    print(data[:10])
    print(data.keys())
    daysInSecs = 60*60*24*4 # 4 days
    now = get_current_time_in_secs() # Gets the num of secs since they started counting in the 70ies

    # Yahoo finance cant download 'LGVW'
    data = data.loc[data["time_stamp"] < now - daysInSecs] # Not get data from the last 4 days

    faults = ['LGVW', 'EIGI', 'CFII']
    data = remove_row(data, "Ticker", faults)

    print("len(data):", len(data))   

    labels = joblib.load('data/labels_insider')
    change_in_price_SEC_buy = joblib.load("data/change_in_sec_vs_buy_date_insider")
    
    if len(labels) != len(data):
        labels_extra = [make_insider_labels(data.iloc[len(labels)+row], 3) for row in range(len(data)-len(labels))]
        joblib.dump(labels+labels_extra, "data/labels_insider")
        data['labels'] = np.array(labels+labels_extra)>0
    else:
        data['labels'] = labels
    if len(change_in_price_SEC_buy) != len(data):
        change_in_price_SEC_buy_extra = [change_in_sec_vs_buy_date(data.iloc[row+len(change_in_price_SEC_buy)]) for row in range(len(data)-len(change_in_price_SEC_buy))]
        joblib.dump(change_in_price_SEC_buy+change_in_price_SEC_buy_extra, "data/change_in_sec_vs_buy_date_insider")
        data['price_change_SEC_to_buy'] = np.array(change_in_price_SEC_buy+change_in_price_SEC_buy_extra    )

    else:
        data['price_change_SEC_to_buy'] = np.array(change_in_price_SEC_buy)
    
    data = data_strings_to_floats(data)

    print("len(data['labels']):",len(data['labels']))
    return data

    data['buy_to_sec']  = change_buy_to_sec

def get_current_time_in_secs():
    """
    Returns the amount of seconds since they started counting
    in the 1970ies

    Returns:
        Float
    """
    now = datetime.datetime.now()
    d = datetime.datetime.strptime(now.strftime("%Y %b %d %I:%M %p"), "%Y %b %d %I:%M %p")
    local_time = time.mktime(d.timetuple())
    return local_time

def remove_row(data, col, values):
    for value in values:
        index_rm = data.index[data[col] == value].tolist()
        data = data.drop(index_rm)
    return data

def cross_validation(data):
    n_points = len(data)
    pct_80, pct_90 = int(0.80*n_points), int(0.90*n_points)
    train_data = data[      :pct_80]
    val_data   = data[pct_80:pct_90]
    test_data  = data[pct_90:      ]
    return train_data, val_data, test_data

def make_lgb_data_set(data):
    data_np = np.array(data.loc[:,["#Shares", "Value", "#Shares Total", "price_change_SEC_to_buy"]])
    print("make_lgb_data_set len(data):", len(data))
    print("make_lgb_data_set len(data['labels']):", len(data['labels']))
    
    return lgb.Dataset(data_np, label = np.array(data['labels']))

def machine_learn(lgbData):
    model = lgb.train(lgbData)

def stringsList_to_floatList(list_of_comma_numbers):
    return list(map(lambda x: float(x.replace(',',"")),list_of_comma_numbers))

def data_strings_to_floats(data):
    data["#Shares"]         = stringsList_to_floatList(data["#Shares"])
    data["Value"]           = stringsList_to_floatList(data["Value"])
    data["#Shares Total"]   = stringsList_to_floatList(data["#Shares Total"])
    data["Cost"]            = stringsList_to_floatList(data["Cost"])
    # data["Cost"] = list(map(lambda x: float(x),data["Cost"]))
    return data

if __name__=="__main__":
    data = load_data()
    print(data.loc[data['labels']>0.1])
    labels = np.array(data['labels'])>0.1
    print(data['labels'][:10])
    print(len(labels[labels])/float(len(data['labels'])))

    train, val, test = cross_validation(data)
    train_lgb = make_lgb_data_set(pd.concat([train,val]))
    test_lgb  = make_lgb_data_set(test)
    
    param = {'objective': 'binary'}
    param['metric'] = ['binary_logloss']

    model = lgb.train(param, train_lgb, 10000)#, early_stopping_rounds=100,)

    test_np = np.array(test.loc[:,["#Shares", "Value", "#Shares Total", "price_change_SEC_to_buy"]])
    test_labels = np.array(test['labels'])
    print(test_labels[:10])
    ypred = model.predict(test_np, num_iteration=model.best_iteration)    
    
    plt.scatter(ypred, test_labels)
    # plt.hist(ypred[test_labels>0], bins=30, label="Sig")
    # plt.hist(ypred[test_labels<0.0], bins=30, label="Bkg")
    plt.legend()
    plt.show()