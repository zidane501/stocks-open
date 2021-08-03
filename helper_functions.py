from scipy.stats import linregress
import numpy as np, pandas as pd, matplotlib.pyplot as plt
import lightgbm as lgb
import joblib, random
import math 

def get_local_minima_and_maxima(pandasData):
    minimas = []
    maximas = [] 
    for i in range(len(pandasData)):
        if i==0 or i==len(pandasData)-1:
            minimas.append(0)
            maximas.append(0)
            continue
        if (pandasData[i-1] > pandasData[i] and pandasData[i] < pandasData[i+1]):
            minimas.append(pandasData[i])
        else: minimas.append(0)

        if (pandasData[i-1] < pandasData[i] and pandasData[i] > pandasData[i+1]):
            maximas.append(pandasData[i])
        else: maximas.append(0)

    return { "minimas": np.array(minimas), "maximas": np.array(maximas)}

def lin_regr(pandasData):
    slopeData = []
    for i in range(len(pandasData)):
        slope5 = linregress(pandasData[i-5:i]) if i > 4 else 0
        slope30 = linregress(pandasData[i-30:i]) if i > 4 else 0
        dval = pandasData[i-1]-pandasData[i]
        slopeData.append((dval, slope5, slope30))

def clean_day_data(path, days):
    """
    Funtion that makes sure that only the same companies 
    are in the pandas dataframes for different days

    Arguments:
        days: {[python list]} -- [list of csv-files with day data from nasdaq or other stockmarkets]

    Returns:
        [pandas dataframe] -- [dataframe with Index: biz-code, and cols: csv-filenames (without .csv)]
    """
    pd_days_dict = {}

    biz_list_common = set([])
    j = 0
    for file in days:
        # Put dicts with data and business lists into pd_days_dict
        pd_days_dict[file] = {}
        pd_days_dict[file]['data'] = pd.read_csv(f'./{path}/' + file).sort_values(by=['Symbol'])
        pd_days_dict[file]['biz_list'] = pd_days_dict[file]['data']['Symbol']
           
        # Finding the common businesses 
        if j == 0:
            # The first iteration is just the biz_list
            biz_list_common = set(pd_days_dict[file]['biz_list'])
        else: biz_list_common = biz_list_common.intersection(set(pd_days_dict[file]['biz_list'])) 
        
        j+=1
    
    pandas_data = pd.DataFrame()
    for file in days:
        # Find unique biz for this day
        unique_biz = list(set(pd_days_dict[file]['data']['Symbol']) - set(biz_list_common))
        # Set the Symbol column as the index column
        pd_days_dict[file]['data'] = pd_days_dict[file]['data'].set_index('Symbol')
        # Remove the rows with the biz that are unique to that day
        pd_days_dict[file]['data'] = pd_days_dict[file]['data'].drop(unique_biz)
        # Add the closing data to a new pandas DF
        pandas_data[file[:-4]] = pd_days_dict[file]['data']['Close']
        
    ### Old code that works and does the same but only for 4 days
    # Function that takes the data from different days and 
    #Getting the pandas dataframes
    # d0,d1,d2,d3 = pd_days_dict[days[-1]],pd_days_dict[days[-2]], pd_days_dict[days[-3]], pd_days_dict[days[-4]] 
    
    # Getting the company names to remove different names
    # p3_biz = list(d3['Symbol']); p2_biz = list(d2['Symbol']); p1_biz = list(d1['Symbol']); p0_biz = list(d0['Symbol'])
    # common_names = set(p3_biz).intersection(set(p2_biz)).intersection(set(p1_biz)).intersection(set(p0_biz))
    # common_names = set(p3_biz + p2_biz + p1_biz + p0_biz)
    # unique_names3 = list(set(p3_biz) - common_names); unique_names2 = list(set(p2_biz) - common_names); unique_names1 = list(set(p1_biz) - common_names); unique_names0 = list(set(p0_biz) - common_names) 
    
    # d3 = d3.set_index('Symbol'); d2 = d2.set_index('Symbol'); d1 = d1.set_index('Symbol'); d0 = d0.set_index('Symbol')
    
    # d3 = d3.drop(unique_names3); d2 = d2.drop(unique_names2); d1 = d1.drop(unique_names1); d0 = d0.drop(unique_names0)
    
    # p3 = d3['Close']; p2 = d2['Close']; p1 = d1['Close']; p0 = d0['Close']
    
    # pandas_data = pd.DataFrame()
    # pandas_data["d0_close"], pandas_data["d1_close"], pandas_data["d2_close"], pandas_data["d3_close"] = p0, p1, p2, p3

    return pandas_data

def calc_cost(n, df1, account):
    """[Calculates Saxobanks 0.02 cent cost pr stock purchase]

    Args:
        n ([int]): [description]
        df1 ([pandas dataFrame]): [dataFrame containing features and stockprizes for comapanies]
        account ([float]): [accountvalue]

    Returns:
        [float]: [cost of trading the stocks]
    """
    money_to_buy    = account/float(n)                          ;#print('money_to_buy:'  , money_to_buy)
    num_buys_array  = np.floor(np.array(money_to_buy/df1['This day'])); #print("num_buys_array:", num_buys_array)
    cost            = np.sum(num_buys_array*0.02)               ; #print("cost:"          , cost)
    
    return 0.2

def calc_ml_vars(pandas_df, daysCloseData):
    """
    [Calculates the variables needed for machine learning]

    Arguments:
        pandas_df {[pandas dataframe]} -- [dataframe with Index: biz-code, and cols: csv-filenames (without .csv)
                                           A dataframe generated with clean_day_data-function.]
        daysCloseData {[python list]} -- [list of csv file names from csv files downloaded with stockmarket data]

    Returns:
        [dict of pandas dataframes] -- [keys: csv-filenames (without .csv) for each day. 
                  Items: pandas dataframes, with machinelearning vars as columns
                  d0 (present day closing data)    d1 (Day before)    decc     decc0      guesses_min  labels,
                  Index: Symbol]
    """
    ml_model = lgb.Booster(model_file = "./lgb_model_50leaves.txt")
    
    print('sorted(daysCloseData)[0], sorted(daysCloseData)[-1]:', sorted(daysCloseData)[0], sorted(daysCloseData)[-1])
    # # print('pandas_df.index:',pandas_df.index)
    # ndq_comp0 = pandas_df['IXIC'][0]
    # ndq_comp1 = pandas_df['IXIC'][1]
    # print("Nasdaq composit change:",(ndq_comp1 - ndq_comp0)/ndq_comp0)
    # ml_model = lgb.Booster(model_file = "./lgb_model.txt")
    account = [100]
    days_ml_vars = {}
    score = 0 # used to calculate labels score
    total_ave_pct = 0

    n = 10 # How many stocks to show and bet on per day

    # Forloop over all days data
    rand_list = [100]
    average   = [100]
    average_p0= []    
    closeData = sorted(daysCloseData)[:-1]
    joblib.dump(closeData, 'data/closeData')

    days = len(closeData)
    for i in range(days):
        print(f"\nDay: daysCloseData[{i}]", daysCloseData[i])
        # range(daysCloseData[3:-1]): The first 4 has to be used for calculating decc coefficients and the last on has to used for labels
        pd_day = pd.DataFrame()
        p_l,p0,p1,p2,p3 = pandas_df[daysCloseData[i+1]],pandas_df[daysCloseData[i]], pandas_df[daysCloseData[i-1]], pandas_df[daysCloseData[i-2]], pandas_df[daysCloseData[i-3]]
        
        if i<3: continue

        v0 = p0-p1
        v1 = p1-p2
        acc = v0-v1

        mask = p0>250
        mask_acc = acc>0
        mask = np.logical_and(acc>0, acc<1, mask , p0<0 , p1<0)

        p_l,p0,p1,p2,p3 = p_l[mask],p0[mask],p1[mask],p2[mask],p3[mask]


        
        #### decc0
        decc                  = np.array((p1/p0)/(p2/p1))
        decc_bool             = np.logical_and(p2-p1>0, p1-p0>0)
        decc_bool_not         = np.logical_not(decc_bool)
        decc[decc_bool_not]   = 1
        
        #### decc0
        decc0                 = np.array((p2/p1)/(p3/p2))
        decc0_bool            = np.logical_and(p3-p2>0, p2-p1>0)
        decc0_bool_not        = np.logical_not(decc0_bool)
        decc0[decc0_bool_not] = 1

        #### guesses_min
        # if (deceleration_coefficient_min < 1 or deceleration_coefficient_min0 < 1) and curve_descend_bool and decc_coef_bool and decc_coef0_bool:
            # guesses_min.append(p0) if i > 4 else guesses_min.append(0)
        
        guesses_min_bool              = np.logical_or(decc<1, decc0<1)

        curve_descend_bool            = np.logical_and(p3-p0>0, decc_bool)
        curve_descend_bool            = np.logical_and(curve_descend_bool, decc0_bool)
        
        guesses_min_bool              = np.logical_and(guesses_min_bool, curve_descend_bool)

        guesses_min                   = np.zeros(len(p0)) 
        guesses_min[guesses_min_bool] = None
        pd_day["Yesterday"]           = p1 # Yesterday
        pd_day["This day"]            = p0 # Present Day
        pd_day["Next day"]            = p_l
        pd_day["decc"]                = decc
        # pd_day["decc0"]               = decc0
        pd_day["guesses_min"]         = guesses_min
        
        predict_arr = np.array([np.array(p0), np.array(p1), decc, decc0, guesses_min_bool])
        #### Predict using ML
        predictions = ml_model.predict(predict_arr.T)
        print("predictions:", predictions)
        pd_day["ml_predictions"] = predictions

        #### labels
        labels                   = np.logical_and(p0<p1, p0<p_l)
        pd_day["labels"]         = labels
        pd_day["value change %"] = (p_l-p0)*100/p0

        frac_buy = (account[-1]/float(n))

        #### Print result
        # print("len(pd_day.index):", len(pd_day.index))
        df1 = pd_day.drop(pd_day[pd_day['This day']<1].index) # Remove all prices less than 2
        df1 = df1.drop(df1[df1['This day']>700].index) # Remove all prices less than 2
        # print("df1['This day']:", df1['This day'])
        # print("len(df1.index):", len(df1['This day']))
    
        df1 = df1.sort_values('ml_predictions', ascending = False) # Change df1. to pd_day to not remove companies from list
        df1 = df1[:n] if 35<i<12 else df1[:n]
        # df1 = df1[-n:] if 35<i<12 else df1[-n:] # Test whether to look for high vals or low when the index goes up or down
        # val_cha_pct = np.array(df1['value change %'])
        # top70 = val_cha_pct[np.array(df1["ml_predictions"])>0.65]
        # n_over_70 = n if len(top70)>=n else len(top70)  
        # print("top70:", top70, ":top70")
        # top10_over_70_mean = np.mean(top70[:n_over_70]) if top70.any() else 0
        # 
        # top70_df = pd.DataFrame()

        ### See which to bet on day to day
        change_rand_picks = random.choices(pd_day['value change %'], k = 10)
        rand_list.append(rand_list[-1] + np.mean(change_rand_picks)*rand_list[-1]/float(100))
        average.append(average[-1] + np.mean(pd_day['value change %'])*average[-1]/float(100))
        average_p0.append(np.mean(p0))
        
        # Calculate costs of buying
        cost = calc_cost(n, df1, account[-1]); #cost = 0;

        account.append(account[-1] + account[-1]*np.mean(df1['value change %'])/float(100)-cost)
        
        # account += account*top10_over_70_mean/100

        print("mean(df1['value change %']):", np.mean(df1['value change %']), "| mean(True)", np.mean(df1['labels']), "| account",account[-1])
        print("\nmean value change:", np.mean(df1['value change %']), '%\n')

        score += np.sum(df1['labels'])
        total_ave_pct += np.mean(df1['value change %'])

        #Add dataframe to dict
        days_ml_vars[closeData[i]] = pd_day
        
        if i > days - 5: print(df1, ":df1")
        print('closeData[i]:', closeData[i])

    print("score/float(len(daysCloseData[3:])):", score/float(len(closeData[3:])))
    print('total_ave_pct:', total_ave_pct)
    
    ### For plotting Nasdaq index curve
    # nasdaq_index_IXIC = pd.read_csv('IXIC.csv').sort_values(by=['Date'])
    # nasdaq_IXIC_close = nasdaq_index_IXIC['Close']
    # nasdaq_IXIC_date  = nasdaq_index_IXIC['Date']

    # joblib.dump(nasdaq_IXIC_date, 'nasdaq_IXIC_date')

    if 1: 
        fig, ax = plt.subplots(figsize=(15,15))
        filter = list(map(lambda x: True if x%1.0<=0 else False, range(len(account)-1)))
        ax.plot(np.array(account[1:])[filter], '-x', label = 'Model')
        # ax.plot(np.array(rand_list[1:])[filter],     label = 'Random')
        # ax.plot(np.array(average[1:])[filter],       label ='Average')
        ax.plot(np.array(average_p0),       label ='Ave p0')

        # plt.plot(np.array(nasdaq_index_IXIC)[filter], '-x')

        print("len(account), len(closeData):", len(account), len(closeData))
        # plt.xticks([50,100,150],[daysCloseData[44], daysCloseData[88], daysCloseData[132]])
        dates_data = np.array(closeData[3:])[filter]
        dates_data = [i[-2:]+"."+i[-4:-2] for i in dates_data]
        
        # ax.set_xticks(range(len(dates_data)), dates_data)
        # ax.tick_params(labelrotation=45)
        ax.set_xticklabels(dates_data, rotation=45)
        ax.set_yscale('log')
        ax.legend()
        plt.show()
    
    return days_ml_vars


def calc_acc_down(pandas_df, daysCloseData):
    """
    Predicts on basis of curve going down, acceleration positive and value more than 250

    Arguments:
        pandas_df {[pandas dataframe]} -- [dataframe with Index: biz-code, and cols: csv-filenames (without .csv)
                                           A dataframe generated with clean_day_data-function.]
        daysCloseData {[python list]} -- [list of csv file names from csv files downloaded with stockmarket data]

    Returns:
        [dict of pandas dataframes] -- [keys: csv-filenames (without .csv) for each day. 
                  Items: pandas dataframes, with machinelearning vars as columns
                  d0 (present day closing data)    d1 (Day before)    decc     decc0      guesses_min  labels,
                  Index: Symbol]
    """
    # ml_model = lgb.Booster(model_file = "./lgb_model_50leaves.txt")
    
    print('sorted(daysCloseData)[0], sorted(daysCloseData)[-1]:', sorted(daysCloseData)[0], sorted(daysCloseData)[-1])
    
    account = [10000]
    days_ml_vars = {}
    score = 0 # used to calculate labels score
    total_ave_pct = 0

    n = 1 # How many stocks to show and bet on per day

    # Forloop over all days data
    average_p0= []    
    closeData = sorted(daysCloseData)[:-1]
    joblib.dump(closeData, 'data/closeData')
    sig = 0
    bkg = 0
    days = len(closeData)
    for i in range(days):
        # print("Yyoyoyoyoyoyyo\nyo\nyo\nyo\nyo\nyo\nyo\nyo\n")
        print(f"\nDay: daysCloseData[{i}]", daysCloseData[i])
        # range(daysCloseData[3:-1]): The first 4 has to be used for calculating decc coefficients and the last on has to used for labels
        pd_day = pd.DataFrame()
        p_l,p0,p1,p2,p3 = pandas_df[daysCloseData[i+1]],pandas_df[daysCloseData[i]], pandas_df[daysCloseData[i-1]], pandas_df[daysCloseData[i-2]], pandas_df[daysCloseData[i-3]]
        
        # pd_day["dbf Yesterday"]           = p2 # Yesterday
        # pd_day["Yesterday"]           = p1 # Yesterday
        # pd_day["This day"]            = p0 # Present Day
        # pd_day["Next day"]            = p_l

        if i<3: continue

        v0 = (p0-p1)/p1
        v1 = (p1-p2)/p2
        acc = v0-v1
        v_rel = v1/v0
        print("\nlen(p0) before mask:", len(p0))
        # acc_mask = np.logical_and(acc>0, acc<1)
        acc_mask = v_rel>1.0
        # acc_mask = np.logical_or(acc_mask, acc>2.5)

        mask = np.logical_and(p0>5, p0<100000)
        mask = np.logical_and(mask, acc_mask)
        mask = np.logical_and(mask, v0  < 0)
        mask = np.logical_and(mask, v1  < 0)
        
        p_l,p0,p1,p2,p3 = p_l[mask],p0[mask],p1[mask],p2[mask],p3[mask]
        
        if len(p0)<1:
            print('continue')
            continue
        print("len(p0) after mask:",len(p0), "\n")
        
        # pd_day["Yesterday"]           = p1 # Yesterday
        # pd_day["This day"]            = p0 # Present Day
        # pd_day["Next day"]            = p_l
        # pd_day["acc"]                 = acc[mask]
        
        #### Predict using ML

        #### labels
        labels                   = p0<p_l
        pd_day["labels"]         = labels # Present Day
        pd_day["p0"]             = p0 # Present Day
        pd_day["p_l"]            = p_l # Next Day
        pd_day["acc"]            = v_rel
        pd_day["value change %"] = (p_l-p0)*100/p0

        # frac_buy = (account[-1]/float(n))

        #### Print result
       
        df1 = pd_day.sort_values('acc', ascending = False) # Change df1. to pd_day to not remove companies from list
        # df1 = df1[:n] if 35<i<12 else df1[:n]
        # df1 = df1[-n:] if 35<i<12 else df1[-n:] # Test whether to look for high vals or low when the index goes up or down
        # val_cha_pct = np.array(df1['value change %'])
        # top70 = val_cha_pct[np.array(df1["ml_predictions"])>0.65]
        # n_over_70 = n if len(top70)>=n else len(top70)  
        # print("top70:", top70, ":top70")
        # top10_over_70_mean = np.mean(top70[:n_over_70]) if top70.any() else 0
        # 
        # top70_df = pd.DataFrame()

        ### See which to bet on day to day
        # change_rand_picks = random.choices(pd_day['value change %'], k = 10)
        
        # Calculate costs of buying
        # cost = calc_cost(n, df1, account[-1]); #cost = 0;
        
        if not pd.isna(df1.iloc[0]['value change %']): # Check for nan in pandas
            cost = 0.02*account[-1]/df1.iloc[0]['p0']
            account.append(account[-1] + (account[-1]*df1.iloc[0]['value change %']/float(100)-cost))

            sig += len(np.array(df1['labels'])[np.array(df1['labels'])])
            bkg += len(np.array(df1['labels'])[np.array(df1['labels'])<0.5])
        print("sig/(sig+bkg):", sig/(sig+bkg))
        print("account[-1]:", account[-1])
        print("cost:", cost)
        print("df1.iloc[0]:\n", df1.iloc[0])
        print("df1.iloc[0]['value change %']:", df1.iloc[0]['value change %'], type(df1.iloc[0]['value change %']))
        # account += account*top10_over_70_mean/100

        # print("mean(df1['value change %']):", np.mean(df1['value change %']), "| mean(True)", np.mean(df1['labels']), "| account",account[-1])
        # print("\nmean value change:", np.mean(df1['value change %']), '%\n')

        score += np.sum(df1['labels'])
        total_ave_pct += np.mean(df1['value change %'])

        #Add dataframe to dict
        days_ml_vars[closeData[i]] = pd_day
        
        if i > days-20: print("df1:\n",df1) #days - 5: print(df1, ":df1")
        # print('closeData[i]:', closeData[i], "p0:", p0)

    print("account[-1]:",account[-1])
    print("score/float(len(daysCloseData[3:])):", score/float(len(closeData[3:])))
    print('total_ave_pct:', total_ave_pct)
    print(days)
    ### For plotting Nasdaq index curve
    # nasdaq_index_IXIC = pd.read_csv('IXIC.csv').sort_values(by=['Date'])
    # nasdaq_IXIC_close = nasdaq_index_IXIC['Close']
    # nasdaq_IXIC_date  = nasdaq_index_IXIC['Date']

    # joblib.dump(nasdaq_IXIC_date, 'nasdaq_IXIC_date')

    if 1: 
        fig, ax = plt.subplots(figsize=(15,15))
        filter = np.array(list(map(lambda x: True if x%1.0<=0 else False, range(len(account)-1))))
        ax.plot(np.array(account[1:])[filter], '-x', label = 'Model')
        # ax.plot(np.array(rand_list[1:])[filter],     label = 'Random')
        # ax.plot(np.array(average[1:])[filter],       label ='Average')
        ax.plot(np.array(average_p0),       label ='Ave p0')

        # plt.plot(np.array(nasdaq_index_IXIC)[filter], '-x')

        print("len(account), len(closeData):", len(account), len(closeData))
        # plt.xticks([50,100,150],[daysCloseData[44], daysCloseData[88], daysCloseData[132]])
        dates_data = np.array(closeData[3:])[filter]
        dates_data = [i[-2:]+"."+i[-4:-2] for i in dates_data]
        
        # ax.set_xticks(range(len(dates_data)), dates_data)
        # ax.tick_params(labelrotation=45)
        ax.set_xticklabels(dates_data, rotation=45)
        ax.set_yscale('log')
        ax.legend()
        plt.show()
    
    return days_ml_vars