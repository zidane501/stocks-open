

from scipy.stats import linregress
import numpy as np, pandas as pd
import lightgbm as lgb



def predict_real(pandas_df, daysCloseData):
    """
    [predicts who to bet on day to day (Make sure to download the latest data). Prints out who to bet on]

    Arguments:
        pandas_df {[pandas dataframe]} -- [dataframe with Index: biz-code, and cols: csv-filenames (without .csv)
                                           A dataframe generated with clean_day_data-function.]
        daysCloseData {[python list]} -- [list of csv file names from csv files downloaded with stockmarket data]

    """
    # ml_model = lgb.Booster(model_file = "./lgb_model_50leaves.txt")
    ml_model = lgb.Booster(model_file = "./lgb_model.txt") # original model
    days_ml_vars = {}

    # Print to check whether it is the latest correct filenames
    print("daysCloseData[-4:]:", daysCloseData[-4:])

    # Forloop over all days data
    for i in range(len([daysCloseData[-1]])):
        print("i:", i)
        # range(daysCloseData[3:-1]): The first 4 has to be used for calculating decc coefficients and the last one has to used for labels
        pd_day = pd.DataFrame()
        p0,p1,p2,p3 = pandas_df[daysCloseData[i-1]], pandas_df[daysCloseData[i-2]], pandas_df[daysCloseData[i-3]], pandas_df[daysCloseData[i-4]]
        
        
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

        pd_day[daysCloseData[-2]]     = p1 # Yesterday
        pd_day[daysCloseData[-1]]     = p0 # Present Day
        pd_day["decc"]                = decc
        pd_day["decc0"]               = decc0
        pd_day["guesses_min"]         = guesses_min
        
        predict_arr = np.array([np.array(p0), np.array(p1), decc, decc0, guesses_min_bool])
        #### Predict using ML
        predictions = ml_model.predict(predict_arr.T)
        # print(predictions)
        pd_day["ml_predictions"] = predictions


        #### Print result
        df1 = pd_day.drop(pd_day[pd_day[daysCloseData[-1]]<1.5].index) # Remove all stocks worth less than 1.5

        # df1 = pd_day.sort_values('ml_predictions',ascending = False)
        df1 = df1.sort_values('ml_predictions',ascending = False)
        df1 = df1[-10:]
        
        print(df1, ":df1")
