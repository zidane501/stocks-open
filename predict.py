import lightgbm as lgb
import numpy as np, pandas as pd
import joblib
from skopt.space import Real, Integer
from hyper_parameter_tuning import hyper_parameter_tuning_main
from sklearn.metrics import make_scorer, log_loss
from skopt import gp_minimize, dump, load
import os
from helper_functions import clean_day_data, calc_ml_vars, calc_acc_down
from predict_real import predict_real

nasdaq = 1
penny = 0
if nasdaq: # Data from nasdaq
    ### get data from last 3 days
    nasdaq_closing_days_files = sorted(os.listdir('./nasdaq_data'))
    # print(nasdaq_closing_days_files)

    ### - put it on the right form
    pandas_data_df = clean_day_data('nasdaq_data', nasdaq_closing_days_files)

    #pandas_feature_df has columns from closing data, and the column names are pandas nasdaq_closing_days_files[i][:-4]
    daysCloseData_list = [i[:-4] for i in nasdaq_closing_days_files]

    # This one is for testing 
    # calc_ml_vars(pandas_data_df, daysCloseData_list)
    calc_acc_down(pandas_data_df, daysCloseData_list)
    
    # This one is for betting
    # predict_real(pandas_data_df, daysCloseData_list)

if penny: #data from otcbb
    otcbbclosing_days_files = sorted(os.listdir('./OTCBB'))
    pandas_data_df = clean_day_data('OTCBB', otcbbclosing_days_files)
    daysCloseData_list = [i[:-4]for i in otcbbclosing_days_files]
    calc_ml_vars(pandas_data_df, daysCloseData_list)
    # predict_real(pandas_data_df, daysCloseData_list)

# def calc_deceleration_coefficient0(pandas_df):
#     p0,p1,p2,p3 = pandas_df['d0_close'], pandas_df['d1_close'], pandas_df['d2_close'], pandas_df['d3_close']
#     decc = np.array((p2/p1)/(p3/p2))
#     # print("pandas_df:",pandas_df)

#     for i in range(len(pandas_df)):
#         point0, point1, point2 = p0[i], p1[i], p2[i]
#         if not (p3[i]-p2[i] > 0 and (p2[i]-p1[i]) > 0) : 
#             decc[i] = 1
#         pandas_df["decc_coef0"] = decc
    
#     # print("pandas_df:",pandas_df)
#     return pandas_df

# def calc_guesses_min(pandas_df):
#     # p0,p1,p2,p3 = pandas_df['d0_close'], pandas_df['d1_close'], pandas_df['d2_close'], pandas_df['d3_close']
    
#     # curve_descend_bool = 1 if p3-p0>0 and p3-p2>0 and p2-p1>0 and p1-p0>0 else 0
#     # d4curve = p3-p0
#     # if (deceleration_coefficient_min < 1 or deceleration_coefficient_min0 < 1) and curve_descend_bool :
#     #     guesses_min.append(p0) if i < 4 else guesses_min.append(None) 
    
#     for i in range(len(pandas_df)):
#         point0, point1, point2, point3 = pandas_df["d0_close"][i], pandas_df["d1_close"][i], pandas_df["d2_close"][i],  pandas_df["d3_close"][i]
#         curve_descend_bool = 1 if point3-point0>0 and point3-point2>0 and point2-point1>0 and point1-point0>0 else 0
#         guesses_min = np.zeros(len(pandas_df)) 
#         if (pandas_df['decc_coef'][i] < 1 or pandas_df['decc_coef0'][i] < 1) and curve_descend_bool :
#             guesses_min[i] = None
#         pandas_df['guesses_min'] = guesses_min
#         return pandas_df
         


# pandas_feature_df = calc_deceleration_coefficient(pandas_feature_df)
# # print("pandas_feature_df:", pandas_feature_df)
# pandas_feature_df = calc_deceleration_coefficient0(pandas_feature_df)
# pandas_feature_df = calc_guesses_min(pandas_feature_df)

# print("pandas_feature_df:",pandas_feature_df)


# ##### predict
# # - Make labels
# def label_maker(days):
#     print(len(days))
#     day1= pd.read_csv('./nasdaq_data/' + days[0]).sort_values(by=['Symbol']).Close
#     day2= pd.read_csv('./nasdaq_data/' + days[1]).sort_values(by=['Symbol']).Close
#     day3= pd.read_csv('./nasdaq_data/' + days[2]).sort_values(by=['Symbol']).Close
#     print("day1:",day1,":day1", "type(day1):", type(day1))

# label_maker(nasdaq_closing_days_files[-3:])
# - load model
# - predict
# - find the companies to bet on