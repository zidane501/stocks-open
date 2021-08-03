# import urllib, selenium
import matplotlib.pyplot as plt
# from numpy.core.function_base import linspace
from urllib.request import Request, urlopen
from bs4 import BeautifulSoup
# import requests
# import webbrowser
import pandas as pd, numpy as np, joblib
import datetime
import time, yfinance as yf
import os, re 
# from selenium import webdriver
# from selenium.webdriver.common.action_chains import ActionChains
# from selenium.webdriver.support.ui import WebDriverWait
# from selenium.webdriver.support import expected_conditions as EC
# from selenium.webdriver.common.by import By
# from selenium.common.exceptions import TimeoutException

from helper_functions import clean_day_data

def get_eod_data():
    nasdaq_closing_days_files = sorted(os.listdir('./nasdaq_data'))
    # print(nasdaq_closing_days_files)

    ### - put it on the right form
    pandas_data_df = clean_day_data('nasdaq_data', nasdaq_closing_days_files)

    corr = []
    std_total = []
    change_in_pct = []
    

    ##### Parameter (days wait before sale):
    label_day              = 5 # Best 7
    change_pr_day_selected = []
    sig_pr_day             = []
    bkg_pr_day             = []
    
    if 0:
        for i in range(len(pandas_data_df.keys())-label_day):
        
            ##### Parameter (stock value):
            stock_max_val = 100.0
            less_than_2 = np.logical_and(np.array(pandas_data_df[pandas_data_df.keys()[i]])<stock_max_val, np.array(pandas_data_df[pandas_data_df.keys()[i]])>1.0)
            
            today = np.array(pandas_data_df[pandas_data_df.keys()[i]])[less_than_2]
            tmr =   np.array(pandas_data_df[pandas_data_df.keys()[i+1]])[less_than_2]
            days3 = np.array(pandas_data_df[pandas_data_df.keys()[i+label_day]])[less_than_2]

            # f = lambda x: np.array(pandas_data_df[pandas_data_df.keys()[i+x]])[less_than_2]

            
            diff = np.array((tmr-today)/today)
            ##### Parameter:
            mask_pct = np.logical_and(diff>0.1, diff<10.9)
            
            print(pandas_data_df.keys()[i], len(diff[mask_pct]))
            
            
            today, tmr, days3 = today[mask_pct], tmr[mask_pct], days3[mask_pct]

            ##### Parameter (number of days to calculate linear corelation):
            line_n = 10
            if len(diff[mask_pct])>0:
                if i > line_n + 1:
                                    
                    change_pct_pr_day = (days3-tmr)/tmr
                    # std = [np.std(pandas_data_df.loc[biz, pandas_data_df.keys()[i-20]: pandas_data_df.keys()[i]]) for biz in pandas_data_df.index[less_than_2]]
                    if 1:
                        corr_straight_line = [np.corrcoef(pandas_data_df.loc[biz, pandas_data_df.keys()[i-line_n]: pandas_data_df.keys()[i]], np.linspace(pandas_data_df.loc[biz, pandas_data_df.keys()[i-line_n]], pandas_data_df.loc[biz, pandas_data_df.keys()[i]], line_n + 1))[0,1] for biz in pandas_data_df.index[less_than_2][mask_pct]]
                        corr += list(corr_straight_line)
                    
                    # Sell if the stock has fallen more than x pct
                    drop = 0.0
                    for k, biz in enumerate(pandas_data_df.index[less_than_2][mask_pct]):
                        # i = int(np.round(i))
                        # print("i+label_day:",i+label_day)
                        # print(pandas_data_df.keys()[i+label_day])
                        dev_pct_change_till_salesday = (pandas_data_df.loc[biz, pandas_data_df.keys()[i]: pandas_data_df.keys()[i+label_day]] - pandas_data_df.loc[biz, pandas_data_df.keys()[i]])/pandas_data_df.loc[biz, pandas_data_df.keys()[i]]
                        for early_sale in dev_pct_change_till_salesday:
                            if early_sale <drop:
                                # print("change_pct_pr_day[k], early_sale:",change_pct_pr_day[k], early_sale)
                                change_pct_pr_day[k] = early_sale

                    # i = int(np.round(i))
                    std = [np.std((pandas_data_df.loc[biz, pandas_data_df.keys()[i-line_n]: pandas_data_df.keys()[i]]))/np.mean(pandas_data_df.loc[biz, pandas_data_df.keys()[i-line_n]: pandas_data_df.keys()[i]]) for biz in pandas_data_df.index[less_than_2][mask_pct]]
                    std_total += std
                    
                    # change_pct_pr_day = (days3-tmr)/tmr
                    change_in_pct += list(change_pct_pr_day)

                    if np.mean((days3-tmr)/tmr) > 0.3:
                        # corr_straight_line
                        print(list(zip(np.array(pandas_data_df.index)[less_than_2][mask_pct], (tmr-days3)/tmr, today , std)))
                    
                    corr_line_std_mask = np.logical_and(np.array(corr_straight_line) > 0.95, np.array(std) > 0.1)
                    change_pr_day_selected += [np.mean(change_pct_pr_day[corr_line_std_mask])] 
                    
                    cpd_cls_mask = change_pct_pr_day[corr_line_std_mask]
                    
                    # if len(cpd_cls_mask)>0:
                    sig_pr_day += [len(cpd_cls_mask[cpd_cls_mask>0])]
                    bkg_pr_day += [len(cpd_cls_mask[cpd_cls_mask<0])]
                    # else: 
                    #     sig_pr_day += [-0.1]
                    #     bkg_pr_day += [-0.1]
        
        joblib.dump(sig_pr_day, 'data/sig_pr_day_cpd_cls_mask')
        joblib.dump(bkg_pr_day, 'data/bkg_pr_day_cpd_cls_mask')

        print("len(corr), len(change_in_pct):", len(corr), len(change_in_pct))
        
        change_in_pct, corr, std_total = np.array(change_in_pct), np.array(corr), np.array(std_total) 
        # change_in_pct, std_total = np.array(change_in_pct), np.array(std_total) 
        # cut_off = 100 
        # change_in_pct, corr = change_in_pct[change_in_pct<cut_off], corr[change_in_pct<cut_off] 

        # print("len(corr), len(change_in_pct):", len(corr), len(change_in_pct))
        mask = np.logical_and(std_total>0.1, corr>0.95)
        # print("np.mean(change_in_pct[corr>0.90]):", np.mean(change_in_pct[corr>0.90]))
        print("np.mean(change_in_pct[mask]):", np.mean(change_in_pct[mask]))
        print("len(Sig):", len(change_in_pct[mask][change_in_pct[mask]>0]), len(change_in_pct[mask][change_in_pct[mask]>0])/float(len(change_in_pct[mask])))
        print("len(Bkg):", len(change_in_pct[mask][change_in_pct[mask]<0]))
        
        change_pr_day_selected = np.array(change_pr_day_selected)
        print("Days len(Sig):", len(change_pr_day_selected[change_pr_day_selected>0]))
        print("Days len(Bkg):", len(change_pr_day_selected[change_pr_day_selected<0]))
        
        print("Days mean(Sig):", np.mean(change_pr_day_selected[change_pr_day_selected>0]))
        print("Days mean(Bkg):", np.mean(change_pr_day_selected[change_pr_day_selected<0]))

    sig_pr_day = np.array(joblib.load('data/sig_pr_day_cpd_cls_mask'))
    bkg_pr_day = np.array(joblib.load('data/bkg_pr_day_cpd_cls_mask')) 
    
    # bkg_pr_day = bkg_pr_day[sig_pr_day>100]
    # sig_pr_day = sig_pr_day[sig_pr_day>100]
    # plt.scatter(std_total[mask], change_in_pct[mask], marker = '.')
    print('len(sig_pr_day):', len(sig_pr_day))
    print('len(bkg_pr_day):', len(bkg_pr_day))
    plt.plot(np.array(sig_pr_day),'.', label = 'Sig')
    plt.plot(np.array(bkg_pr_day),'.', label = 'Bkg')
    plt.hlines(0, -1,400)
    plt.legend()
    plt.show()

def rentes_rente(money, pct1, n1, pct2, n2,):
    print( money*((1+pct1)**(n1))*(1+pct2)**(n2))

def today():
    nasdaq_closing_days_files = sorted(os.listdir('./nasdaq_data'))
    # print(nasdaq_closing_days_files)

    ### - put it on the right form
    pandas_data_df = clean_day_data('nasdaq_data', nasdaq_closing_days_files)
    nDays = 53
    less_than_2 = np.logical_and(np.array(pandas_data_df[pandas_data_df.keys()[-nDays]])<1000, np.array(pandas_data_df[pandas_data_df.keys()[-nDays]])>1)
        
    yesterday = pandas_data_df[pandas_data_df.keys()[-nDays-1]][less_than_2]
    today     = pandas_data_df[pandas_data_df.keys()[-nDays]][less_than_2]
    label     = pandas_data_df[pandas_data_df.keys()[-nDays-1+30]][less_than_2]
    pct_change = (today-yesterday)/yesterday
    
    label_pct =  (label - today)/today
    
    line_n = 10
    
    mask_pct = np.logical_and(pct_change > 0.15, pct_change<10)

    corr_straight_line = [np.corrcoef(pandas_data_df.loc[biz, pandas_data_df.keys()[-nDays-1-line_n]: pandas_data_df.keys()[-nDays-1]], np.linspace(pandas_data_df.loc[biz, pandas_data_df.keys()[-nDays-1-line_n]], pandas_data_df.loc[biz, pandas_data_df.keys()[-nDays-1]], line_n + 1))[0,1] for biz in pandas_data_df.index[less_than_2][mask_pct]]
    std = [np.std(pandas_data_df.loc[biz, pandas_data_df.keys()[-nDays-1-line_n]: pandas_data_df.keys()[-nDays-1]]) for biz in pandas_data_df.index[less_than_2][mask_pct]]

    corr_straight_line, std = np.array(corr_straight_line), np.array(std)

    mask_corr_std = np.logical_and(corr_straight_line > 0.95, std > 0.1)

    print('name', 'value', 'corr', 'std')

    label_pct_mask = label_pct[mask_pct][mask_corr_std]

    for i in list(zip(today[mask_pct][mask_corr_std].index, today[mask_pct][mask_corr_std], corr_straight_line[mask_corr_std], std[mask_corr_std], label_pct[mask_pct][mask_corr_std])):
        # if i[2]>0.95 and i[3] > 0.1:
        print(i[0], "{:.2f}".format(i[1]), "{:.2f}".format(i[2]), "{:.2f}".format(i[3]), "{:.2f}".format(i[4]))
        # ave_pct_pr_day = map(lambda x: day = [],i)
    print('name', 'value', 'corr', 'std')
    # print("label_pct",label_pct)
    print("mean(label_pct):", "{:.2f}".format(np.mean(label_pct[mask_pct][mask_corr_std])))
    # print("corr_straight_line:", corr_straight_line)
    # print("std:", std)
    print(np.mean(label_pct[mask_pct][mask_corr_std]))
    print('len(Sig)',len(label_pct_mask[label_pct_mask > 0])/len(label_pct_mask))

def LH(bkg,sig, delta):
    hist_sig, bin_edges = np.histogram(sig, bins=200)
    hist_sig, bin_edges = np.histogram(sig, bins=200)


if __name__=="__main__":
    get_eod_data()
    # today()



