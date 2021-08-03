import matplotlib.pyplot as plt, numpy as np, joblib ,pandas as pd
from scipy import stats
from helper_functions import get_local_minima_and_maxima
import lightgbm as lgb
# Import the yfinance. If you get module not found error the run !pip install yfiannce from your Jupyter notebook
import yfinance as yf  


def freq(np_array):
    return len(np_array[np_array>0.5])/len(np_array)


# dat = joblib.load('total_score_|freq_test')
  
# print(dat[:3])
# score = [i[3] for i in dat]
# # print(sorted(score[-10:]))
# for i in dat:
#     if i[3]>0.54: print(i)


if 1:
    # data = joblib.load('closingDataDict1', mmap_mode='r') # if you cant load-> # sudo s, # ulimit -n 65535

    comps  = joblib.load('data/'+'freq_test_companies')
    closeDataAll  = joblib.load('data/'+'freq_test_closeData2d')
    guess = joblib.load('data/'+'freq_test_guess_2d')
    if 1:
        co = []
        # correct_guess_list2d = []
        # close = []
        # for comp in data.keys():
        #     co.append(comp)
        #     correct_guess_list2d.append(np.array(data[comp]['correct_guesses']))
        #     close.append(data[comp]['dataClose'])
        
        # joblib.dump(co,'freq_test_companies')
        # joblib.dump(close,'freq_test_closeData2d')
        # joblib.dump(correct_guess_list2d,'freq_test_guess_2d')

        

        # print(co[0])
        # print(closeDataAll[0][0])
        # print(gue[0][0])

        # closeAll = np.concatenate(closeDataAll)
        # print(closeAll[:10])
        # diff = closeAll[:-1]-closeAll[1:]
        # print("np.mean(diff>0):", np.mean(diff>0)) # 0.46

    total_score = []

    for ran in range(1,20):
        
        for dec_small in np.linspace(0.1, 0.5, 5):
            for dec_big in np.linspace(0.5, 0.9, 5):
                
                total_freq      = []
                total_new_freq  = []

                score_freq = 0
                score_comp = 0

                for comp in range(len(comps)):
                    # print("\n\n",comp)

                    correct_guess = guess[comp]
                    correct_guess[correct_guess<0] = 0

                    # For looking at closeData in stead of guesses. Delete these three lines to go back to
                    closeData = np.array(closeDataAll[comp]) 
                    correct_guess = np.array(closeData[1:])-np.array(closeData[:-1])
                    correct_guess = correct_guess>0
                    # total_freq += list(correct_guess) 

                    # print("correct_guess[:10]:", correct_guess[:10])
                    # print("closeData[:10]:", closeData[:10])

                    if len(correct_guess[correct_guess>0])>0 and len(correct_guess)>50: # Avoid division by zero
                        correct_freq  = np.mean(correct_guess)
                        # print('correct_guess:', correct_guess)
                        # print('correct freq :    ', correct_freq)
                        # print('freq correct_guess', freq(np.array(correct_guess)))
                        new_guess = []
                        n = ran
                        # for i in list(range(len(correct_guess)))[n:]:
                        for i in list(range(len(closeData)))[n:-1]: # Added the -1 for closeData bc of errror.
                            if ((np.mean(correct_guess[i-n:i]) < dec_small) or (np.mean(correct_guess[i-n:i]) > dec_big)): # and (correct_guess[i-1] > 0.5) :
                                new_guess.append(correct_guess[i])
                        
                        total_new_freq += new_guess # Evaluation
                        
                        if len(new_guess)>0: freq_new_guess = np.mean(new_guess)
                        else: continue
                        
                        # if correct_freq > 0.5:
                        score_comp += 1
                        score_freq += freq_new_guess - correct_freq
                        
                        # print('new_guess:', new_guess)
                        # print('\nfreq_new_guess, correct_freq', freq_new_guess, correct_freq)
                        
                        # print('\nscore_freq:', score_freq)
                score = (dec_small, dec_big, ran, np.mean(total_new_freq))
                print("score:", score, np.mean(correct_guess), comp)
                total_score.append((dec_small, dec_big, ran, np.mean(total_new_freq), len(total_new_freq), np.mean(correct_guess), comp))
                # print('\n\n\n###\nscore_freq:', score_freq)    
                # print('\nscore_comp:         ', score_comp)    
                # print('\nscore_freq average: ', score_freq/score_comp)    

                # print('\ntotal_new_freq: ', np.mean(total_new_freq))    
                # print('\ntotal_freq: ', np.mean(total_freq))    
    joblib.dump(total_score, 'data/total_closeScore_|freq_test')
    
    # print(total_score)