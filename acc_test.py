import matplotlib.pyplot as plt, numpy as np, joblib ,pandas as pd
from scipy import stats
from helper_functions import get_local_minima_and_maxima
import lightgbm as lgb
# Import the yfinance. If you get module not found error the run !pip install yfiannce from your Jupyter notebook
import yfinance as yf  
import os
from datetime import datetime

# datetime object containing current date and time
now = datetime.now()
# dd/mm/YY H:M:S
dt_string = now.strftime("%Y-%m-%d-%H-%M-%S")

# command = 'find . -size -2M -exec cp -r {} '+f'{settings.back_up_dir} '+ '\;'
description = "" # "down_v_rel_Added_fee"
backup_folder = f'./backup/acc_test/{dt_string}{description}'


def calc_angle(x1,x2,x3):
    a = np.sqrt(np.abs(x2-x1)**2+1)
    b = np.sqrt(np.abs(x3-x2)**2+1)
    c = np.sqrt(2**2+np.abs(b**2-a**2))
    angle = np.arccos((a**2+b**2-c**2)/(2*a*b))
    return angle

def freq(np_array):
    return len(np_array[np_array>0.5])/len(np_array)


# dat = joblib.load('data/'+'total_score_|freq_test')
  
# print(dat[:3])
# score = [i[3] for i in dat]
# # print(sorted(score[-10:]))
# for i in dat:
#     if i[3]>0.54: print(i)

save = True
if save:

    data = joblib.load('data/'+'closingDataDict1', mmap_mode='r') # if you cant load-> # sudo -s, # ulimit -n 65535
    
    
    # comps  = joblib.load('freq_test_companies')
    # closeDataAll  = joblib.load('freq_test_closeData2d')
    # guess = joblib.load('freq_test_guess_2d')
    
    total_score = []

    sig_angles = []
    bkg_angles = []

    frequency = False

    dict_vars = {'angles':[], 'acc':[]}
    sig_acc = np.array([])
    bkg_acc = np.array([])

    sig_d_acc = np.array([])
    bkg_d_acc = np.array([])

    sig_dd_acc = np.array([])
    bkg_dd_acc = np.array([])

    sig_ddd_acc = np.array([])
    bkg_ddd_acc = np.array([])
    
    sig_dddd_acc = np.array([])
    bkg_dddd_acc = np.array([])
    
    sig_down_acc = []
    bkg_down_acc = []

    sig_freq = []
    bkg_freq = []

    sig_v_rel = []
    bkg_v_rel = []

    sig_down_v_rel = []
    bkg_down_v_rel = []

    for comp in data:
        # print(comp)
        # print("\n\n",comp)

        # correct_guess = guess[comp]
        # correct_guess[correct_guess<0] = 0

        # For looking at closeData in stead of guesses. Delete these three lines to go back to
        # closeData = np.array(closeDataAll[comp])
        
        closeData = np.array(data[comp]['dataClose'])
        # if np.mean(closeData)<250:
        #     continue

        # closeData = closeData - 0.02 # 0.02 is the saxobank fee for one stock purchase

        correct_guess = closeData[1:]-(closeData[:-1]+0.02)
        pct_change = correct_guess/closeData[:-1]

        correct_guess = correct_guess>0.000

        # print("np.mean(pct_change[correct_guess]):", np.mean(pct_change[correct_guess]))
        

        # total_freq += list(correct_guess) 

        # print("correct_guess[:10]:", correct_guess[:10])
        # print("closeData[:10]:", closeData[:10])

        v = np.array(closeData[1:])-np.array(closeData[:-1])
        acc = v[1:]-v[:-1]
        d_acc = acc[1:]-acc[:-1]
        dd_acc = d_acc[1:]-d_acc[:-1]
        ### All acc
        sig_acc = np.concatenate([sig_acc, acc[:-1][correct_guess[2:] ]]) 
        bkg_acc = np.concatenate([bkg_acc, acc[:-1][correct_guess[2:] < 0.5 ]]) 

        ### Change in acc
        sig_d_acc = np.concatenate([sig_d_acc, d_acc[:-1][correct_guess[3:] ]]) 
        bkg_d_acc = np.concatenate([bkg_d_acc, d_acc[:-1][correct_guess[3:] < 0.5 ]]) 

        ### delta*delta acc
        sig_dd_acc = np.concatenate([sig_dd_acc, dd_acc[:-1][correct_guess[4:] ]]) 
        bkg_dd_acc = np.concatenate([bkg_dd_acc, dd_acc[:-1][correct_guess[4:] < 0.5 ]]) 

        ### delta*delta acc
        ddd_acc = dd_acc[1:]-dd_acc[:-1]
        sig_ddd_acc = np.concatenate([sig_ddd_acc, ddd_acc[:-1][correct_guess[5:] ]]) 
        bkg_ddd_acc = np.concatenate([bkg_ddd_acc, ddd_acc[:-1][correct_guess[5:] < 0.5 ]]) 

        ### delta*delta acc
        dddd_acc = ddd_acc[1:]-ddd_acc[:-1]
        sig_dddd_acc = np.concatenate([sig_dddd_acc, dddd_acc[:-1][correct_guess[6:] ]]) 
        bkg_dddd_acc = np.concatenate([bkg_dddd_acc, dddd_acc[:-1][correct_guess[6:] < 0.5 ]]) 

        ### v1/v0 relationship
        v_rel = v[:-1]/v[1:]
        sig_v_rel = np.concatenate([sig_v_rel, v_rel[correct_guess[1:]]]) 
        bkg_v_rel = np.concatenate([bkg_v_rel, v_rel[correct_guess[1:]<0.5]])

        if 1:
            for i in range(len(closeData)-1): # Added the -1 for closeData bc of errror.
                
                if i < 3: continue
                
                ### Freq:
                if frequency: 
                    sig_freq, bkg_freq = freq(i, correct_guess, sig_freq, bkg_freq)

                

                ### Down acc
                if 1:
                    if v[i-2]<0 and v[i-1]<0: # and acc[i-1]>0:
                        if 0:
                            if correct_guess[i]:
                                sig_down_v_rel.append(v[i-2]/v[i-1])
                            else:
                                bkg_down_v_rel.append(v[i-2]/v[i-1])
                        if 1:
                            if correct_guess[i]:
                                sig_down_acc.append(acc[i-1])
                            else:
                                bkg_down_acc.append(acc[i-1])


                ### Angle   
                if 0:
                    angle = calc_angle(closeData[i-2], closeData[i-1], closeData[i])
                        
                    if correct_guess[i]:
                        sig_angles.append(angle)
                    else:
                        bkg_angles.append(angle)

    print("len(acc):",len(list(bkg_acc)+list(sig_acc)), 'len(down_acc)', len(list(bkg_down_acc)+list(sig_down_acc)))

    dict_vars['acc'] = [bkg_acc, sig_acc]
    dict_vars['down_acc'] = [bkg_down_acc, sig_down_acc]
    
    if frequency: 
        dict_vars['freq'] = [bkg_freq, sig_freq]

print('1')
if save:
    # joblib.dump([sig_angles, bkg_angles], "sig_and_bkg_angles")
    # joblib.dump(dict_vars, 'dict_vars')
    pass
else:
    # sig_angles, bkg_angles = joblib.load("sig_and_bkg_angles")
    dict_vars = joblib.load('dict_vars')
print('2')

#####

command1      = f'mkdir {backup_folder}'             ; os.system(command1)
command2      = f'cp ./acc_test.py {backup_folder}'  ; os.system(command2)

def plot_vars(bkg_var, sig_var, range0, range1):

    nbins = 50
    sig_hist, sig_edges = np.histogram(sig_var, bins=nbins,  range=[range0, range1])
    bkg_hist, bkg_edges = np.histogram(bkg_var, bins=nbins,  range=[range0, range1])

    # print("sig_hist:", sig_hist)
    # print("bkg_hist:", bkg_hist)
    # pct = sig_hist[sig_hist>0]/(sig_hist[sig_hist>0]+bkg_hist[bkg_hist>0])
    pct = sig_hist/(sig_hist+bkg_hist)


    print("len(sig_var):", len(sig_var))
    print("len(bkg_var):", len(bkg_var))

    fig, (ax1,ax2, ax3) = plt.subplots(1,3, figsize=[20,12])
    # ax1.hist(sig_var, bins=nbins, label='sig', alpha=0.6, range=[0, np.pi]) # For angles (range)
    # ax1.hist(bkg_var, bins=nbins, label='bkg', alpha=0.6, range=[0, np.pi]) # For angles (range)
    ax1.hist(sig_var, bins=nbins, label='sig', alpha=0.6, range=[range0, range1])
    ax1.hist(bkg_var, bins=nbins, label='bkg', alpha=0.6, range=[range0, range1])

    # print("pct:", pct)
 
    ax2.plot((bkg_edges[:-1]+bkg_edges[1:])/2, pct)
    ax2.set_xlim((range0, range1))
    ax2.axhline(y=0.5, color='r', linestyle='-')
    
    df = pd.DataFrame()
    df['v_rel'] = v_rel
    df['pct_change'] = pct_change[1:]
    df1 = df.sort_values('pct_change')
    ax3.hist2d(df['pct_change'], df1['v_rel'], bins = 75, range=[[-0.5,0.5],[-10,10]])

 
    ax1.legend()
    plt.savefig(f'{backup_folder}/hists_{nbins}.png')
    plt.show()

# bkg_acc, sig_acc           = dict_vars['acc']
# bkg_down_acc, sig_down_acc = dict_vars['down_acc']
# bkg_freq, sig_freq         = dict_vars['freq']

# i = -12

# bkg_freq17, sig_freq17 = list(map(lambda x: x[i] , bkg_freq)), list(map(lambda x: x[i] , sig_freq))

plot_vars(bkg_down_v_rel, sig_down_v_rel, -10, 10)

def freq(i, correct_guess, sig_freq, bkg_freq):
    if correct_guess[i] and i > 16:
        sig_freq.append([np.mean(correct_guess[i-17:i]), np.mean(correct_guess[i-16:i]), np.mean(correct_guess[i-15:i]),
                            np.mean(correct_guess[i-14:i]), np.mean(correct_guess[i-13:i]), np.mean(correct_guess[i-12:i]),
                            np.mean(correct_guess[i-11:i]), np.mean(correct_guess[i-10:i]), np.mean(correct_guess[i- 9:i]),
                            np.mean(correct_guess[i- 8:i]), np.mean(correct_guess[i- 7:i]), np.mean(correct_guess[i- 6:i]),
                            np.mean(correct_guess[i- 5:i]), np.mean(correct_guess[i- 4:i]), np.mean(correct_guess[i- 3:i]),
                            np.mean(correct_guess[i- 2:i]), np.mean(correct_guess[i- 1:i])])
    else:
        bkg_freq.append([np.mean(correct_guess[i-17:i]), np.mean(correct_guess[i-16:i]), np.mean(correct_guess[i-15:i]),
                            np.mean(correct_guess[i-14:i]), np.mean(correct_guess[i-13:i]), np.mean(correct_guess[i-12:i]),
                            np.mean(correct_guess[i-11:i]), np.mean(correct_guess[i-10:i]), np.mean(correct_guess[i- 9:i]),
                            np.mean(correct_guess[i- 8:i]), np.mean(correct_guess[i- 7:i]), np.mean(correct_guess[i- 6:i]),
                            np.mean(correct_guess[i- 5:i]), np.mean(correct_guess[i- 4:i]), np.mean(correct_guess[i- 3:i]),
                            np.mean(correct_guess[i- 2:i]), np.mean(closeData[i- 1:i])])
    return sig_freq, bkg_freq



command1      = f'mkdir {backup_folder}'             ; os.system(command1)
command2      = f'cp ./acc_test.py {backup_folder}'  ; os.system(command2)
