import glob,os,sys
import numpy as np
import pandas as pd

#Run this file after the training and testing has successfully been run for a particular model and particular dataset for different sequence lengths.


def calculate_mse_wmse(resultsdir):
    #@param resultsdir should be a string containing /full/path/to/ijcai_19_code_and_data/results/ghl_small/DyAtH/DyAtH_mean/
    SEQUENCELENGTH=int(sys.argv[2])  #10,90,110,130
    print(resultsdir,SEQUENCELENGTH)
    for item in glob.glob(resultsdir+"/SEQUENCE_LENGTH_{}_*ITERNUM_1/".format(SEQUENCELENGTH)):
            mse=list()
            wmse=list()
            for testdir in glob.glob(item+"results_test/*"):
                pred=pd.read_csv(testdir+"/timeordered_test_predictions.csv")
                tgts=pd.read_csv(testdir+"/timeordered_test_targets.csv")
                print(pred.as_matrix().shape,tgts.as_matrix().shape)
                assert pred.shape==tgts.shape,"Predictions and targets should be of same shape."
                predarr=np.reshape(pred.as_matrix(),newshape=(int(pred.shape[0]/SEQUENCELENGTH),SEQUENCELENGTH,pred.shape[1]))
                tgtarr=np.reshape(tgts.as_matrix(),newshape=(int(tgts.shape[0]/SEQUENCELENGTH),SEQUENCELENGTH,tgts.shape[1]))
                _mse=np.mean(np.square(predarr - tgtarr))
                _seqMSE=np.square(predarr - tgtarr).mean(axis=2).mean(axis=0)
                _wtmse=np.average(_seqMSE,weights=[i/SEQUENCELENGTH for i in range(1,SEQUENCELENGTH+1)])
                print("Dir = {} MSE = {}, Weighted MSE = {}".format(testdir,_mse,_wtmse))
                mse.append(_mse)
                wmse.append(_wtmse)

            print("SeqLen = {} Avg. MSE = {} Avg. WMSE = {}\n\n".format(SEQUENCELENGTH,np.mean(mse),np.mean(wmse)))

