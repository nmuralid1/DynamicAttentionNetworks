"""
 This repository will house the pytorch code for
 LSTM Sequence forecasting.
"""
from __future__ import print_function
import torch
import torch.nn as nn
torch.manual_seed(123)
torch.backends.cudnn.deterministic=True
torch.backends.cudnn.benchmark=False
from torch.autograd import Variable
import torch.optim as optim
import random
import torch.nn.functional as F
import numpy as np
import matplotlib
matplotlib.use('agg')
import matplotlib.pyplot as plt
import pandas as pd
import glob
import json
from datetime import datetime
import math
import logging
import argparse
import sys
import os
import pickle
sys.path.append('../')
sys.path.append('../../')
from helpers import preprocess_timeseries_data,calculate_mse,_3D_pytorch_to_2D_pandas,plot_forecasts,makedirs
from model_train_eval import trainIters
from rnn import EncoderHierAttn,DecoderHierAttn

torch.manual_seed(123) 
#CREATE ARGUMENT PARSER.
parser=argparse.ArgumentParser()
parser.add_argument("datainputdir")
parser.add_argument("trainingfilename")
parser.add_argument("rnnobject")
parser.add_argument("logoutputfile")
parser.add_argument("loglevel")
parser.add_argument("predictionsoutputdir")
parser.add_argument("hiddensize")
parser.add_argument("batchsize")
parser.add_argument("numlayers")
parser.add_argument("teacherforcing")
parser.add_argument("learningrate")
parser.add_argument("sequencelength")
parser.add_argument("slidingattention")
parser.add_argument("sostoken")
parser.add_argument("numepochs")
parser.add_argument("dropout")
parser.add_argument("iternum")
parser.add_argument("datasetname")
parser.add_argument("hierattnmethod")
parser.add_argument("slidingwindowsize")

#PARSE ARGUMENTS
args = parser.parse_args()

#SETUP LOGGING
if args.loglevel=='WARNING':
	loglevel=logging.WARNING
elif args.loglevel=='INFO':
	loglevel=logging.INFO
else:
	loglevel=logging.DEBUG

root = logging.getLogger()
root.setLevel(loglevel)
ch = logging.StreamHandler(sys.stdout)
ch.setLevel(loglevel)
formatter = logging.Formatter('%(levelname)s - %(message)s')
ch.setFormatter(formatter)
root.addHandler(ch)
print("Log output file = {}, level = {}".format(args.logoutputfile,loglevel))

#LOAD TRAINING DATA.
trainfile=args.datainputdir+args.trainingfilename

# HYPERPARAMETERS
HIDDEN_SIZE = int(args.hiddensize)
NUM_LAYERS  = int(args.numlayers)
TEACHER_FORCING_RATIO = float(args.teacherforcing) #For now we will not use any teacher forcing.
LEARNING_RATE=float(args.learningrate)
ENCODER_SEQUENCE_LENGTH=int(args.sequencelength)
DECODER_SEQUENCE_LENGTH=ENCODER_SEQUENCE_LENGTH
SOS_TOKEN=int(args.sostoken)  #We must make sure this is a symbol that isn't present in our dataset.
NUM_EPOCHS=int(args.numepochs)
BATCH_SIZE=int(args.batchsize)
DROPOUT=float(args.dropout)
SLIDING_ATTENTION=json.loads(args.slidingattention.lower())
HIERARCHICAL_ATTN_METHOD=args.hierattnmethod
SLIDING_WINSIZE=int(args.slidingwindowsize)

# CREATE
RESULTS_OUTPUT_DIR_NAME=args.predictionsoutputdir+"/"+"SEQUENCE_LENGTH_{}_NUMLAYERS_{}_HIDDEN_SIZE_{}_DROPOUT_{}_TEACHERFORCING_{}_RNNOBJECT_{}_ITERNUM_{}".format(ENCODER_SEQUENCE_LENGTH,NUM_LAYERS,HIDDEN_SIZE,DROPOUT,args.teacherforcing,args.rnnobject,args.iternum)+"/"
print("Results Output Dir Name = {}".format(RESULTS_OUTPUT_DIR_NAME))
MODEL_OUTPUT_DIR=RESULTS_OUTPUT_DIR_NAME+"model/"
VALIDATION_OUTPUT_FILE=RESULTS_OUTPUT_DIR_NAME+"validation_mse.txt"
#Create figures train and validation directories.
print("Model_output_dir = {}".format(MODEL_OUTPUT_DIR))
makedirs(MODEL_OUTPUT_DIR)
makedirs(RESULTS_OUTPUT_DIR_NAME+"figures_train")
makedirs(RESULTS_OUTPUT_DIR_NAME+"figures_validation")

root.info("Results output dir path: {}".format(RESULTS_OUTPUT_DIR_NAME))


##PREPROCESS TIMESERIES DATA.
if "tep" in args.datasetname:
	#columnsofinterest=['MEAS_A_Feed', 'MEAS_D_Feed', 'MEAS_E_Feed', 'MEAS_A_C Feed','MEAS_Recycle_flow', 'MEAS_Reactor_feed', 'MEAS_Reactor_pressure','MEAS_Reactor_level', 'MEAS_Reactor_temperature', 'MEAS_Purge_rate']
	columnsofinterest=['MEAS_A_Feed', 'MEAS_D_Feed', 'MEAS_E_Feed', 'MEAS_A_C Feed', 'MEAS_Recycle_flow', 'MEAS_Reactor_feed', 'MEAS_Reactor_pressure', 'MEAS_Reactor_level', 'MEAS_Reactor_temperature', 'MEAS_Purge_rate', 'MEAS_Sep_temperature', 'MEAS_Sep_level', 'MEAS_Sep_pressure', 'MEAS_Sep_underflow', 'MEAS_Stripper_level', 'MEAS_Stripper_pressure', 'MEAS_Stripper_underfow', 'MEAS_Stripper_temperature', 'MEAS_Steam_flow', 'MEAS_Compressor_work', 'MEAS_Reactor_cool_temperature', 'MEAS_Condo_cool_temperature', 'MEAS_Feed_A', 'MEAS_Feed_B', 'MEAS_Feed_C', 'MEAS_Feed_D', 'MEAS_Feed_E', 'MEAS_Feed_F', 'MEAS_Purge_A', 'MEAS_Purge_B', 'MEAS_Purge_C', 'MEAS_Purge_D', 'MEAS_Purge_E', 'MEAS_Purge_F', 'MEAS_Purge_G', 'MEAS_Purge_H', 'MEAS_Product_D', 'MEAS_Product_E', 'MEAS_Product_F', 'MEAS_Product_G', 'MEAS_Product_H']
	data_df,data = preprocess_timeseries_data(trainfile,sequence_length=ENCODER_SEQUENCE_LENGTH,slidingwindow=False,columnsofinterest=columnsofinterest)
	df_columns=columnsofinterest
elif "ghl" in args.datasetname:
	columnsofinterest=['RT_level_ini','RT_temperature.T','HT_temperature.T','inj_valve_act','heater_act']
	df_columns=columnsofinterest
	data_df,data = preprocess_timeseries_data(trainfile,sequence_length=ENCODER_SEQUENCE_LENGTH,slidingwindow=False,columnsofinterest=columnsofinterest)

elif args.datasetname=="electricityloaddiagrams":
	columnsofinterest=["MT_001","MT_002","MT_003","MT_004","MT_005","MT_006","MT_007","MT_008","MT_009","MT_010","MT_011","MT_012","MT_013","MT_014","MT_015","MT_016","MT_017","MT_018","MT_019","MT_020"]	
	data_df,data=preprocess_timeseries_data(trainfile,sequence_length=ENCODER_SEQUENCE_LENGTH,slidingwindow=False,columnsofinterest=columnsofinterest)
	df_columns=columnsofinterest


INPUT_SIZE  = int(len(data_df.columns)/ENCODER_SEQUENCE_LENGTH)  #Number of timeseries
OUTPUT_SIZE = INPUT_SIZE
TRAINENDIDX=int(0.80*data_df.shape[0])
BATCH_SIZE=min(TRAINENDIDX,BATCH_SIZE)
NUM_BATCHES=int(TRAINENDIDX/BATCH_SIZE) #We want not more than 500 instances per batch so this is how we calculate numbatches.


root.info("Sequence Length = {}, TRAINENDIDX = {}, TOTAL NUMBER OF INSTANCES = {}, NUM BATCHES = {}".format(ENCODER_SEQUENCE_LENGTH,TRAINENDIDX,data_df.shape[0],NUM_BATCHES))
root.info("INPUTSIZE = {}, OUTPUTSIZE = {}".format(INPUT_SIZE,OUTPUT_SIZE))
root.info("columns in input df = {}".format(data_df.columns))
root.info("Experiment Info : Seqence Length = {}, Hidden Size = {}, Teacher Forcing = {}, Dropout = {}, Iter Num = {}".format(args.sequencelength,args.hiddensize,args.teacherforcing,args.dropout,args.iternum))

#ENCODER, DECODER INSTANTIATION
encoder = EncoderHierAttn(INPUT_SIZE, HIDDEN_SIZE,ENCODER_SEQUENCE_LENGTH, NUM_LAYERS,args.rnnobject,root)
encoder.cuda() ##CUDA
decoder = DecoderHierAttn(INPUT_SIZE,HIDDEN_SIZE,NUM_LAYERS,OUTPUT_SIZE,args.rnnobject,ENCODER_SEQUENCE_LENGTH,DECODER_SEQUENCE_LENGTH,SLIDING_WINSIZE,DROPOUT,root)
decoder.cuda() ##CUDA

train_input = Variable(torch.from_numpy(data[:TRAINENDIDX,:]), requires_grad=False)  #n  X m  for each of the 97 rows, sequence values range from (0 - n-1)
train_input = train_input.float()
validation_input = Variable(torch.from_numpy(data[TRAINENDIDX:,:]), requires_grad=False)  #3 X m for each of the 3 rows, sequence values range from (0 - n-1)
validation_input = validation_input.float()
root.info("Train input size {}, validation input size {}".format(train_input.size(),validation_input.size()))

#TRAINING + VALIDATION
losses,val_losses,validation_mse,train_predictions,train_targets,validation_predictions,validation_targets,attention_weights=trainIters(encoder,decoder,train_input,validation_input,NUM_BATCHES, NUM_LAYERS,TEACHER_FORCING_RATIO,NUM_EPOCHS,SOS_TOKEN,root,SLIDING_ATTENTION,HIERARCHICAL_ATTN_METHOD,learning_rate=LEARNING_RATE)
root.info("After training and validation , before writing predictions to output file \n\n")

#CONVERT TRAIN_PREDICTIONS, TRAIN_TARGETS, VALIDATION_PREDICTIONS, VALIDATION_TARGETS TO DATAFRAME AND WRITE TO SEPARATE CSV FILES FOR EVALUATION LATER ON.

train_pred_df=_3D_pytorch_to_2D_pandas(train_predictions.cpu().detach().numpy(),seqdim=1,column_names=df_columns,groupby="time_order") #.detach() is used to be able to convert variables to numpy arrays whose requires_grad flag is set to True.
train_pred_df.to_csv(RESULTS_OUTPUT_DIR_NAME+"timeordered_train_predictions.csv",index=False)

train_targets_df=_3D_pytorch_to_2D_pandas(train_targets.cpu().numpy(),seqdim=1,column_names=df_columns,groupby="time_order")
train_targets_df.to_csv(RESULTS_OUTPUT_DIR_NAME+"timeordered_train_targets.csv",index=False)

validation_predictions_df=_3D_pytorch_to_2D_pandas(validation_predictions.cpu().detach().numpy(),seqdim=1,column_names=df_columns,groupby="time_order")
validation_predictions_df.to_csv(RESULTS_OUTPUT_DIR_NAME+"timeordered_validation_predictions.csv",index=False)
validation_targets_df=_3D_pytorch_to_2D_pandas(validation_targets.cpu().numpy(),seqdim=1,column_names=df_columns,groupby="time_order")
validation_targets_df.to_csv(RESULTS_OUTPUT_DIR_NAME+"timeordered_validation_targets.csv",index=False)

root.debug("attention weights shape = {}".format(attention_weights.size()))

#ATTENTION WEIGHTS
torch.save(attention_weights,MODEL_OUTPUT_DIR+'attention_weights_val_with_teacherforcing.pt')
pickle.dump(attention_weights.cpu().detach().numpy(),open(MODEL_OUTPUT_DIR+"attention_weights_val.pkl","wb"))

####################  PLOTTING ##################
#Check whether figure directories exist. If not create them.

# PLOT TRAINING AND VALIDATION FORECASTS.
plot_forecasts(train_pred_df.iloc[:10000],train_targets_df.iloc[:10000],RESULTS_OUTPUT_DIR_NAME+"figures_train/")
plot_forecasts(validation_predictions_df,validation_targets_df,RESULTS_OUTPUT_DIR_NAME+"figures_validation/")

# PLOT LOSSES
fig,ax=plt.subplots(1,1,figsize=(20,6))
ax.plot(losses)
ax.set_title("Training loss",fontsize=20)
fig.savefig(RESULTS_OUTPUT_DIR_NAME+"figures_train/"+"trainingloss.png")

fig,ax=plt.subplots(1,1,figsize=(20,6))
ax.plot(val_losses)
ax.set_title("Validation loss",fontsize=20)
fig.savefig(RESULTS_OUTPUT_DIR_NAME+"figures_validation/"+"validationloss.png")

################ END PLOTTING ###############

######## SAVE STATE ##############

#WRITE TRAINING & VALIDATION LOSS IN THE FOLLOWING FORMAT: HiddenSize,NumEpochs,SequenceLength,NumberTrainingInstances,RNN-Type,Dropoutpercentage,Teacher Forcing Ratio,Validation MSE
with open(VALIDATION_OUTPUT_FILE,"a") as f:
	f.write(str(HIDDEN_SIZE)+","+str(NUM_EPOCHS)+","+str(ENCODER_SEQUENCE_LENGTH)+","+str(train_input.data.numpy().shape[0])+","+args.rnnobject+","+str(DROPOUT)+","+str(TEACHER_FORCING_RATIO)+","+str(validation_mse)+"\n")

f.close()

#SAVE MODEL
root.info("saving model encoder")
encodersavepath=MODEL_OUTPUT_DIR+"encoder.pt"
torch.save(encoder.state_dict(),encodersavepath)
root.info("saving model decoder")
decodersavepath=MODEL_OUTPUT_DIR+"decoder.pt"
torch.save(decoder.state_dict(),decodersavepath)
############## END SAVE STATE ##############################

print("Model Paramters: = Hidden Size = {}, Num Epochs = {}, Sequence Length = {}, Num Train Instances = {}, RNN Object = {}, DROPOUT = {}, VALIDATION MSE = {}".format(str(HIDDEN_SIZE),str(NUM_EPOCHS),str(ENCODER_SEQUENCE_LENGTH),str(train_input.data.numpy().shape[0]),args.rnnobject,str(DROPOUT),str(validation_mse)))
