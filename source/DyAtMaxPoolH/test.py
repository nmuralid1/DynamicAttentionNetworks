"""
 This repository will house the pytorch code for
 LSTM Sequence forecasting Testing.
"""
from __future__ import print_function
import torch
import torch.nn as nn
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
import time
import math
import argparse
import logging
import os
import pickle
import sys
sys.path.append('../')
sys.path.append('../../')
from helpers import create_data_sequences,preprocess_targets_data,calculate_mse_tensor,_3D_pytorch_to_2D_pandas,plot_forecasts,makedirs
from rnn import EncoderHierAttn, DecoderHierAttn
from model_train_eval import testIters


torch.manual_seed(123)
#SET UP ARGUMENT PARSER
parser=argparse.ArgumentParser()
parser.add_argument("datainputdir")
parser.add_argument("rnnobject")
parser.add_argument("logoutputfile")
parser.add_argument("loglevel")
parser.add_argument("predictionsoutputdir")
parser.add_argument("hiddensize")
parser.add_argument("batchsize")
parser.add_argument("numlayers")
parser.add_argument("sequencelength")
parser.add_argument("slidingattention")
parser.add_argument("sostoken")
parser.add_argument("dropout")
parser.add_argument("teacherforcing")
parser.add_argument("iternum")
parser.add_argument("datasetname")
parser.add_argument("hierattnmethod")
parser.add_argument("slidingwindowsize")

#PARSE ARGUMENTS
args = parser.parse_args()
inputdir=args.datainputdir

#SETUP LOGGING
if args.loglevel=="WARNING":
	loglevel=logging.WARNING
elif args.loglevel=="INFO":
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

mse_overall_test=list()

#DATASET PARAMETERS
if "tep" in args.datasetname:
	columnsofinterest = ['MEAS_A_Feed', 'MEAS_D_Feed', 'MEAS_E_Feed', 'MEAS_A_C Feed', 'MEAS_Recycle_flow', 'MEAS_Reactor_feed', 'MEAS_Reactor_pressure', 'MEAS_Reactor_level', 'MEAS_Reactor_temperature', 'MEAS_Purge_rate', 'MEAS_Sep_temperature', 'MEAS_Sep_level', 'MEAS_Sep_pressure', 'MEAS_Sep_underflow', 'MEAS_Stripper_level', 'MEAS_Stripper_pressure', 'MEAS_Stripper_underfow', 'MEAS_Stripper_temperature', 'MEAS_Steam_flow', 'MEAS_Compressor_work', 'MEAS_Reactor_cool_temperature', 'MEAS_Condo_cool_temperature', 'MEAS_Feed_A', 'MEAS_Feed_B', 'MEAS_Feed_C', 'MEAS_Feed_D', 'MEAS_Feed_E', 'MEAS_Feed_F', 'MEAS_Purge_A', 'MEAS_Purge_B', 'MEAS_Purge_C', 'MEAS_Purge_D', 'MEAS_Purge_E', 'MEAS_Purge_F', 'MEAS_Purge_G', 'MEAS_Purge_H', 'MEAS_Product_D', 'MEAS_Product_E', 'MEAS_Product_F', 'MEAS_Product_G', 'MEAS_Product_H']

elif "ghl" in args.datasetname:
	columnsofinterest = ['RT_level_ini','RT_temperature.T','HT_temperature.T','inj_valve_act','heater_act']

elif args.datasetname=="electricityloaddiagrams":
	columnsofinterest=["MT_001","MT_002","MT_003","MT_004","MT_005","MT_006","MT_007","MT_008","MT_009","MT_010","MT_011","MT_012","MT_013","MT_014","MT_015","MT_016","MT_017","MT_018","MT_019","MT_020"]


#HYPERPARAMETERS
HIDDEN_SIZE = int(args.hiddensize)
INPUT_SIZE  = len(columnsofinterest)  #Number of timeseries
OUTPUT_SIZE = INPUT_SIZE
NUM_LAYERS  = int(args.numlayers)
SOS_TOKEN=int(args.sostoken)  #We must make sure this is a symbol that isn't present in our dataset.
ENCODER_SEQUENCE_LENGTH=int(args.sequencelength)
DECODER_SEQUENCE_LENGTH=ENCODER_SEQUENCE_LENGTH
DROPOUT=float(args.dropout)
RESULTS_OUTPUT_DIR_NAME=args.predictionsoutputdir+"/"+"SEQUENCE_LENGTH_{}_NUMLAYERS_{}_HIDDEN_SIZE_{}_DROPOUT_{}_TEACHERFORCING_{}_RNNOBJECT_{}_ITERNUM_{}".format(ENCODER_SEQUENCE_LENGTH,NUM_LAYERS,HIDDEN_SIZE,DROPOUT,args.teacherforcing,args.rnnobject,args.iternum)+"/"
MODEL_OUTPUT_DIR=RESULTS_OUTPUT_DIR_NAME+"model/"
SLIDINGATTENTION=json.loads(args.slidingattention.lower())
HIERARCHICAL_ATTN_METHOD=args.hierattnmethod
SLIDING_WINDOWSIZE=int(args.slidingwindowsize)

# ITERATE OVER MULTIPLE TEST FILES.
root.info("inputdir = {}".format(inputdir))
root.info("Files = {}".format(glob.glob(inputdir+"*.csv")))
for idx,testfile in enumerate(sorted(glob.glob(inputdir+"*.csv"))):
	if "test_" not in testfile:
		continue

	#parse csv file.
	data_df = pd.read_csv(testfile)
	root.info("Test File Name = {}, Test Data Size = {}".format(testfile, data_df.shape))
	data_df = data_df[columnsofinterest]
		
	#Data Normalization
	mean_norm_data_df=(data_df - data_df.mean(axis=0))/(data_df.max(axis=0) - data_df.min(axis=0))

	root.info("columns in input df = {}".format(columnsofinterest))
	
	#Preprocess Time-Series Data.
	data_df,data = create_data_sequences(mean_norm_data_df,sequence_length=ENCODER_SEQUENCE_LENGTH,slidingwindow=False)
	test_input = Variable(torch.from_numpy(data), requires_grad=False)  #n  X m  for each of the 97 rows, sequence values range from (0 - n-1)
	test_input = test_input.float()
	
	root.info("Test File name = {}, Preprocessed Test input size {}, Test Input size before preprocessing = {}".format(testfile,test_input.size(),mean_norm_data_df.shape[0]))
	
	#HYPERPARAMETERS 2.
	NUM_INSTANCES=data_df.shape[0]
	BATCHSIZE=int(args.batchsize)
	BATCHSIZE=min(NUM_INSTANCES,BATCHSIZE)
	NUM_BATCHES=int(NUM_INSTANCES/BATCHSIZE)
	root.info("Results output dir path: {}".format(RESULTS_OUTPUT_DIR_NAME))
	root.info("Sequence Length = {}, TOTAL NUMBER OF INSTANCES = {}".format(ENCODER_SEQUENCE_LENGTH,NUM_INSTANCES))

	#Check whether figure directories exist. If not create them.
	TESTOUTPUTDIR=RESULTS_OUTPUT_DIR_NAME+"results_test/"+os.path.basename(testfile).split(".csv")[0]
	makedirs(TESTOUTPUTDIR)
	TESTOUTPUTFILE=TESTOUTPUTDIR+"/"+"test_mse.txt"
	
	root.info("Results output dir path: {}".format(RESULTS_OUTPUT_DIR_NAME))
	root.info("Sequence Length = {}, TOTAL NUMBER OF INSTANCES = {}".format(ENCODER_SEQUENCE_LENGTH,NUM_INSTANCES))

	#Instantiate Attention Encoder
	encoder = EncoderHierAttn(INPUT_SIZE, HIDDEN_SIZE,ENCODER_SEQUENCE_LENGTH,NUM_LAYERS,args.rnnobject,root)
	encoder.load_state_dict(torch.load(MODEL_OUTPUT_DIR+"encoder.pt"))
	encoder.eval()
	encoder.cuda() ##CUDA

	#Instantiate Attention Decoder
	decoder = DecoderHierAttn(INPUT_SIZE,HIDDEN_SIZE,NUM_LAYERS,OUTPUT_SIZE,args.rnnobject,ENCODER_SEQUENCE_LENGTH,DECODER_SEQUENCE_LENGTH,SLIDING_WINDOWSIZE,DROPOUT,root)
	decoder.load_state_dict(torch.load(MODEL_OUTPUT_DIR+"decoder.pt"))
	decoder.eval()
	decoder.cuda() ##CUDA
	
	#Run Testing and retrive MSE, Smoothed MSE values.
	mse_per_timestep,predictions,targets,attention_weights,hiddenstsimmat=testIters(encoder,decoder,test_input,NUM_BATCHES,NUM_LAYERS,SOS_TOKEN,float(args.teacherforcing),root,SLIDINGATTENTION,HIERARCHICAL_ATTN_METHOD)

	#SAVE Attention Weights To File.
	pickle.dump(attention_weights.cpu().detach().numpy(),open(MODEL_OUTPUT_DIR+"attention_weights_test.pkl","wb"))
	pickle.dump(hiddenstsimmat.cpu().detach().numpy(),open(MODEL_OUTPUT_DIR+"hierHidSt_AttnHidSt_Ctxvec_Sim.pkl","wb"))

	root.info("Shape of MSE Per Timestep = {}, Shape of prediction = {}, Shape of targets = {}".format(len(mse_per_timestep),predictions.size(),targets.size()))
	mse_overall_test.append(np.mean(mse_per_timestep)) #Get the global mean squared error per test set.
		
	root.info("Shape of mse_per_sequence = {}".format(len(mse_per_timestep)))
	
	#HIDDEN_SIZE,SEQUENCE_LENGTH,RNN-TYPE,DROPOUT,TEACHER_FORCING_RATIO,MSE
	root.info("Output file being written to: {}".format(TESTOUTPUTFILE))
	root.info("Output String = {}".format(str(HIDDEN_SIZE)+","+str(ENCODER_SEQUENCE_LENGTH)+","+str(args.rnnobject)+","+str(DROPOUT)+","+str(np.mean(mse_per_timestep))))
	with open(TESTOUTPUTFILE,"a") as g:
		g.write(str(HIDDEN_SIZE)+","+str(ENCODER_SEQUENCE_LENGTH)+","+str(args.rnnobject)+","+str(DROPOUT)+","+str(args.teacherforcing)+","+str(np.mean(mse_per_timestep))+"\n")

	test_predictions_df=_3D_pytorch_to_2D_pandas(predictions.cpu().detach().numpy(),seqdim=1,column_names=columnsofinterest,groupby="time_order")
	test_predictions_df.to_csv(TESTOUTPUTDIR+"/timeordered_test_predictions.csv",index=False)	

	test_targets_df=_3D_pytorch_to_2D_pandas(targets.cpu().numpy(),seqdim=1,column_names=columnsofinterest,groupby="time_order")
	test_targets_df.to_csv(TESTOUTPUTDIR+"/timeordered_test_targets.csv",index=False)

	#Set all torch tensors to None.
	attention_weights=None
	hiddenstsimmat=None
	predictions=None
	targets=None
	torch.cuda.empty_cache()	
	plot_forecasts(test_predictions_df,test_targets_df,TESTOUTPUTDIR+"/")
