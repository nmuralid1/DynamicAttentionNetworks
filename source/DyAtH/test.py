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

torch.manual_seed(123) #Set random seed for reproducibility.
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

"""
PREPROCESS TIMESERIES DATA. 
 If you want to run the model on a new dataset and select a sub-set of columns, add a condition to the following set of conditional statements 

 `elif "newdatasetsubstring" in args.datasetname:  
    ...Do Something...`

This needs to be the same snippet added to the corresponding train.py file as the subset of columns selected in both sets is assumed to be the same.
"""

if "tep" in args.datasetname:
	columnsofinterest = ['MEAS_A_Feed', 'MEAS_D_Feed', 'MEAS_E_Feed', 'MEAS_A_C Feed', 'MEAS_Recycle_flow', 'MEAS_Reactor_feed', 'MEAS_Reactor_pressure', 'MEAS_Reactor_level', 'MEAS_Reactor_temperature', 'MEAS_Purge_rate', 'MEAS_Sep_temperature', 'MEAS_Sep_level', 'MEAS_Sep_pressure', 'MEAS_Sep_underflow', 'MEAS_Stripper_level', 'MEAS_Stripper_pressure', 'MEAS_Stripper_underfow', 'MEAS_Stripper_temperature', 'MEAS_Steam_flow', 'MEAS_Compressor_work', 'MEAS_Reactor_cool_temperature', 'MEAS_Condo_cool_temperature', 'MEAS_Feed_A', 'MEAS_Feed_B', 'MEAS_Feed_C', 'MEAS_Feed_D', 'MEAS_Feed_E', 'MEAS_Feed_F', 'MEAS_Purge_A', 'MEAS_Purge_B', 'MEAS_Purge_C', 'MEAS_Purge_D', 'MEAS_Purge_E', 'MEAS_Purge_F', 'MEAS_Purge_G', 'MEAS_Purge_H', 'MEAS_Product_D', 'MEAS_Product_E', 'MEAS_Product_F', 'MEAS_Product_G', 'MEAS_Product_H']

elif "ghl" in args.datasetname:
	columnsofinterest = ['RT_level_ini','RT_temperature.T','HT_temperature.T','inj_valve_act','heater_act']

elif args.datasetname=="electricityloaddiagrams":
	columnsofinterest=["MT_001","MT_002","MT_003","MT_004","MT_005","MT_006","MT_007","MT_008","MT_009","MT_010","MT_011","MT_012","MT_013","MT_014","MT_015","MT_016","MT_017","MT_018","MT_019","MT_020"]
	#"MT_021","MT_022","MT_023","MT_024","MT_025","MT_026","MT_027","MT_028","MT_029","MT_030","MT_031","MT_032","MT_033","MT_034","MT_035","MT_036","MT_037","MT_038","MT_039","MT_040","MT_041","MT_042","MT_043","MT_044","MT_045","MT_046","MT_047","MT_048","MT_049","MT_050","MT_051","MT_052","MT_053","MT_054","MT_055","MT_056","MT_057","MT_058","MT_059","MT_060","MT_061","MT_062","MT_063","MT_064","MT_065","MT_066","MT_067","MT_068","MT_069","MT_070","MT_071","MT_072","MT_073","MT_074","MT_075","MT_076","MT_077","MT_078","MT_079","MT_080","MT_081","MT_082","MT_083","MT_084","MT_085","MT_086","MT_087","MT_088","MT_089","MT_090","MT_091","MT_092","MT_093","MT_094","MT_095","MT_096","MT_097","MT_098","MT_099","MT_100","MT_101","MT_102","MT_103","MT_104","MT_105","MT_106","MT_107","MT_108","MT_109","MT_110","MT_111","MT_112","MT_113","MT_114","MT_115","MT_116","MT_117","MT_118","MT_119","MT_120","MT_121","MT_122","MT_123","MT_124","MT_125","MT_126","MT_127","MT_128","MT_129","MT_130","MT_131","MT_132","MT_133","MT_134","MT_135","MT_136","MT_137","MT_138","MT_139","MT_140","MT_141","MT_142","MT_143","MT_144","MT_145","MT_146","MT_147","MT_148","MT_149","MT_150","MT_151","MT_152","MT_153","MT_154","MT_155","MT_156","MT_157","MT_158","MT_159","MT_160","MT_161","MT_162","MT_163","MT_164","MT_165","MT_166","MT_167","MT_168","MT_169","MT_170","MT_171","MT_172","MT_173","MT_174","MT_175","MT_176","MT_177","MT_178","MT_179","MT_180","MT_181","MT_182","MT_183","MT_184","MT_185","MT_186","MT_187","MT_188","MT_189","MT_190","MT_191","MT_192","MT_193","MT_194","MT_195","MT_196","MT_197","MT_198","MT_199","MT_200","MT_201","MT_202","MT_203","MT_204","MT_205","MT_206","MT_207","MT_208","MT_209","MT_210","MT_211","MT_212","MT_213","MT_214","MT_215","MT_216","MT_217","MT_218","MT_219","MT_220","MT_221","MT_222","MT_223","MT_224","MT_225","MT_226","MT_227","MT_228","MT_229","MT_230","MT_231","MT_232","MT_233","MT_234","MT_235","MT_236","MT_237","MT_238","MT_239","MT_240","MT_241","MT_242","MT_243","MT_244","MT_245","MT_246","MT_247","MT_248","MT_249","MT_250","MT_251","MT_252","MT_253","MT_254","MT_255","MT_256","MT_257","MT_258","MT_259","MT_260","MT_261","MT_262","MT_263","MT_264","MT_265","MT_266","MT_267","MT_268","MT_269","MT_270","MT_271","MT_272","MT_273","MT_274","MT_275","MT_276","MT_277","MT_278","MT_279","MT_280","MT_281","MT_282","MT_283","MT_284","MT_285","MT_286","MT_287","MT_288","MT_289","MT_290","MT_291","MT_292","MT_293","MT_294","MT_295","MT_296","MT_297","MT_298","MT_299","MT_300","MT_301","MT_302","MT_303","MT_304","MT_305","MT_306","MT_307","MT_308","MT_309","MT_310","MT_311","MT_312","MT_313","MT_314","MT_315","MT_316","MT_317","MT_318","MT_319","MT_320","MT_321","MT_322","MT_323","MT_324","MT_325","MT_326","MT_327","MT_328","MT_329","MT_330","MT_331","MT_332","MT_333","MT_334","MT_335","MT_336","MT_337","MT_338","MT_339","MT_340","MT_341","MT_342","MT_343","MT_344","MT_345","MT_346","MT_347","MT_348","MT_349","MT_350","MT_351","MT_352","MT_353","MT_354","MT_355","MT_356","MT_357","MT_358","MT_359","MT_360","MT_361","MT_362","MT_363","MT_364","MT_365","MT_366","MT_367","MT_368","MT_369","MT_370"]

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
	decoder = DecoderHierAttn(INPUT_SIZE,HIDDEN_SIZE,NUM_LAYERS,OUTPUT_SIZE,args.rnnobject,ENCODER_SEQUENCE_LENGTH,DECODER_SEQUENCE_LENGTH,DROPOUT,root)
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
