#!/bin/bash

MODELTYPE="DyAtH"
SOURCEDIR="$SOURCEPARDIR"/"$MODELTYPE" 
HIERARCHICAL_ATTN_METHOD="mean"
PREDICTIONSOUTPUTPARDIR="$PREDICTIONSOUTPUTPARDIR"/"$MODELTYPE"/ 
#LOGOUTPUTFILEPARDIR="$LOGOUTPUTFILEPARDIR"/ #Add /full/path/to/logoutputfile/ .
  
MODELTYPE=$MODELTYPE"_"$HIERARCHICAL_ATTN_METHOD  #Model Type update.
PREDICTIONSOUTPUTDIR="$PREDICTIONSOUTPUTPARDIR"/"$MODELTYPE"
LOGOUTPUTFILE="$LOGOUTPUTFILEPARDIR"/"$MODELTYPE/logs/encoder_decoder_regular_test.log"

#model parameters
HIDDENSIZE=$1  
NUMLAYERS=1
TEACHERFORCING=$4
SEQUENCELENGTH=$2   #One of 10,90,110,130
SOSTOKEN=-1              #Leave this unchanged.
SLIDINGATTENTION="True"  #Leave this unchanged.
BATCHSIZE=100
DROPOUT=$3 #We left dropout to 0.0 in our experiments.
ITERNUM=$5    #Ignore this parameter.

time python -B $SOURCEDIR/test.py $DATAINPUTDIR $RNNOBJECT $LOGOUTPUTFILE $LOGLEVEL $PREDICTIONSOUTPUTDIR $HIDDENSIZE $BATCHSIZE $NUMLAYERS $SEQUENCELENGTH $SLIDINGATTENTION $SOSTOKEN $DROPOUT $TEACHERFORCING $ITERNUM $DATASETNAME $HIERARCHICAL_ATTN_METHOD
