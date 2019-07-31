#!/bin/bash

MODELTYPE="DyAtMaxPoolH"
echo sourcepardir "$SOURCEPARDIR"
SOURCEDIR="$SOURCEPARDIR"/"$MODELTYPE"   
HIERARCHICAL_ATTN_METHOD=$6
PREDICTIONSOUTPUTPARDIR="$PREDICTIONSOUTPUTPARDIR"/"$MODELTYPE"/  #Add /full/path/to/predictions/output/directory/which/will/be/automatically/created/during/training/ .
MODELTYPE=$MODELTYPE"_"$HIERARCHICAL_ATTN_METHOD  #Model Type update.
PREDICTIONSOUTPUTDIR="$PREDICTIONSOUTPUTPARDIR"/"$MODELTYPE"  
LOGOUTPUTFILE="$LOGOUTPUTFILEPARDIR"/"$MODELTYPE/logs/encoder_decoder_regular_train.log"

#model parameters
HIDDENSIZE=$1            
NUMLAYERS=1
TEACHERFORCING=$4
LEARNINGRATE=0.001
SEQUENCELENGTH=$2         #One of 10,90,110,130
BATCHSIZE=500
SOSTOKEN=-1               #Leave this unchanged.
SLIDINGATTENTION="True"   #Leave this unchanged.
NUMEPOCHS=200   #Set to 200 (can be set to any integer value)             
DROPOUT=$3	    
ITERNUM=$5         
SLIDING_WINDOWSIZE=$7
 
time python -B $SOURCEDIR/train.py $DATAINPUTDIR $TRAININGFILENAME $RNNOBJECT $LOGOUTPUTFILE $LOGLEVEL $PREDICTIONSOUTPUTDIR $HIDDENSIZE $BATCHSIZE $NUMLAYERS $TEACHERFORCING $LEARNINGRATE $SEQUENCELENGTH $SLIDINGATTENTION $SOSTOKEN $NUMEPOCHS $DROPOUT $ITERNUM $DATASETNAME $HIERARCHICAL_ATTN_METHOD $SLIDING_WINDOWSIZE
