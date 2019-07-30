#!/bin/bash

#HYPERPARAMETERS
export PARDIR="/home/nik90/source/ijcai_19_code_and_data"  #Change this.
export SOURCEPARDIR="$PARDIR"/"source" #specify /full/path/to/ijcai_19_code_and_data/source
export DATASETNAME="ghl_small" #one of tep,ghl,electricity
export DATAINPUTDIR="$PARDIR"/"datasets"/"$DATASETNAME"/ #specify /full/path/to/ijcai_19_code_and_data/data/$DATASETNAME/
export TRAININGFILENAME="train_200000_seed_11_vars_23.csv"
export PREDICTIONSOUTPUTPARDIR="$PARDIR"/"results"/"$DATASETNAME" #Specify /full/path/to/ijcai_19_code_and_data/results/$DATASETNAME
export RNNOBJECT="GRU"
export LOGOUTPUTFILEPARDIR="$PARDIR"/"logoutputdir"/"$DATASETNAME" #Specify /full/path/to/ijcai_19_code_and_data/logoutputdir/$DATASETNAME
export LOGLEVEL="INFO"  #Can change this to DEBUG, WARNING or leave it at INFO.
export SCHEDULEDSAMPLING="False" 

seqlength=10      #One of {10,90,110,130}
hidden=96         #If seqlength = 10, then hidden = 30, otherwise hidden = 96.
dropout=0.0       
tfratio=0.0    
iternum=1         

#DyatMaxPoolH Model Training and Testing.
#sh "$SOURCEPARDIR/DyAtMaxPoolH/train.sh" $hidden $seqlength $dropout $tfratio $iternum "mean" $seqlength  #Uncomment this line to train DyAt-MaxPool-H model.
#sh "$SOURCEPARDIR/DyAtMaxPoolH/test.sh" $hidden $seqlength $dropout $tfratio $iternum "mean" $seqlength   #Uncomment this line to test DyAt-MaxPool-H model.

#DyAtH Model Training and Testing.
#sh "$SOURCEPARDIR/DyAtH/train.sh" $hidden $seqlength $dropout $tfratio $iternum "mean" $seqlength  #Uncomment this line to train DyAt-MaxPool-H model.
#sh "$SOURCEPARDIR/DyAtH/test.sh" $hidden $seqlength $dropout $tfratio $iternum "mean" $seqlength   #Uncomment this line to test DyAt-MaxPool-H model.


