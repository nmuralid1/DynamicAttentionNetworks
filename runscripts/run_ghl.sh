#!/bin/bash

#HYPERPARAMETERS
export PARDIR="/home/nik90/source/DynamicAttentionNetworks"  #Change this.

export SOURCEPARDIR="$PARDIR"/"source" #specify /full/path/to/DynamicAttentionNetworks/source

export DATASETNAME="ghl_small" #The DATASETNAME character needs to contain one of the substrings{tep, ghl, electricity}. If you wish to run the models on a new dataset, you need to add the appropriate section to the train.py and test.py files in source/MODELNAME/{train,test}.py

export DATAINPUTDIR="$PARDIR"/"datasets"/"$DATASETNAME"/ #specify /full/path/to/DynamicAttentionNetworks/data/$DATASETNAME/

export TRAININGFILENAME="train_200000_seed_11_vars_23.csv"

export PREDICTIONSOUTPUTPARDIR="$PARDIR"/"results"/"$DATASETNAME" #Specify /full/path/to/DynamicAttentionNetworks/results/$DATASETNAME

export RNNOBJECT="GRU"

export LOGOUTPUTFILEPARDIR="$PARDIR"/"logoutputdir"/"$DATASETNAME" #Specify /full/path/to/DynamicAttentionNetworks/logoutputdir/$DATASETNAME

export LOGLEVEL="INFO"  #Can change this to DEBUG, WARNING or leave it at INFO.

export SCHEDULEDSAMPLING="False" 

seqlength=90      #One of {10,90,110,130} used for our experiments. However, this can be set to any integer value as required.
hidden=96        #If seqlength = 10, then hidden = 30, otherwise hidden = 96. Again, can be set to any integer value as required.
dropout=0.0      #Dropout percentage.
tfratio=0.0      #If Teacher forcing is to be incorporated, set this to a value between [0 - 1). 
iternum=1        #If you want to run multiple experiments with the same aforementioned hyperparameters, (seqlength, hidden,dropout etc.) updating iternum (1,2,3...) for each experimental run will ensure all models and results are stored in separate directories appended with ITERNUM_i where 'i' is the specific iternum value.

#DyatMaxPoolH Model Training and Testing.
#Uncomment the following line to train the DyAt-MaxPool-H model.
#sh "$SOURCEPARDIR/DyAtMaxPoolH/train.sh" $hidden $seqlength $dropout $tfratio $iternum "mean" $seqlength

#Uncomment the following line to test the DyAt-MaxPool-H model.
#sh "$SOURCEPARDIR/DyAtMaxPoolH/test.sh" $hidden $seqlength $dropout $tfratio $iternum "mean" $seqlength  

########################################################################################################

#DyAtH Model Training and Testing.
#Uncomment the following line to train the DyAt-H model.
#sh "$SOURCEPARDIR/DyAtH/train.sh" $hidden $seqlength $dropout $tfratio $iternum "mean" $seqlength 

#Uncomment the following line to test the DyAt-H model.
#sh "$SOURCEPARDIR/DyAtH/test.sh" $hidden $seqlength $dropout $tfratio $iternum "mean" $seqlength  


