# Requirements:
  - anaconda 4.3.21
  - python   3.6.2
  - cuda     8.0.54
  - pytorch  0.4.*
  - gcc      5.4.0 
 
All models require a GPU to train and were run on a single GPU with 16 GB memory. The models were run on a server with a slurm task scheduled but run scripts are now presented purely as shell scripts deviod of the slurm related commands.

## RUN SCRIPTS
 ###  runscripts/   
 - Contains all model execution scripts and the evaluation script for training and testing the forecasting models and evaluation model performance on the tep, ghl and electricity datasets.

- *run_ghl.sh* 
    - This is the shell script to run the ghl dataset. It contains two models DyAtH, DyAtMaxPoolH and the training and testing scripts for each. In each case, the training scripts need to be run  to completion first before invoking the testing scripts for the model.
	Run: `sh run_ghl.sh`

	- In the run script, the '$PARDIR' parameter need to be set indicating the full path to wherever the DynamicAttentionNetworks/ directory is stored. Instructions for setting the rest of the model hyperparameters are provided within the shell scripts.
	
### helpers.py 
  - This script contains a `calculate_mse_wmse()` function --- This is the evaluation script which calculates MSE (Mean Squared Error) and WMSE (Weighted Mean Squared Error) for each dataset per model / sequence length. 
  - It accepts two parameters, the path to the results directory (string with full path) and a sequence length (integer).
	Ex: If we were trying to calcualte the MSE, WMSE for the DyAtH model for the ghl_small dataset and sequence length 90, we would supply the following command while calling the helper function.
```
         import helpers
         helpers.calculate_mse_wmse(/full/path/to/DynamicAttentionNetworks/results/DyAtH/DyAtH_mean/,90)
```

## source/ 
    DyAtH/
    	> train.py  -- Contains model training logic.
    	> test.py	  -- Contains model testing logic.
    	> model_train_eval.py -- Contains helper functions and sub-routines related to model training and testing.
    	> train.sh  -- Shell script to invoke train.py
    	> test.sh	  -- Shell script to invoke test.py
    	> rnn.py  -- Contains the Sequence to Sequence Architecture Definition.

	DyAtMaxPoolH/	
 		> Similar structure as in DyAtH.
## datasets/
    ghl_small/ - GHL dataset used in the paper.


## notebooks/
    GHL\ Experiments.ipynb  
        -This notebook depicts how once trainining and testing has been conducted for a specific model (DyAt-H, DyAt-Maxpool-H) and a specific sequence length, it can be evaluated to obtain the average MSE, WMSE values.

Note, error values might not be recovered exactly as the torch.cuda.manual_seed() was not set during experimentation.
