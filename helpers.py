"""
    This contains helper methods to load, preprocess data and evaluate prediction results.

"""
import glob
import os
import time
import math
import numpy as np
import pandas as pd
#from pandas import ewma as pd_ema
try:
    import torch
except ImportError:
    pass
import matplotlib
matplotlib.use('agg')
import matplotlib.pyplot as plt


#Model Evaluation Code
def calculate_mse_wmse(resultsdir,sequence_length,exp_num=1):
    #@param resultsdir should be a string containing /full/path/to/ijcai_19_code_and_data/results/ghl_small/DyAtH/DyAtH_mean/
    SEQUENCELENGTH=int(sequence_length)  #10,90,110,130

    for item in glob.glob(resultsdir+"/SEQUENCE_LENGTH_{}_*ITERNUM_{}/".format(SEQUENCELENGTH,exp_num)):
            mse=list()
            wmse=list()
            for testdir in glob.glob(item+"results_test/*"):
                pred=pd.read_csv(testdir+"/timeordered_test_predictions.csv")
                tgts=pd.read_csv(testdir+"/timeordered_test_targets.csv")
                assert pred.shape==tgts.shape,"Predictions and targets should be of same shape."
                predarr=np.reshape(pred.as_matrix(),newshape=(int(pred.shape[0]/SEQUENCELENGTH),SEQUENCELENGTH,pred.shape[1]))
                tgtarr=np.reshape(tgts.as_matrix(),newshape=(int(tgts.shape[0]/SEQUENCELENGTH),SEQUENCELENGTH,tgts.shape[1]))
                _mse=np.mean(np.square(predarr - tgtarr))
                _seqMSE=np.square(predarr - tgtarr).mean(axis=2).mean(axis=0)
                _wtmse=np.average(_seqMSE,weights=[i/SEQUENCELENGTH for i in range(1,SEQUENCELENGTH+1)])
                print("Test File = {} MSE = {}, Weighted MSE = {}".format(os.path.basename(testdir),round(_mse,5),round(_wtmse,5)))
                mse.append(_mse)
                wmse.append(_wtmse)

            print("SeqLen = {} Avg. MSE = {} Avg. WMSE = {}\n\n".format(SEQUENCELENGTH,round(np.mean(mse),5),round(np.mean(wmse),5)))




# Helpers
def calculate_mse(predicted,actual):
    """
        @param predicted: numinstances X input_size torch Variable or tensor.
        @param actual: numinstances X inputsize torch Variable or tensor.
        @return  mean(sum(sq(actual - pred)))  #Scalar float value
    """

    err=(predicted - actual)
    sqerr=err**2   #element-wise square.
    mse=torch.mean(torch.sum(sqerr,dim=1))
    return mse.data[0]

def calculate_mse_tensor(predicted,actual):
    """
        @param prediction: numinstances X sequence_length X inputsize numpy ndarray.
        @param actual: numinstances X sequence_length X inputsize numpy ndarray.
        @return mse_per_sequence: python list of mean of the squared error across all time-series at each time-step. i.e  len(mse_per_sequence) = numinstances * sequence_length
        The list mse_per_sequence that is returned is essentially a mean of all the time-series for each time-step in the testing set.
    """
    err = actual - predicted
    sqerr=err**2
    sum_per_sequence=np.sum(sqerr,axis=1)  #numinstances X inputsize
    mse_per_timestep=np.mean(sqerr,axis=2).ravel() #flatten the mean squared error and return as a python list
    return mse_per_timestep

#def exponentially_weighted_moving_average(X, time_period = 10):
#    out = pd_ema(pd.Series(X), span=time_period, min_periods=time_period)
#    out[:time_period-1] = np.cumsum(X[:time_period-1]) / np.asarray(range(1,time_period))
#    return out

def _preprocess_timeseries_data_for_imputation(data_df,sequence_length=3, training_size=100):
    original_columns=data_df.columns
    num_of_sequences = data_df.shape[0] // sequence_length
    ignore_rows = data_df.shape[0] - (num_of_sequences * sequence_length)
    data_array = data_df.as_matrix()[ignore_rows: , :].reshape(-1, sequence_length,
                                                               data_df.shape[1])

    if isinstance(training_size, float):
        training_size = int(training_size * data_array.shape[0])

    select_indices = np.random.choice(np.arange(1, data_array.shape[0]-1),
                                      size=training_size, replace=True)
    ydata = data_array[select_indices]
    xindex = np.empty((select_indices.shape[0]*2,), dtype=select_indices.dtype)
    xindex[0::2] = select_indices - 1
    xindex[1::2] = select_indices + 1
    xdata = data_array[xindex].reshape(-1, sequence_length * 2, data_df.shape[1])
    if xdata.shape[0] != ydata.shape[0]:
        raise Exception('wrong!!!!!!!!')
    return xdata, ydata

def preprocess_timeseries_data(inputdatadir,sequence_length=3,slidingwindow=False,columnsofinterest=None,columnstodrop=None):
    dfs=list()
    arrs=list()
    for _file in glob.glob(inputdatadir):
        datadf=pd.read_csv(_file)
        if 'Time' in datadf.columns:
            datadf.drop(['Time'],axis=1,inplace=True)
        if columnstodrop is not None:
            datadf = datadf.drop(columnstodrop,axis=1)
        elif columnsofinterest is not None:
            datadf = datadf[columnsofinterest]

        datadf = (datadf - datadf.mean(axis=0))/(datadf.max(axis=0) - datadf.min(axis=0)) #Mean Normalization Range (-1,1)

        print("Shape of input df = {}".format(datadf.shape))
        data_df,data_array=create_data_sequences(datadf,sequence_length=sequence_length,slidingwindow=slidingwindow)
        print("Shape of output data_df = {}, data_array = {}".format(data_df.shape,data_array.shape))
        print("Shape of Data DF = {}, data_df[-3].shape = {}, data_df[-2].shape = {}, data_df[-1].shape = {}".format(data_df.shape,data_df.iloc[-3,:].shape,data_df.iloc[-2,:].shape,data_df.iloc[-1,:].shape))
        print("File = {}".format(_file))
        dfs.append(data_df)
        arrs.append(data_array)

    df = pd.concat(dfs).reset_index().drop('index',axis=1) 
    arr = np.concatenate(arrs)
    print("Shape of Concatenated DF = {}".format(df.shape))
    print("Shape of Concatenated List of Arrays = {}".format(arr.shape))

    return df , arr

def create_data_sequences(data_df,sequence_length=3,slidingwindow=True):
    """
          Helper function to convert the input dataframe to a 3D numpy array of type (num_instances,sequence_length,input_size).
          This method makes it easy to convert the input data into something that PyTorch can consume easily.
          This is requried for a Seq2Seq model in pytorch.

          @param data_df: The data is expected to be in the form of a dataframe where each index (row) represents a time step.
          @param sequence_length: This represents the window length in a sliding window experiment.

        Example Method Call:
            _,data = preprocess_timeseries_data(inputdf,sequence_length=SEQUENCE_LENGTH)

            Here, we ignore the dataframe which is the first output and only use the numpy array representation of the data
            in our experiments.

    """
    original_columns=data_df.columns

    for i in range(1,sequence_length):
        for col in original_columns:
            data_df[col+"_shift"+str(i)]=data_df[col].shift(-i) #Here we create a new column that essentially upshifts values by 1.

    data_df.dropna(axis=0,how='any',inplace=True)  #Here we want to drop a ROW (axis=0) if ANY of the values of the row in the dataframe are NaN.
                                                   #This is a side-effect of the upshift above. Pandas automatically fills upshifted columns with NaN.
                                                   #Here we just drop those rows.
    if slidingwindow==False:
        data_df = data_df.iloc[data_df.index%sequence_length==0]

    numinstances=data_df.shape[0]
    data_as_array=data_df.as_matrix()
    numinst=data_as_array.shape[0]
    numfeatures=len(original_columns)
    data_array=np.reshape(data_as_array,newshape=(numinst,sequence_length,numfeatures))
    
    return data_df,data_array

def preprocess_targets_data(targets_df,sequence_length=3,column='DANGER',slidingwindow=True):
    """

        We need to preprocess targets data and then unroll the sequences once again.
        Note after preprocessing, we ignore the very first row of the dataframe as the decoder doesn't see this value.

    """
    target_df=targets_df.copy(deep=True)
    target_df=pd.DataFrame(target_df[column],columns=[column])
    for i in range(1,sequence_length):
        target_df[column+"_shift"+str(i)]=target_df[column].shift(-i) #Here we create a new column that essentially upshifts values by 1.

    target_df.dropna(axis=0,how='any',inplace=True)  #Here we want to drop a ROW (axis=0) if ANY of the values of the row in the dataframe are NaN.
                           #This is a side-effect of the upshift above. Pandas automatically fills upshifted columns with NaN.
                           #Here we just drop those rows.
    if slidingwindow==False:
        target_df = target_df.iloc[target_df.index%sequence_length==0]
        target_df = target_df.iloc[1:] #The decoder doesn't predict the first instance in the test set.
    targets_unrolled=target_df.as_matrix().ravel()
    unrolled_df=pd.DataFrame(targets_unrolled,columns=[column])
    numinst=targets_unrolled.shape[0]
    numfeatures=1
    target_indices=list(unrolled_df[(unrolled_df[list(unrolled_df.columns)]== 1.0).any(axis=1)].index.values)
    return unrolled_df,target_indices




def load_test_files(inputdir):
    files=dict()
    for idx,_file in enumerate(glob.glob(inputdir+"*.csv")):
        if "train_" in _file:
            continue

        files[os.path.basename(_file)]=pd.read_csv(_file)
    return files

#def asMinutes(s):
#    m = math.floor(s / 60)
#    s -= m * 60
#    return '%dm %ds' % (m, s)
#
#
#def timeSince(since, percent):
#    now = time.time()
#    s = now - since
#    es = s / (percent)
#    rs = es - s
#    return '%s (- %s)' % (asMinutes(s), asMinutes(rs))



def showPlot(points,title):
    #plt.figure()
    fig, ax = plt.subplots(1,1,figsize=(20,6))
    # this locator puts ticks at regular intervals
    loc = ticker.MultipleLocator(base=0.2)
    ax.yaxis.set_major_locator(loc)
    ax.plot(points)
    ax.set_title(title,fontsize=20)

def _3D_pytorch_to_2D_pandas(ndarray,seqdim=1,column_names=[],groupby="sequence_order"):
    """
        @param ndarray: a 3D numpy array (batchsize,sequence_length,input_size)
        @param seqdim: This parameter indicates the dimension in `tensor` that has the sequence_length.
        @param column_names: A list of strings indicating the names of each column in ndarray axis = 2. i.e. in the input_size dimension.
	@param groupby: This is an important parameter that changes the return type of the function. If the function has groupby `sequence_order` it is the default behavior and the dataframe returned will have values grouped by `Sequence` wherein all values of Sequence 0 will appear followed by all values of Sequence 1 etc..
          The other option of groupby is `time_order`, if this is the case, then the dataframe returned will be similar to the input dataframe wherein values will be grouped by temporal measurements.

        This function will be used to convert the 3D pytorch tensor into a 2D dataframe. This dataframe can then be saved into a csv file.
        @return df: The data frame will essentially have an index named `Sequence` wherein each value of the index will indicate the sequence time-step for all the instances in a single batch.
        Ex: Let us consider the following array of dimension (4,2,3).
           [
            [[1 , 2 ,3  ],
              [4 , 5 , 6 ]],

            [[ -1 , -2 ,- 3 ],
              [ -4 , -5 ,- 6 ]],

            [[ -7 , -8 , -9 ],
              [ -10 , -11 , -12 ]],

            [[ 7 , 8 , 9 ],
              [ 10 , 11 ,12 ]]
           ]

        The data frame for the aforementioned array will yield the following:

              Column1       Column2     Column3
         Sequence
        0           1                 2               3
        0          -1                -2              -3
        0          -7                -8              -9
        0           7                 8               9
        1           4                 5               6
        1          -4                -5              -6
        1         -10               -11             -12
        1          10                11              12

       If groupby == time_order
        We first drop the Sequence column.

       index    Column1       Column2    Column3
        0         1              2         3
        1         4              5         6
        2        -1             -2        -3
        3        -4             -5        -6
        4        -7             -8        -9
        5        -10            -11       -12
        6         7              8         9
        7         10             11        12
    """
    if groupby.lower()=="sequence_order":
        pan=pd.Panel(ndarray)
        df=pan.swapaxes(0,2).to_frame()
        df.index = df.index.droplevel('minor')
        df.index.name='Sequence'

        if len(column_names)>0:
            df.columns=column_names
        return df

    elif groupby.lower()=="time_order":
        reshapedarray=ndarray.reshape((ndarray.shape[0]*ndarray.shape[1],ndarray.shape[2]))
        df=pd.DataFrame(reshapedarray,columns=column_names)
        return df

def plot_forecasts(pred_df,target_df,figsavedir=None,anom_indices=None,
                   dynamic_anomaly_thresholds_df=None,mse_array=None,
                   smoothed_mse_array=None,anomtype=None):

	residuals= target_df - pred_df
	for col in pred_df.columns:
		if col=="Sequence":
			continue

		fig,ax=plt.subplots(2,1,figsize=(30,14))
		ax[0].plot(pred_df[col].values,c='g')
		ax[0].plot(target_df[col].values,c='b',linewidth=1.1)
		ax[0].legend(['Predicted','Actual'],fontsize=20)
		ax[0].set_title("Column = "+col,fontsize=22)
		residuals[col].plot(ax=ax[1])
		if dynamic_anomaly_thresholds_df is not None:
			dynamic_anomaly_thresholds_df[col].plot(ax=ax[1],c='y')

		ax[1].set_title("Residuals {}".format(col))
		if anom_indices is not None:
			target_df.loc[anom_indices,col].plot(ax=ax[0],c='r')
			residuals[col].iloc[anom_indices].plot(ax=ax[1],c='r')

		if figsavedir is not None:
			fig.savefig(figsavedir+str(col)+".png")

	if mse_array is not None and smoothed_mse_array is not None and anomtype is not None:
		fig,ax=plt.subplots(2,1,figsize=(20,10))
		ax[0].plot(mse_array)
		ax[1].plot(smoothed_mse_array)
		if anom_indices is not None:
			pd.Series(mse_array.ravel()).iloc[anom_indices].plot(ax=ax[0],c='r')
			pd.Series(smoothed_mse_array.ravel()).iloc[anom_indices].plot(ax=ax[1],c='r')
		if figsavedir is not None:
			fig.savefig(figsavedir+"evaluation_test_mse_and_smoothed_mse_{}.png".format(anomtype))

#Residual Calculation, Anomaly Detection and Evaluation
def calculate_residuals(pred_df,target_df):
    return target_df - pred_df

#def plot_forecasts(pred_df,target_df,figsavedir=None,anom_indices=None,dynamic_anomaly_thresholds_df=None):
#	print("Shape of pred_df = {}, target_df = {}".format(pred_df.shape,target_df.shape))
#	pred_grp=pred_df.groupby("Sequence")
#	batch_size=0 #All batches should be of equal size.
#	for key, df in pred_grp:
#		batch_size=df.shape[0]
#		break  #just iterating once should be enough since all batch_sizes should be equal.
#
#	residuals=calculate_residuals(pred_df,target_df) #Absolute Value
#	for col in pred_df.columns:
#		if col=="Sequence":
#			continue
#		pred_list=list()
#		actual_list=list()
#		for i in range(batch_size):
#			pred=pred_df[col].iloc[i::batch_size].values.ravel()
#			actual=target_df[col].iloc[i::batch_size].values.ravel()
#			pred_list.extend(pred)
#			actual_list.extend(actual)
#
#		fig,ax=plt.subplots(2,1,figsize=(30,14))
#		ax[0].plot(pred_list,c='g')
#		ax[0].plot(actual_list,c='b',linewidth=1.1)
#		ax[0].legend(['Predicted','Actual'],fontsize=20)
#		ax[0].set_title("Column = "+col,fontsize=22)
#		residuals[col].plot(ax=ax[1])
#		if dynamic_anomaly_thresholds_df is not None:
#			dynamic_anomaly_thresholds_df[col].plot(ax=ax[1],c='y')
#
#		ax[1].set_title("Residuals {}".format(col))
#		if anom_indices!=None:
#			_series=pd.Series(actual_list)
#			_series.iloc[anom_indices].plot(ax=ax[0],c='r')
#			residuals[col].iloc[anom_indices].plot(ax=ax[1],c='r')
#
#		if figsavedir!=None:
#			fig.savefig(figsavedir+str(col)+".png")
#
##Residual Calculation, Anomaly Detection and Evaluation
#def calculate_residuals(pred_df,target_df):
#    pred_grp=pred_df.groupby("Sequence")
#    batch_size=0 #All batches should be of equal size.
#    for key, df in pred_grp:
#            batch_size=df.shape[0]
#            break  #just iterating once should be enough since all batch_sizes should be equal.
#
#    residuals=dict() #column names are keys, absolute residual lists are values.
#    cols=list()
#    for col in pred_df.columns:
#            if col=="Sequence":
#                    continue
#            cols.append(col)
#            pred_list=list()
#            actual_list=list()
#            for i in range(batch_size):
#                    pred=pred_df[col].iloc[i::batch_size].values.ravel()
#                    actual=target_df[col].iloc[i::batch_size].values.ravel()
#                    pred_list.extend(pred)
#                    actual_list.extend(actual)
#            residuals[col]=(np.array(pred_list) - np.array(actual_list)) #note: residuals are not absolute values.
#
#    return pd.DataFrame(residuals,columns=cols)

def static_anomaly_thresholds(val_preds,val_targets,SENSITIVITY_THRESHOLD=0.001):
    """
	@param val_preds: The validation predictions dataframe.
	@param val_targets: The validation targets dataframe.
    """
    SENSITIVITY_THRESHOLD=0.001
    dynamic_thresholds=dict()
    for col in val_preds.columns:
        if col=="Sequence":
            continue

        residuals=val_preds[col].values - val_targets[col].values #raw residual value.
        anom_idx=np.argsort(residuals)[int(residuals.shape[0]*(1 - SENSITIVITY_THRESHOLD))]
        anom_thr=1.1*residuals[anom_idx]
        anom_neg_thr=-anom_thr
        dynamic_thresholds['positive_thr']=anom_thr
        dynamic_thresholds['negative_thr']=anom_neg_thr

        #plot
#         fig,ax=plt.subplots(1,1,figsize=(20,6))
#         ax.plot(residuals)
#         ax.axhline(anom_thr,c='k')
#         ax.axhline(anom_neg_thr,c='k')
#         ax.set_title(col,fontsize=20)

    return dynamic_thresholds

def calculate_dynamic_threshold(residuals,window,initial_anomaly_threshold):
    """
        Given a list of residuals, an initial anomaly threshold obtained from the validation evaluation, and a window, calculate the dynamic anomaly threshold for a time-series in a sliding window manner.
    """
    pass

def calculate_precision_recall_f1(residuals,anomaly_thresholds, ground_truth_anomalies):
    """
        Given the residuals and anomaly thresholds for a particular time-series, figure out which time-steps exceed the anomaly threshold and assign them as anomalies. Once we obtain the predicted time-steps when the model thinks the system has experienced anomalies, we compare this with the ground truth anomalies list and calculate TP, FP, TN, FN , precision, recall and F1 score.
    """
    pass


def makedirs(path):
	"""
		In python3 this can be achieved by os.makedirs(fullfilepath,exist_ok=True). Here we have to define our own recursive solution.
	"""
	sub_path = os.path.dirname(path)
	if not os.path.exists(sub_path):
		makedirs(sub_path)
	if not os.path.exists(path):
		os.mkdir(path)
