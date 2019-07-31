"""
This script will house, training, testing helper functions which can be used in conjunction with the system architecture to train, test and evaluate neural network architectures.

"""
import torch
import torch.nn as nn
from torch.autograd import Variable
import torch.optim as optim
#import torch.nn.functional as F

import random
import numpy as np
from datetime import datetime
import time

import sys
sys.path.append("../")
from helpers import calculate_mse,calculate_mse_tensor

def _encoder_sequence_iter(input_tensor,encoder,num_layers,logger):
	
	encoder_cellstate = encoder.initHidden(numlayers=num_layers,batchsize=input_tensor.size(0)) #Special case only for LSTM.	
	encoder_hidden = encoder.initHidden(numlayers=num_layers,batchsize=input_tensor.size(0)) #Called once per batch or once per call to the forward method?
	hidden_states=torch.zeros(input_tensor.size(0),encoder.sequence_length,encoder.hidden_size) #accumulate all hidden states from the encoder.
	sequence_length=encoder.sequence_length
	hidden_size=encoder.hidden_size

	for ei in range(sequence_length): #Iterate over each sequence in input tensor. i.e iterate over seq_size dimension.
		#Iterate over each sequence of the instance.
		inp_t = input_tensor[:,ei,:].contiguous().view(input_tensor.size(0),1,input_tensor.size(2))
		logger.debug("input_t.size() = {}".format(inp_t.size()))	
		inp_t = inp_t.cuda()                        ## CUDA
		encoder_hidden = encoder_hidden.cuda()      ## CUDA
		if encoder.rnnobject=="LSTM":
			encoder_cellstate=encoder_cellstate.cuda()  ##CUDA
			encoder_output, (encoder_hidden,encoder_cellstate) = encoder(inp_t, encoder_hidden,encoder_cellstate)
		else:
			encoder_output,encoder_hidden = encoder(inp_t,encoder_hidden)

		hidden_states[:,ei,:]=encoder_hidden[-1,:,:].unsqueeze(dim=0) #Store only hidden state from top most hidden layer in stacked RNNs.

	if encoder.rnnobject=="LSTM":
		return encoder_hidden,cell_state,hidden_states
	else:
		return encoder_hidden,hidden_states

def _decoder_sequence_iter(input_tensor,target_tensor,decoder,encoder_hiddenstate,SOS_TOKEN,teacher_forcing_ratio,prev_hidden_states,criterion,logger,sliding_attention,hierattnmethod,encoder_cellstate=None,test=False):
	loss = 0 #Initialize loss variable.
	#First input to the decoder. Size: batch_size,1,input_size
	decoder_input = Variable(torch.FloatTensor([[SOS_TOKEN]*input_tensor.size(2)]*input_tensor.size(0)),requires_grad=True) 
	decoder_input = decoder_input.view(decoder_input.size(0),1,decoder_input.size(1))
	decoder_hidden = encoder_hiddenstate #First hidden input to the decoder is the last hidden state of the encoder.
	sequence_length = decoder.sequence_length

	if decoder.rnnobject=="LSTM":
		decoder_cellstate =  encoder_cellstate #Maybe we shouldn't pass the cell state on?

	# Without teacher forcing: use its own predictions as the next input
	use_teacher_forcing = True if random.random() < teacher_forcing_ratio else False
	#We iterate per sequence, to obtain outputs.
	
	#Store Predictions.
	predictions_batch = torch.zeros(target_tensor.size(0),target_tensor.size(1),target_tensor.size(2)).cuda()

	attention_vectors=list() #Store attention vectors.
	context_vectors=list() #List and eventually a matrix of (Batchsize X hiddensize) context vectors.
	#contextvector=None    #Instantiated either to None or to be a mean of the encoder hidden states.
	contextvector=torch.mean(prev_hidden_states,dim=1).unsqueeze(1)  #Instantiated either to None or to be a mean of the encoder hidden states. (batchsize , 1, hiddensize)
	hierhiddenstateSimilarityMatrix = torch.zeros(target_tensor.size(0),sequence_length,2)  #(batchsize,sequencelength,2). The third dimension (i.e. 2) will house two separate similarity matrices. 1. (hierarchical_attentional_hidden_state,contextvector) 2. (hierarchical_attentional_hidden_state,attentional_hidden_state)
	
	for di in range(sequence_length):
		decoder_input = decoder_input.cuda()           ## CUDA
		logger.debug("decoder_input.size() = {}".format(decoder_input.size()))
		if decoder.rnnobject=="LSTM":
			decoder_output, (decoder_hidden,decoder_cellstate),attentional_hidden_state,attention_vector,hierarchical_attentional_hidden_state = decoder(decoder_input,prev_hidden_states,decoder_hidden,contextvector=contextvector,hierattnmethod=hierattnmethod,cell_state=decoder_cellstate)
		else:
			decoder_output, decoder_hidden,attentional_hidden_state,attention_vector,hierarchical_attentional_hidden_state = decoder(decoder_input,prev_hidden_states,decoder_hidden,contextvector=contextvector,hierattnmethod=hierattnmethod)
		logger.debug("Decoder Training prev_hidden_states.size() = {}, decoder_hidden.size() = {}".format(prev_hidden_states.size(),decoder_hidden.size()))
		#Update Contextvector (this happens only once, i.e. after first decoder unit is evaluated.)
		if (contextvector is not None) and (test is True):
			logger.debug("Attentional Hidden State Size = {}, Context Vector Size = {}, Hier. Attn. Hidden St. Size = {}".format(attentional_hidden_state.size(),contextvector.size(),hierarchical_attentional_hidden_state.size()))
			#hierarchical_attentional_hidden_state = (batchsize X hiddensize), attentional_hidden_state = (batchsize X 1 X hiddensize), contextvector = (batchsize X 1 X hiddensize)	
			hierarchical_attentional_hidden_state=hierarchical_attentional_hidden_state.squeeze(dim=1)
			hierattn_ctxvec_sim=nn.functional.cosine_similarity(contextvector,hierarchical_attentional_hidden_state.unsqueeze(1),dim=2) #(batchsize X 1)
			hierattn_attnhiddnst_sim=nn.functional.cosine_similarity(attentional_hidden_state,hierarchical_attentional_hidden_state.unsqueeze(1),dim=2) #(batchsize X 1)	
			hierhiddenstateSimilarityMatrix[:,di,0] = hierattn_ctxvec_sim.squeeze()
			hierhiddenstateSimilarityMatrix[:,di,1] = hierattn_attnhiddnst_sim.squeeze()			
			
		if contextvector is None:
			contextvector = attentional_hidden_state		
		
		if sliding_attention==True:
			#SLIDING ATTENTION 
			prev_hidden_states=torch.cat((prev_hidden_states[:,1:,:],decoder_hidden[-1,:,:].unsqueeze(dim=0).permute(1,0,2)),dim=1) #for now just use decoder hidden state we can also use attentional_hidden state if required later. (batch_size , seq_len, hidden_size) dim of prev_hidden_states.
	
		#logger.debug("decoder_output.size() = {}, decoder_hidden.size() = {}".format(decoder_output.size(),decoder_hidden.size()))
		logger.debug("Attention Vector Sample = {}".format(attention_vector[0:2,:,:]))
		attention_vectors.append(attention_vector)
		#logger.debug("decoder_output.size() = {}".format(decoder_output.size()))
		#decoder_output = decoder_output.view(decoder_output.size(0),decoder_output.size(2))  #Change n X m X k to n X k
		decoder_output = decoder_output.squeeze() #batchsize X inputsize from batchsize X 1 X inputsize.
		logger.debug("after re-shape, decoder_output.size() = {}, target_tensor[:,di].size() = {}".format(decoder_output.size(),target_tensor[:,di,:].size()))
		loss += criterion(decoder_output, target_tensor[:,di,:])
		predictions_batch[:,di,:]=decoder_output
		logger.debug("prev_hidden_states.size() = {}, decoder_hidden.size() = {}".format(prev_hidden_states.size(),decoder_hidden.size()))
		if use_teacher_forcing: #Teacher Forcing.
			decoder_input = target_tensor[:,di,:].contiguous().view(target_tensor.size(0),1,target_tensor.size(2)) # Teacher forcing
		else:
			decoder_input = decoder_output.view(decoder_output.size(0),1,decoder_output.size(1))
	
		
	#TODO: Need to do cosine similarity and return cosine similarity matrix between contextvector, hierarchical_attentional_hidden_states; attentional_hidden_states and hierarchical_attentional_hidden_states.
	if test is True:
		return attention_vectors,predictions_batch,loss,hierhiddenstateSimilarityMatrix
	else:
		return attention_vectors,predictions_batch,loss


def train(inputbatch, targetbatch, encoder, decoder, encoder_optimizer, decoder_optimizer,num_layers,criterion,teacher_forcing_ratio,SOS_TOKEN,logger,sliding_attention,hierattnmethod):

	encoder_optimizer.zero_grad()
	decoder_optimizer.zero_grad()

	sequence_length = inputbatch.size(1)    #This is the sequence length of the input tensor.
	target_sequence_length = targetbatch.size(1)  #This is the sequence length of the target tensor.
	loss = 0
	logger.debug("Sequence_length = {}, target_sequence_length = {}".format(sequence_length,target_sequence_length))
	#Iterate over each instance of the input batch.
	input_tensor  = inputbatch # batch_size X seq_size X input_size
	target_tensor = targetbatch # batch_size X seq_size X input_size
	
	#Encoder
	if encoder.rnnobject=="LSTM":
		encoder_hidden,encoder_cellstate,hidden_states=_encoder_sequence_iter(input_tensor,encoder,num_layers,logger)
	else:
		encoder_hidden,hidden_states=_encoder_sequence_iter(input_tensor,encoder,num_layers,logger)
	
	logger.debug("Hidden_States.size() final = {}".format(hidden_states.size()))
	prev_hidden_states=hidden_states.cuda() #Just store it in a separate variable.
 	
	#Decoder
	if decoder.rnnobject=="LSTM":
		attention_vectors,predictions_batch,loss=_decoder_sequence_iter(input_tensor,target_tensor,decoder,encoder_hidden,SOS_TOKEN,teacher_forcing_ratio,prev_hidden_states,criterion,logger,sliding_attention,hierattnmethod,encoder_cellstate,test=False)	
	else:
		attention_vectors,predictions_batch,loss=_decoder_sequence_iter(input_tensor,target_tensor,decoder,encoder_hidden,SOS_TOKEN,teacher_forcing_ratio,prev_hidden_states,criterion,logger,sliding_attention,hierattnmethod,test=False)		

	loss.backward()
	encoder_optimizer.step()
	decoder_optimizer.step()
	#The loss is a 1-D variable of size 1. So we just return the data.
	return loss.item(), predictions_batch

  
def testNew(inputbatch, targetbatch, encoder, decoder,num_layers,criterion,SOS_TOKEN,teacher_forcing_ratio,logger,sliding_attention,hierattnmethod,istest=False):
	"""
		@param: inputbatch: (batchsize X sequence_length X input_size) Variable which represents data fed into the encoder.
		@param: targetbatch: (batchsize X sequence_length X input_size) Variable which represents data fed into the decoder (during teacher forcing) or data that is not fed.
		@param: encoder: The encoder network object being tested.
		@param: decoder:  The decoder network object being tested.
		@param: criterion: Used only during validation to record the validation loss. But for testing as well, we can just supply a criterion.
	"""

	#Run Encoder
	if encoder.rnnobject=="LSTM":	
		encoder_hidden,cell_state,hidden_states=_encoder_sequence_iter(inputbatch,encoder,num_layers,logger)
	else:
		encoder_hidden,hidden_states=_encoder_sequence_iter(inputbatch,encoder,num_layers,logger)

	
	#First input to the decoder. Size: batch_size,1,input_size
	prev_hidden_states=hidden_states.cuda()

	mse_predictions=list()
	mse_per_timestep=list()
	
	#Run Decoder
	if decoder.rnnobject=="LSTM":
		if istest is False:
			attention_vectors,decoder_predictions,loss = _decoder_sequence_iter(inputbatch,targetbatch,decoder,encoder_hidden,SOS_TOKEN,teacher_forcing_ratio,prev_hidden_states,criterion,logger,sliding_attention,hierattnmethod,cell_state,test=test)
		elif istest is True:
			attention_vectors,decoder_predictions,loss,hierAttnSimMat= _decoder_sequence_iter(inputbatch,targetbatch,decoder,encoder_hidden,SOS_TOKEN,teacher_forcing_ratio,prev_hidden_states,criterion,logger,sliding_attention,hierattnmethod,cell_state,test=istest)

	else:	
		if istest is False:
			attention_vectors,decoder_predictions,loss = _decoder_sequence_iter(inputbatch,targetbatch,decoder,encoder_hidden,SOS_TOKEN,teacher_forcing_ratio,prev_hidden_states,criterion,logger,sliding_attention,hierattnmethod,test=istest)
		elif istest is True:	
			attention_vectors,decoder_predictions,loss,hierAttnSimMat= _decoder_sequence_iter(inputbatch,targetbatch,decoder,encoder_hidden,SOS_TOKEN,teacher_forcing_ratio,prev_hidden_states,criterion,logger,sliding_attention,hierattnmethod,test=istest)

	
	attention_tensor=torch.stack(attention_vectors)
	logger.debug("Attention Tensor Size = {}, Attention List Length = {}, Single Attention Vector Size = {}".format(attention_tensor.size(),len(attention_vectors),attention_vectors[0].size()))

	mse_per_timestep=calculate_mse_tensor(decoder_predictions.cpu().detach().numpy(),targetbatch.cpu().detach().numpy()) #returns a list containing the mean-squared error per timestep.
	if istest is False:
		return loss.item(),mse_per_timestep,decoder_predictions,attention_tensor
	elif istest is True:
		return loss.item(),mse_per_timestep,decoder_predictions,attention_tensor,hierAttnSimMat


def test(inputbatch, targetbatch, encoder, decoder,num_layers,criterion,SOS_TOKEN,teacher_forcing_ratio,logger,sliding_attention):
	"""
		@param: inputbatch: (batchsize X sequence_length X input_size) Variable which represents data fed into the encoder.
		@param: targetbatch: (batchsize X sequence_length X input_size) Variable which represents data fed into the decoder (during teacher forcing) or data that is not fed.
		@param: encoder: The encoder network object being tested.
		@param: decoder:  The decoder network object being tested.
		@param: criterion: Used only during validation to record the validation loss. But for testing as well, we can just supply a criterion.
	"""

	logger.info("inside test(), inputbatch.size() = {}, targetbatch.size() = {}, SOS_TOKEN = {}".format(inputbatch.size(),targetbatch.size(),SOS_TOKEN))
	#Done
	encoder_hidden = encoder.initHidden(numlayers=num_layers,batchsize=inputbatch.size(0))
	use_teacher_forcing = True if random.random() < teacher_forcing_ratio else False

	if encoder.rnnobject=="LSTM":
		encoder_cellstate = encoder.initHidden(numlayers=num_layers,batchsize=inputbatch.size(0))

	sequence_length = inputbatch.size(1)    #This is the sequence length of the input tensor.
	target_sequence_length = targetbatch.size(1)  #This is the sequence length of the target tensor.
	hidden_size=encoder_hidden.size(2)
	loss = 0

	#Iterate over each instance of the input batch.
	input_tensor  = inputbatch # batch_size X seq_size X input_size
	target_tensor = targetbatch # batch_size X seq_size X input_size
	attention_vectors=list()
	hidden_states=torch.zeros(input_tensor.size(0),sequence_length,hidden_size)

	use_teacher_forcing = True if random.random() < teacher_forcing_ratio else False

	for ei in range(sequence_length): #Iterate over each sequence in input tensor. i.e iterate over seq_size dimension.
		#Iterate over each sequence of the instance.
		inp_t = input_tensor[:,ei,:].contiguous().view(input_tensor.size(0),1,inputbatch.size(2))
		inp_t = inp_t.cuda()                      ## CUDA
		logger.debug("input_t.size() = {}".format(inp_t.size()))
		encoder_hidden = encoder_hidden.cuda()    ## CUDA
		logger.debug("encoder_hidden.size() = {}".format(encoder_hidden.size()))
		if encoder.rnnobject=="LSTM":
			encoder_cellstate = encoder_cellstate.cuda() ## CUDA
			encoder_output, (encoder_hidden,encoder_cellstate) = encoder(inp_t, encoder_hidden,encoder_cellstate)
		else:
			encoder_output, encoder_hidden = encoder(inp_t, encoder_hidden)

		hidden_states[:,ei,:]=encoder_hidden[-1,:,:].unsqueeze(dim=0)
	
	#First input to the decoder. Size: batch_size,1,input_size
	decoder_input = Variable(torch.FloatTensor([[SOS_TOKEN]*inputbatch.size(2)]*inputbatch.size(0)),requires_grad=True) 
    
	#print("Size of decoder input before reshape",decoder_input.size())
	decoder_input = decoder_input.view(decoder_input.size(0),1,decoder_input.size(1))
	decoder_hidden = encoder_hidden #First hidden input to the decoder is the last hidden state of the encoder.
	if decoder.rnnobject=="LSTM":
		decoder_cellstate = encoder_cellstate  #First Cell State of the decoder is the last cell state of the encoder.

	#We iterate per sequence, to obtain outputs. There is no notion of teacher forcing for the validation/testing set.
	decoder_predictions=torch.zeros(target_tensor.size(0),target_tensor.size(1),target_tensor.size(2)) #we will store the validation predictions in this tensor for plotting and analysis.
	prev_hidden_states=hidden_states.cuda()

	mse_predictions=list()
	mse_per_timestep=list()
	
	attention_vectors=list()	
	contextvector=None
	for di in range(target_sequence_length):
		decoder_input = decoder_input.cuda()    ## CUDA
		logger.debug("decoder_input.size() = {}".format(decoder_input.size()))
		if decoder.rnnobject=="LSTM":
			decoder_output, (decoder_hidden,decoder_cellstate),attentional_hidden_state,attention_vector= decoder(decoder_input,prev_hidden_states,decoder_hidden,decoder_cellstate)
			
		else:
			decoder_output, decoder_hidden,attentional_hidden_state,attention_vector = decoder(decoder_input, prev_hidden_states,decoder_hidden)
		attention_vectors.append(attention_vector)
		if use_teacher_forcing:
			decoder_input = target_tensor[:,di,:].contiguous().view(target_tensor.size(0),1,target_tensor.size(2)) # Teacher forcing 
		else:
			decoder_input = decoder_output

		decoder_output = decoder_output.view(decoder_output.size(0),decoder_output.size(2)) #change size from m X n X k to m X k
		if criterion!=None:
			loss += criterion(decoder_output, target_tensor[:,di,:])
		
		#_mse,per_sequence_mse_from_tensor(testtargets_arraytestpreds_array)
		logger.debug("Input to calculate_mse function = decoder_output_size = {}, target_tensor_size = {}".format(decoder_output.size(),target_tensor[:,di,:].size()))
		_mse=calculate_mse(decoder_output,target_tensor[:,di,:]) #inputs: decoder_output.size() =  target_tensor[:,di,:] =  batch_size X num_columns;  
		mse_predictions.append(_mse) #The MSELoss criterion doesn't for some reason seem to be giving the correct results so we use our own mean squared error criterion per batch.
		decoder_predictions[:,di,:]=decoder_output #predictions of the decoder for sequence step `di` on the validation data.
		logger.debug("prev_hidden_state.size before cat = {},decoder_hidden = {}".format(prev_hidden_states.size(),decoder_hidden.size()))
		if sliding_attention:
			prev_hidden_states=torch.cat((prev_hidden_states[:,1:,:],decoder_hidden[-1,:,:].unsqueeze(dim=0).permute(1,0,2)),dim=1) #for now just use decoder hidden state we can also use attentional_hidden state if required later.

		logger.debug("prev_hidden_state.size = {}".format(prev_hidden_states.size()))
	
	attention_tensor=torch.stack(attention_vectors)
	logger.debug("Attention Tensor Size = {}, Attention List Length = {}, Single Attention Vector Size = {}".format(attention_tensor.size(),len(attention_vectors),attention_vectors[0].size()))

	#Anomaly Threshold Stuff.
	anomaly_threshold=1.1*np.max(mse_predictions) #This is the anomaly threshold score we are going to use.
	mse_per_timestep=calculate_mse_tensor(decoder_predictions.detach().numpy(),target_tensor.cpu().detach().numpy()) #returns a list containing the mean-squared error per timestep.
	return loss.item(),mse_per_timestep,anomaly_threshold,decoder_predictions,attention_tensor

def testIters(encoder, decoder, testdata, numbatches, num_layers,SOS_TOKEN,teacher_forcing_ratio,logger,sliding_attention=False,hierattnmethod="mean",targets=None):
	"""
		@param encoder: The trained encoder network.
		@param decoder: The trained decoder network.
		@param testdata: Variable containing the pre-processed test data.
		@param SOS_TOKEN: -1
		@param targets: Is an optional argument that if passed actually allows us to isolate exactly which sequences do and don't contain encoder sequences.

		@return mean_squared_error: A list containing the mean squared error at each time-step. It is of length sequence_length*(testdata.shape[0] - 1)
		@return predictions: A tensor containing the predictions from the decoder.
		@return target_tensor: A tensor containing ground truth target time-series values.
	"""
	
	logger.debug("Size of Test Data = {}".format(testdata.size()))
	criterion=nn.MSELoss()
	mean_squared_error=list()
	#Predictions and attention_weights need to be converted into tensors later on.
	predictions=list()
	attention_weights=list()
	target_data=list()
	hiddenstsimmat=list()
	for idx,batch in enumerate(testdata.chunk(numbatches,dim=0)):
		input_tensor = batch[:-1,:,:]
		target_tensor = batch[1:,:,:]
		input_tensor = input_tensor.cuda()
		target_tensor = target_tensor.cuda()
		
		#loss,mean_squared_error_batch,anomaly_threshold,predictions_batch,attention_weights_batch= test(input_tensor,target_tensor,encoder,decoder,num_layers,criterion,SOS_TOKEN,teacher_forcing_ratio,logger,sliding_attention)
		loss,mean_squared_error_batch,predictions_batch,attention_weights_batch,hiddenstatesimilaritymatrix= testNew(input_tensor,target_tensor,encoder,decoder,num_layers,criterion,SOS_TOKEN,teacher_forcing_ratio,logger,sliding_attention,hierattnmethod=hierattnmethod,istest=True)
		hiddenstsimmat.append(hiddenstatesimilaritymatrix)

		mean_squared_error.extend(mean_squared_error_batch)
		predictions.append(predictions_batch)
		attention_weights.append(attention_weights_batch)
		target_data.append(target_tensor)
		
		logger.debug("Attention Weights Batch = {}, Target Tensor Batch = {}, Predictions Tensor Batch = {}".format(attention_weights_batch.size(),target_tensor.size(),predictions_batch.size()))

	#This is just for housekeeping and knowing which indices were actually evaluted by decoders.
	#target_columns=list()
	#for idx,batch in enumerate(targets.chunk(numbatches,dim=0)):
	#	_targets = batch[1:,:,:]  #Batch Size, Sequence Length, Num. Time-Series
	#	target_columns.append(_targets)
		
	#Convert List of Target Tensors and Attention Weights to tensors.
	total_attention_weights=sum([wt.size(1) for wt in attention_weights]) 
	total_target_data = sum([tgt.size(0) for tgt in target_data])
	#total_targets_ground_truth_columns=sum([tgt.size(0) for tgt in target_columns])

	target_tensor = torch.zeros(total_target_data,target_data[0].size(1),target_data[0].size(2))
	attention_tensor = torch.zeros(attention_weights[0].size(0),total_attention_weights,attention_weights[0].size(2),attention_weights[0].size(3))
	predictions_tensor = torch.zeros_like(target_tensor)
	#target_columns_tensor=torch.zeros(total_targets_ground_truth_columns,target_columns[0].size(1),target_columns[0].size(2))

	ctr=0
	for idx in range(len(target_data)):
		target_tensor[ctr:ctr+target_data[idx].size(0),:,:] = target_data[idx]
		attention_tensor[:,ctr:ctr+attention_weights[idx].size(1),:,:] = attention_weights[idx]
		predictions_tensor[ctr:ctr+target_data[idx].size(0),:,:] = predictions[idx]		
		#target_columns_tensor[ctr:ctr+target_data[idx].size(0),:,:] = target_columns[idx]
		logger.debug("Target Tensor Begin Idx = {}, End Idx = {} Size of Batch = {}".format(ctr,ctr+target_data[idx].size(0),target_data[idx].size()))
		logger.debug("Attention Tensor Begin Idx = {}, End Idx = {}, Size of Batch = {}".format(ctr,ctr+attention_weights[idx].size(1),attention_weights[idx].size()))
		#logger.info("Target Columns Tensor Begin Idx = {}, End Idx = {}, Size of Batch = {}".format(ctr,ctr+target_columns[idx].size(0),target_columns[idx].size()))
		logger.debug("==============")
		ctr+=target_data[idx].size(0)

	logger.info("Predictions Tensor Size {}, Attention Tensor Size = {}, len(mean_squared_error) = {}, Size of Test Data = {}, Target Tensor = {}".format(predictions_tensor.size(),attention_tensor.size(),len(mean_squared_error),testdata.size(),target_tensor.size()))
	hiddenstsimmat = torch.cat(hiddenstsimmat,dim=0) #concat a list of (batchsize,sequencelength,2) along dim 0.
	#return mean_squared_error,predictions_tensor,target_tensor,attention_tensor,target_columns_tensor
	return mean_squared_error,predictions_tensor,target_tensor,attention_tensor,hiddenstsimmat


def trainIters(encoder, decoder, inputdata,testdata, numbatches, num_layers,teacher_forcing_ratio,NUM_EPOCHS,SOS_TOKEN,logger,sliding_attention=False,hierattnmethod='mean',print_every=2, plot_every=2, learning_rate=0.001,early_stopping_threshold=35,validation_tolerance=1e-3):
	logger.info("Inside trainIters(), inputdata.size() = {}, testdata.size() = {}, numbatches = {}. teacher_forcing_ratio = {}, NUM_EPOCHS = {}, SOS_TOKEN = {}, learning_rate = {}".format(inputdata.size(),testdata.size(),numbatches,teacher_forcing_ratio,NUM_EPOCHS,SOS_TOKEN,learning_rate))
	plot_train_losses = []
	plot_val_losses = []
	print_loss_total = 0  # Reset every print_every
	plot_loss_total = 0  # Reset every plot_every

	encoder_optimizer = optim.RMSprop(encoder.parameters(), lr=learning_rate)
	decoder_optimizer = optim.RMSprop(decoder.parameters(), lr=learning_rate)
	criterion = nn.MSELoss()

	patience=0	#if patience exceeds 10 epochs then we early stop.
	val_mse=list()	
	for epoch in range(NUM_EPOCHS):
		epoch_val_loss=0.0
		mse_validation_pred_err_per_epoch=list()
		predictions_per_batch=list()
		targets_per_batch=list()
		anomaly_thresholds_per_batch=list()
		if teacher_forcing_ratio>0: #Ignore it if teacherforcing is disabled to begin with.
			teacher_forcing_ratio = 1.0 - (float(epoch)/NUM_EPOCHS)  #Linear Scheduled Sampling
 
		for idx,batch in enumerate(inputdata.chunk(numbatches,dim=0)) :
			logger.info("\n\nEpoch Number {} Batch number {}\n\n".format(epoch,idx))    
			input_tensor = batch[:-1]  #batch_size,seq_len,input_size
			target_tensor = batch[1:]  #batch_size,seq_len,input_size
			input_tensor = input_tensor.cuda()             ## CUDA
			target_tensor = target_tensor.cuda()           ## CUDA
    			      
			loss, batch_predictions = train(input_tensor, target_tensor, encoder,decoder,encoder_optimizer, decoder_optimizer,num_layers, criterion,teacher_forcing_ratio,SOS_TOKEN,logger,sliding_attention,hierattnmethod)
		
			predictions_per_batch.append(batch_predictions)
			targets_per_batch.append(target_tensor)

			plot_train_losses.append(loss)

			batch_val_loss=0	
			for idx,batch in enumerate(testdata.chunk(numbatches,dim=0)):
				val_input_tensor = batch[:-1,:,:]
				val_target_tensor = batch[1:,:]
				val_input_tensor = val_input_tensor.cuda()     ## CUDA
				val_target_tensor = val_target_tensor.cuda()   ## CUDA
				logger.debug("Val Input Tensor Size = {}, Val Target Tensor Size = {}".format(val_input_tensor.size(),val_target_tensor.size()))

				########## HARDCODING Anomalies Just to check if there is an effect ####################
				#val_target_tensor[500:700,5:8,2] = 2 # Adding Anomalies for values from 500 to 700 to decoder side.
				#val_input_tensor[500:700,:,2] = 500  #Adding Anomalies for values from 500 to 700
				#logger.info("Val Input Tensor 495 - 505 = {}".format(val_input_tensor[498:502,:,2]))

				#val_loss, mse_per_batch,anomaly_threshold,validation_predictions,attention_weights = test(val_input_tensor, val_target_tensor, encoder, decoder, num_layers,criterion,SOS_TOKEN,teacher_forcing_ratio,logger,sliding_attention) #We can consuder validation_batch_predictions as the predictions on the entire validation set for all sequence time-steps because we evaluate on the entire validation set for each batch.
				val_loss, mse_per_batch,validation_predictions,attention_weights = testNew(val_input_tensor, val_target_tensor, encoder, decoder, num_layers,criterion,SOS_TOKEN,teacher_forcing_ratio,logger,sliding_attention,hierattnmethod=hierattnmethod,istest=False)
				logger.debug("Attention Weights Size = {}".format(attention_weights.size()))

				mse_validation_pred_err_per_epoch.append(mse_per_batch)
				batch_val_loss+=val_loss
				#anomaly_thresholds_per_batch.append(anomaly_threshold)

				print_loss_total += loss
				plot_loss_total += loss

				logger.debug("Torch Attention Weights Per Chunk Size = {}".format(attention_weights.size()))

			plot_val_losses.append(batch_val_loss)

		epoch_val_loss = batch_val_loss #The last batch_val_loss in each epoch is the epoch_val_loss.

		targets_per_epoch = torch.cat(targets_per_batch,0) #concatenate list of tensors of ground truth data, each of size (batchsize,sequence_size,input_size) along the batch-size dimension.
		predictions_per_epoch = torch.cat(predictions_per_batch,0) #concatenate list of tensors, each of size (batchsize,sequence_size,input_size) along the batch-size dimension.
		logger.debug("Total number of batches per epoch = {}".format(idx))
		#per_epoch_anomaly_threshold=np.mean(anomaly_thresholds_per_batch)
		logger.debug("Epoch {}, Validation MSE = {}\n\n\n\n======================\n\n\n\n".format(epoch,epoch_val_loss))
		
		#Early Stopping Logic.
		if patience>=early_stopping_threshold:
			break
		elif len(val_mse)>0 and ((val_mse[-1] - epoch_val_loss) < validation_tolerance):
			patience+=1

		val_mse.append(epoch_val_loss)

	logger.info("Size of Target = {}, Size of Predictions = {}".format(targets_per_epoch.size(),predictions_per_epoch.size()))
	logger.info("Validation MSE = {}".format(np.mean(val_mse)))

	return plot_train_losses,plot_val_losses,epoch_val_loss,predictions_per_epoch,targets_per_epoch,validation_predictions,val_target_tensor,attention_weights #per_epoch_anomaly_threshold

