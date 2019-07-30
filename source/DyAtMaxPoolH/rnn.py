import torch
import torch.nn as nn
from torch.autograd import Variable
import torch.optim as optim
import torch.nn.functional as F
import pdb

#SLIDING_WINSIZE=90

class EncoderHierAttn(nn.Module):
    def __init__(self, input_size,hidden_size,sequence_length,num_layers,rnnobject,logger):
        """

        @param input_size: Number of Features in input data instance.
        @param hidden_size: Size of each hidden layer.
        @param num_layers: Number of Recurrent (LSTM or GRU) layers.
        @param rnnobject: one of ['RNN','GRU','LSTM']

        """
        super(EncoderHierAttn, self).__init__()
        self.input_size = input_size
        self.hidden_size = hidden_size
        self.sequence_length=sequence_length
        assert rnnobject in ["GRU","RNN","LSTM"], "RNNObject should be one of ['GRU','RNN','LSTM'], but got {}".format(rnnobject)
        self.rnnobject=rnnobject
        self.rnn = getattr(nn,rnnobject)(input_size, hidden_size,num_layers,batch_first=True)
        self.logger = logger

    def forward(self, input, hidden,cellstate=None):
        #print("in encoder.forward size of input",input.size(),"type of input = ",type(input),"type of hidden = ",type(hidden))
        if self.rnnobject=="LSTM":
            output,(hidden,cellstate) = self.rnn(input,(hidden,cellstate))
            return output,(hidden,cellstate)
        else:
            output, hidden = self.rnn(input, hidden)
            self.logger.debug("EncoderAttn.forward() Output.size() = {}, hidden.size() = {}".format(output.size(),hidden.size()))
            return output, hidden

    def initHidden(self,numlayers,batchsize):
        #This shouldn't have requires_grad = True as this is the encoder hidden layer.
        return Variable(torch.zeros(numlayers,batchsize, self.hidden_size).cuda(),requires_grad=False)

############### ENCODER END ###############################################

############### DECODER START ############################################
class DecoderHierAttn(nn.Module):
    def __init__(self, input_size,hidden_size,num_layers,output_size,rnnobject,encoder_sequence_length,decoder_sequence_length,sliding_winsize,dropoutpercentage,logger):
        super(DecoderHierAttn, self).__init__()
        self.input_size = input_size
        self.hidden_size = hidden_size
        self.encoder_sequence_length = encoder_sequence_length
        self.sequence_length = decoder_sequence_length
       	self.logger = logger
        self.sliding_windowsize=int(sliding_winsize)
        #self.attn is a linear layer which takes as input a vector of "self.hidden_size" i.e
        #self.attn_energies = nn.Linear(self.hidden_size , self.encoder_sequence_length)
        self.attn_energies = nn.Linear(self.hidden_size , self.sliding_windowsize)
        self.tanh=nn.Tanh()
        self.attn_vector = nn.Linear(self.hidden_size*2,self.hidden_size)
        self.hier_attn_vector = nn.Linear(self.hidden_size*2,self.hidden_size)
        self.rnnobject=rnnobject
        self.rnn = getattr(nn,rnnobject)(input_size,hidden_size,num_layers,batch_first=True)
        self.dropout=nn.Dropout(dropoutpercentage)
        self.attn_ctx = nn.Linear(self.hidden_size , self.hidden_size)
        self.forgetgate = nn.Linear(self.hidden_size*2,self.hidden_size)
        self.sigmoid = nn.Sigmoid()
        self.out = nn.Linear(self.hidden_size, output_size)
        self.contextGen = nn.Linear(self.hidden_size, 1)
        self.contextGen2 = nn.Linear(self.hidden_size, self.hidden_size)
        self.maxpool = nn.MaxPool1d(self.encoder_sequence_length)

    def forward(self, input, hidden_states,hidden_prev,contextvector,hierattnmethod='mean',cell_state=None):
        """
            @param: input: The input to the decoder unit.
            @param: hidden_prev: The previous hidden state.
            @param: hidden_states: The previous `n` hidden states passed to the decoder unit as this is what is used to calculate attention.
            @param: contextvector: This is the attentional vector calculated over all the encoder hidden states.
            @param: cell_state: Passed in the case of LSTM (cell state of the previous LSTM cell).
            @param: hierattnmethod: {mean,learn}. If mean, then the final hierarchical attentional vector is calculated as an element-wise mean of the contextvector and attentional vector of the
                current decoder unit. If set to `learn` a separate weight matrix is learned over the two attentional vectors to produce a third attentional vector.

            @return output: The multivariate time-series forecast vector returned.
            @return hidden_t: The hidden state at time t returned from the RNN unit.
            @return cellstate (optional): Returned from an LSTM unit.
            @return h_t : The `attentional hidden state` (i.e. hidden state at time `t` of a decoder unit, after application of attention.)
            @return a_t : The attention distribution which is used to create the context vector.
            @return hier_h_t : This is the attentional vector obtained from either the `mean` or `learn` method.
        """
        self.logger.debug("Input Size = {}".format(input.size()))
        self.logger.debug("Hidden States = {}".format(hidden_states.size()))
       	self.logger.debug("Hidden Prev = {}".format(hidden_prev.size()))
        ctx2gen=self.contextGen2(contextvector).permute(0,2,1)
        _maxpool = self.maxpool(ctx2gen)
        contextvector=_maxpool.permute(0,2,1)
        #contextvector = self.maxpool(self.contextGen2(contextvector).permute(0, 2, 1)).permute(0, 2, 1)
        #pdb.set_trace()
        if self.rnnobject=="LSTM":
            output, (hidden_t,cellstate) = self.rnn(input,(hidden_prev,cellstate))
            a_t = F.softmax(self.attn_energies(hidden_t[-1,:,:]),dim=2)  #Attention weights of t^th decoder unit. Size = (num_layers*num_directions X batch_size X seq_len)
            a_t = a_t.permute(1,0,2)     #In order to get the context vector, we need a_t to be (batch_size X 1 X seq_len) so we use permute [1,0,2]
            self.logger.debug("size of a_t = {}, size of hidden_states = {}".format(a_t.size(),hidden_states.size()))
            c_t = torch.bmm(a_t,hidden_states)   #Check if this is the same as a weighted average. If not we need to change this.
            self.logger.debug("Size of context vector.size() = {}, hidden_t.size() = {}".format(c_t.size(),hidden_t[-1,:,:].unsqueeze(dim=0).size()))
            self.logger.debug("Size of torch.cat.size() = {}".format(torch.cat((c_t,hidden_t[-1,:,:].unsqueeze(dim=0).permute(1,0,2)),dim=2).size()))
            h_t = self.tanh(self.attn_vector(torch.cat((c_t,hidden_t[-1,:,:].unsqueeze(dim=0).permute(1,0,2)),dim=2))) #concatenate along third dimension which is the hidden state. Both tensors being concatenated are of size = (batch_size,1,hidden_size).
            if (hierattnmethod=='mean') :
                hier_h_t = torch.mean(torch.cat((contextvector,h_t),dim=1),dim=1) #Element-wise mean of the calculated attentional_vector and the encoder context vector.

            elif (hierattnmethod=='learn'):
                self.logger.debug("Size of encoder context vector = {}, attentional hidden state size = {}".format(contextvector.size(),h_t.size()))
                self.logger.debug("Size of concatenated vector = {}".format(torch.cat((contextvector,h_t),dim=2)))
                hier_h_t = self.tanh(self.hier_attn_vector(torch.cat((contextvector,h_t),dim=2)))

            h_t_drp=self.dropout(hier_h_t)  #dropout.
            self.logger.debug("Size of h_t_drp = {}".format(h_t_drp.size()))
            output = self.out(h_t_drp)  #output
            #self.logger.info("Size of output = {}, Size of hidden = {}".format(output.size(),h_t_drp.size()))

            return output,(hidden_t,cellstate),h_t,a_t,hier_h_t


        else:
            output, hidden_t = self.rnn(input,hidden_prev)
            self.logger.debug("Size of Hidden States = {}".format(hidden_states.size()))
            #pdb.set_trace()
            a_t = F.softmax(torch.bmm(hidden_states,
                                      self.attn_ctx(hidden_t[-1,:,:]).unsqueeze(dim=2)),
                            dim=1)  #Attention weights of t^th decoder unit. Size = (num_layers*num_directions X batch_size X seq_len)
            #a_t = F.softmax(self.attn_energies(hidden_t[-1,:,:].unsqueeze(dim=0)),dim=2)    #Attention weights of t^th decoder unit. hidden_t dim = (num_layers*num_directions X batch_size X seq_len) we only take the top most hidden layer.
            self.logger.debug("Softmax Attention Energy Size    = {} , Softmax of Attention Energy = {}".format(a_t.size(),a_t.data.cpu().numpy()[0,0,:].ravel().tolist()))
            a_t = a_t.permute(0,2,1)     #In order to get the context vector, we need a_t to be (batch_size X 1 X seq_len) so we use permute [1,0,2]
            self.logger.debug("size of a_t = {}, size of hidden_states = {}".format(a_t.size(),hidden_states.size()))

            c_t = torch.bmm(a_t,hidden_states)   #Check if this is the same as a weighted averrage. If not we need to change this.
            self.logger.debug("context vector.size() = {}, hidden_t.size() = {}".format(c_t.size(),hidden_t[-1,:,:].unsqueeze(dim=0).size()))
            self.logger.debug("Size of torch.cat.size() = {}".format(torch.cat((c_t,hidden_t[-1,:,:].unsqueeze(dim=0).permute(1,0,2)),dim=2).size()))
            h_t = self.tanh(self.attn_vector(torch.cat((c_t,hidden_t[-1,:,:].unsqueeze(dim=0).permute(1,0,2)),dim=2)))
            self.logger.debug("Size of h_t = {}".format(h_t.size()))
            if contextvector is not None:
                if hierattnmethod=='mean':
                    self.logger.debug("Size of encoder context vector = {}, attentional hidden state size = {}".format(contextvector.size(),h_t.size()))
                    self.logger.debug("Size of concatenated vector = {}".format(torch.cat((contextvector,h_t),dim=2)))
                    hier_h_t = torch.mean(torch.cat((contextvector,h_t),dim=1),dim=1) #Element-wise mean of the calculated attentional_vector and the encoder context vector.
                elif hierattnmethod=='learn':
                    self.logger.debug("Size of encoder context vector = {}, attentional hidden state size = {}".format(contextvector.size(),h_t.size()))
                    self.logger.debug("Size of concatenated vector = {}".format(torch.cat((contextvector,h_t),dim=2)))
                    forgetvec = self.sigmoid(self.forgetgate(torch.cat((contextvector, hidden_t[-1, :, :].unsqueeze(dim=0).permute(1, 0, 2)), dim=2))) ###Nikhil changed this line.
                    #forgetvec = self.sigmoid(self.forgetgate(torch.cat((c_t, hidden_t[-1, :, :].unsqueeze(dim=0).permute(1, 0, 2)), dim=2)))
                    hier_h_t = (1 - forgetvec) * contextvector + forgetvec * h_t
                    hier_h_t = self.tanh(self.hier_attn_vector(torch.cat((contextvector,h_t),dim=2)))
            else:
                hier_h_t = h_t #This condition is reached ONLY for the first decoder unit as there is no h+t (contextvector) calculated yet.
            h_t_drp=self.dropout(hier_h_t)  #dropout.
            self.logger.debug("Size of h_t_drp = {}".format(h_t_drp.size()))
            output = self.out(h_t_drp)  #output
            self.logger.debug("Size of output = {}, Size of hidden = {}".format(output.size(),h_t_drp.size()))

            return output,contextvector, hidden_t,h_t,a_t,hier_h_t

