import os
import lasagne
from lasagne.layers import InputLayer, DenseLayer, batch_norm, ConcatLayer, DropoutLayer, BatchNormLayer, Conv2DLayer, DimshuffleLayer, MaxPool2DLayer, ReshapeLayer, NonlinearityLayer,GRULayer
import time
import theano
import theano.tensor as T
import matplotlib
import numpy as np
import matplotlib.pyplot as plt
import cPickle
import sys
import subprocess
import scipy.io
from fuel.datasets import IndexableDataset
from fuel.schemes import ShuffledScheme, SequentialScheme
from fuel.streams import DataStream
from lasagne.nonlinearities import TemperatureSoftmax
from collections import OrderedDict
import h5py
from fuel.datasets import H5PYDataset
from lasagne.regularization import regularize_layer_params, l1
from lasagne.init import GlorotUniform,HeNormal
from lasagne.nonlinearities import TemperatureSoftmax
import random
import pdb

def save(network, wts_path): 
  print('Saving Model ...')
  np.savez(wts_path, *lasagne.layers.get_all_param_values(network))


class AttentionLayer(lasagne.layers.Layer):
    def __init__(self, incoming,num_units,demo_input=None,W=lasagne.init.HeNormal(),
        b=lasagne.init.Constant(0.),**kwargs):
        super(AttentionLayer, self).__init__(incoming, **kwargs)

        num_inputs = self.input_shape[1]
        self.num_units=num_units
        self.W = self.add_param(W, (num_inputs, num_units), name='W_attention')
        self.b = self.add_param(b, (num_units,), name='b_attention')
          
        self.demo_input = demo_input

    def get_output_for(self, input, **kwargs):

      def apply_attention(a_emb,a_dense):
        attn_norm = T.nnet.softmax(a_dense)
        utt_emb = T.dot(attn_norm,a_emb)
        return utt_emb
      
      #compute unnormalized attention scores
      attn_unorm = lasagne.nonlinearities.tanh(T.dot(input,self.W)+self.b)
      attn_unorm = attn_unorm.reshape((-1,250))
      #return attn_unorm
      embs, _ = theano.scan(fn=apply_attention,outputs_info=None,sequences=[self.demo_input,attn_unorm])
      return embs

    def get_output_shape_for(self, input_shape):
        return (input_shape[0], self.num_units)


class RepeatLayer(lasagne.layers.Layer):
    
    def __init__(self, incoming, n, **kwargs):
        super(RepeatLayer, self).__init__(incoming, **kwargs)
        self.n = n

    def get_output_shape_for(self, input_shape):
        return tuple([input_shape[0], self.n] + list(input_shape[1:]))

    def get_output_for(self, input, **kwargs):
        #repeat the input n times
        tensors = [input]*self.n
        stacked = theano.tensor.stack(*tensors)
        dim = [1, 0] + range(2, input.ndim + 1)
        return stacked.dimshuffle(dim)

def iterate_minibatches(inputs, targets, batchsize, shuffle=False):
    assert len(inputs) == len(targets)
    if shuffle:
        indices = np.arange(len(inputs))
        np.random.shuffle(indices)
    
    for start_idx in range(0, len(inputs) - batchsize + 1, batchsize):
    
      if shuffle:
          excerpt = indices[start_idx:start_idx + batchsize]
      else:
          excerpt = slice(start_idx, start_idx + batchsize)
    
    yield inputs[excerpt], targets[excerpt]

##project path - make a new one each time
#one folder for the weigths
project_path='/misc/scratch03/reco/bhattaga/data/i-vectors/dnn_projects/RL/humanoid-random/'
wts_path = os.path.join(project_path,'weights')
epoch_path = os.path.join(project_path,'epoch_weights')

logfile = os.path.join(project_path,'uttRNN-train.log')

if os.path.exists(project_path):
  print('Project folder exits. Deleting...')
  command00 = "rm -r" +" "+ project_path
  process0 = subprocess.check_call(command00.split())
      
  command0 = "mkdir -p" +" "+ project_path
  process = subprocess.check_call(command0.split())
  command1 = "mkdir -p" +" "+ wts_path
  process1 = subprocess.check_call(command1.split())
  command2 = "mkdir -p" +" "+ epoch_path
  process2 = subprocess.check_call(command2.split())
else:
  print('Creating Project folder')
  command0 = "mkdir -p" +" "+ project_path
  process = subprocess.check_call(command0.split())
  command1 = "mkdir -p" +" "+ wts_path
  process1 = subprocess.check_call(command1.split())
  command2 = "mkdir -p" +" "+ epoch_path
  process2 = subprocess.check_call(command2.split())

network={}
x_test = np.random.normal(size=(10,376))
x_test = np.asarray(x_test,dtype='float32')

demo = np.random.normal(size=(10,500,376)).astype('float32')

X = T.matrix(name='features',dtype='float32')
targets = T.matrix(name='targets',dtype='float32')

demonstration = T.tensor3(name='demonstration',dtype='float32')

print("Building network ...")
#read the demonstration
l_in= lasagne.layers.InputLayer((None,250,393), input_var=demonstration)
n_batch,maxlen,_ = l_in.input_var.shape

l_encf = lasagne.layers.GRULayer(l_in, num_units=600, name='GRUEncoder', 
    mask_input=None, gradient_steps=-1,grad_clipping=100,only_return_final=False)
l_encb = lasagne.layers.GRULayer(l_in, num_units=600, name='GRUEncoderB', 
    mask_input=None, gradient_steps=-1,grad_clipping=100,backwards=True, only_return_final=False)

l_concat = lasagne.layers.ElemwiseSumLayer([l_encf,l_encb])

#collect the demonstration embeddings
demo_emb = lasagne.layers.get_output(l_concat)

#forward prop a minibatch of states
network['input'] = InputLayer(shape=(None,376), input_var = X)
batchs,_ = network['input'].input_var.shape

network['ff1'] = DenseLayer(network['input'],200,nonlinearity=lasagne.nonlinearities.tanh,W=lasagne.init.HeNormal())

network['ff2'] = DenseLayer(network['ff1'],200,nonlinearity=lasagne.nonlinearities.tanh,W=lasagne.init.HeNormal())

#need to repeat each of the embedded states seql times
network['repeat'] = RepeatLayer(network['ff2'],250)
#repd = lasagne.layers.get_output(network['repeat']).eval({X:x_test})

#Attention processing
#easiest thing to do is concatenate the state vector with the
#demonstration along the feature axis and then produce an
#unnormalized score
network['pairs'] = ConcatLayer([l_concat,network['repeat']],axis=2)
#pout = lasagne.layers.get_output(network['pairs']).eval({X:x_test,demonstration:demo})

network['reshape-1'] = lasagne.layers.ReshapeLayer(network['pairs'],(-1,800))

network['attention'] = ReshapeLayer(AttentionLayer(network['reshape-1'],num_units=1,demo_input=demo_emb),(batchs,600))
#pout = lasagne.layers.get_output(network['attention']).eval({X:x_test,demonstration:demo})

#concatenate the attention embedding with the state embedding
network['combine'] = ConcatLayer([network['ff2'],network['attention']])

network['ff3'] = DenseLayer(network['combine'],200,nonlinearity=lasagne.nonlinearities.tanh,W=lasagne.init.HeNormal())

network['output'] = DenseLayer(network['ff3'],17,nonlinearity=lasagne.nonlinearities.linear)

network_output = lasagne.layers.get_output(network['output'])

val_prediction = lasagne.layers.get_output(network['output'], deterministic=True)

total_cost = lasagne.objectives.squared_error(network_output, targets) #+ L1_penalty*1e-7
mean_cost = total_cost.mean()

#accuracy function
val_cost = lasagne.objectives.squared_error(val_prediction, targets) #+ L1_penalty*1e-7
val_mcost = val_cost.mean()

#Get parameters of both encoder and decoder
all_parameters = lasagne.layers.get_all_params(network['output'], trainable=True)

print("Trainable Model Parameters")
print("-"*40)
for param in all_parameters:
    print(param, param.get_value().shape)
print("-"*40)

all_grads = T.grad(mean_cost, all_parameters)
Learning_rate = 0.0075
learn_rate = theano.shared(np.array(Learning_rate, dtype='float32'))
lr_decay = np.array(0.1, dtype='float32')

updates = lasagne.updates.adam(all_grads, all_parameters, learn_rate)

train_func = theano.function([X,demonstration,targets], [mean_cost], updates=updates)

val_func = theano.function([X,demonstration, targets], [val_mcost])

######### data ####################

dtrain = np.load('/misc/data15/reco/bhattgau/Rnn/code/Code_/RL/homework/hw1/latest-data/humanoid/expert-states-humaniod-250.npy')
dtrain = np.asarray(dtrain, dtype='float32')
demo_train = dtrain[:200,:,:]
demo_valid = dtrain[200:,:,:]

trange = range(demo_train.shape[0])
vrange = range(demo_valid.shape[0])

states_train = np.copy(demo_train)
states_valid = np.copy(demo_valid)
states_train = np.reshape(states_train,(-1,376))
states_valid = np.reshape(states_valid,(-1,376))

atrain = np.load('/misc/data15/reco/bhattgau/Rnn/code/Code_/RL/homework/hw1/latest-data/humanoid/expert-actions-humanoid-250.npy')
atrain = np .asarray(atrain,dtype='float32')
atrain = np.reshape(atrain,(atrain.shape[0],atrain.shape[1],17))
actions_train = atrain[:200,:,:]
actions_valid = atrain[200:,:,:]

act_train = np.copy(actions_train)
act_valid = np.copy(actions_valid)
act_train = np.reshape(act_train,(-1,17))
act_valid = np.reshape(act_valid,(-1,17))

train_demos = IndexableDataset(indexables=OrderedDict([('states',states_train),('action-demo',act_train)]), axis_labels=OrderedDict([('states',('batch','feat_dim')),('action-demos',('batch','f_dim'))]))
valid_demos = IndexableDataset(indexables=OrderedDict([('states',states_valid),('action-demo',act_valid)]), axis_labels=OrderedDict([('demonstrations',('batch','feat_dim')),('action-demos',('batch','f_dim'))]))

min_val_loss = np.inf
val_prev = 1000

patience=0#patience counter
val_counter=0 
epoch=0
num_epochs=100

print("Starting training...")
    # We iterate over epochs:
while 'true':
    # In each epoch, we do a full pass over the training data:
    train_err = 0
    train_batches = 0

    h1=train_demos.open()
    h2=valid_demos.open()
    
    dscheme = ShuffledScheme(examples=train_demos.num_examples, batch_size=64)
    dscheme1 = SequentialScheme(examples=valid_demos.num_examples, batch_size=32)

    demo_train_stream = DataStream(dataset=train_demos, iteration_scheme=dscheme)
    demo_valid_stream = DataStream(dataset=valid_demos, iteration_scheme=dscheme1)

    start_time = time.time()

    for data in demo_train_stream.get_epoch_iterator():
        s_data,a_data = data
        
        dinds = random.sample(trange,s_data.shape[0])
        sdemos = [demo_train[i,:,:] for i in dinds]
        sdemos = np.asarray(sdemos,dtype='float32')
        sdemos = np.reshape(sdemos,(s_data.shape[0],250,376))

        ademos = [actions_train[i,:,:] for i in dinds]
        ademos = np.asarray(ademos,dtype='float32')
        ademos = np.reshape(ademos,(s_data.shape[0],250,17))

        demo = np.concatenate([sdemos,ademos],axis=2)
        print(demo.shape)

        terr = train_func(s_data,demo,a_data)
        train_err += terr[0]
        train_batches += 1

    val_err = 0
    val_batches = 0

    for data in demo_valid_stream.get_epoch_iterator():
      s_data,a_data = data  
      
      dinds = random.sample(vrange,s_data.shape[0])
      sdemos = [demo_valid[i,:,:] for i in dinds]
      sdemos = np.asarray(sdemos,dtype='float32')
      sdemos = np.reshape(sdemos,(s_data.shape[0],250,376))

      ademos = [actions_valid[i,:,:] for i in dinds]
      ademos = np.asarray(ademos,dtype='float32')
      ademos = np.reshape(ademos,(s_data.shape[0],250,17))

      demo = np.concatenate([sdemos,ademos],axis=2)
      print(demo.shape)

      verr = train_func(s_data,demo,a_data)
      val_err += verr[0]
      val_batches += 1

    epoch+=1
    train_demos.close(h1)
    valid_demos.close(h2)
    
    print("Epoch {} of {} took {:.3f}s Learning Rate {}".format(
          epoch, num_epochs, time.time() - start_time, learn_rate.get_value()))
    print("  training loss:{:.6f}, validation loss:{:.6f}".format((train_err / train_batches), (val_err / val_batches)))
     
    flog1 = open(logfile,'ab')
    flog1.write("Epoch {} of {} took {:.3f}s Learning rate {}\n".format(
        epoch, num_epochs, time.time() - start_time, learn_rate.get_value()))
    flog1.write("  training loss:{:.6f}, validation loss:{:.6f}\n".format((train_err / train_batches),(val_err / val_batches)))
      
    flog1.write("\n")
    flog1.close()
    

    valE = val_err/val_batches
    
    if valE <= min_val_loss:

      #save the network parameters corresponding to this loss
      min_loss_network = network['output'] 
      patience=0
      min_val_loss = valE
      mloss_epoch=epoch+1
      
      mname = 'uttCNN-weights-epoch-%d'%(epoch+1)
      spth = os.path.join(epoch_path,mname+'.npz')
      save(min_loss_network,spth)

    #Patience / Early stopping
    else:
      #increase the patience counter
      patience+=1
      #decrease the learning rate
      learn_rate.set_value(learn_rate.get_value()*lr_decay)
      spth = os.path.join(wts_path,'s2s_rsr_minloss_valincr.npz')
      save(min_loss_network,spth)

    if patience==5:
      break
   
    val_prev = valE

    if val_counter==10:
      spth = os.path.join(wts_path,'s2s_rsr_minloss_valincr.npz')
      save(min_loss_network,spth)
      break #break out
    
    if epoch == num_epochs: 
      spth = os.path.join(wts_path,'s2s_rsr_minloss_nepoch.npz')
      save(min_loss_network, spth)
      break

 

