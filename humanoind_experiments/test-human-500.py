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
from collections import OrderedDict
import h5py
from fuel.datasets import H5PYDataset
from lasagne.regularization import regularize_layer_params, l1
from lasagne.init import GlorotUniform,HeNormal
from lasagne.nonlinearities import TemperatureSoftmax
import gym
import random

class AttentionLayer(lasagne.layers.Layer):
    def __init__(self, incoming,num_units,demo_input=None,W=lasagne.init.Normal(0.01),
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
      attn_unorm = attn_unorm.reshape((-1,500))
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

weights='/misc/scratch03/reco/bhattaga/data/i-vectors/dnn_projects/RL/cloning-human500-v2/weights/s2s_rsr_minloss_valincr.npz'

def model(demonstration,state):
  
  network={}
  print("Building network ...")
  #read the demonstration
  l_in= lasagne.layers.InputLayer((None,500,17), input_var=demonstration)
  n_batch,maxlen,_ = l_in.input_var.shape

  l_encf = lasagne.layers.GRULayer(l_in, num_units=200, name='GRUEncoder', 
      mask_input=None, gradient_steps=100,grad_clipping=100,only_return_final=False)
  l_encb = lasagne.layers.GRULayer(l_in, num_units=200, name='GRUEncoderB', 
      mask_input=None, gradient_steps=100,grad_clipping=100,backwards=True, only_return_final=False)

  l_concat = lasagne.layers.ConcatLayer([l_encf,l_encb],axis=2)

#collect the demonstration embeddings
  demo_emb = lasagne.layers.get_output(l_concat)

#forward prop a minibatch of states
  network['input'] = InputLayer(shape=(None,376), input_var = state)
  batchs,_ = network['input'].input_var.shape

  network['ff1'] = DenseLayer(network['input'],100,nonlinearity=lasagne.nonlinearities.tanh)

  network['ff2'] = DenseLayer(network['ff1'],100,nonlinearity=lasagne.nonlinearities.tanh)

#need to repeat each of the embedded states seql times
  network['repeat'] = RepeatLayer(network['ff2'],500)
#repd = lasagne.layers.get_output(network['repeat']).eval({X:x_test})

#Attention processing
#easiest thing to do is concatenate the state vector with the
#demonstration along the feature axis and then produce an
#unnormalized score
  network['pairs'] = ConcatLayer([l_concat,network['repeat']],axis=2)
#pout = lasagne.layers.get_output(network['pairs']).eval({X:x_test,demonstration:demo})

  network['reshape-1'] = lasagne.layers.ReshapeLayer(network['pairs'],(-1,500))

  network['attention'] = ReshapeLayer(AttentionLayer(network['reshape-1'],num_units=1,demo_input=demo_emb),(batchs,400))

#concatenate the attention embedding with the state embedding
  network['combine'] = ConcatLayer([network['ff2'],network['attention']])

  network['ff3'] = DenseLayer(network['combine'],200,nonlinearity=lasagne.nonlinearities.tanh)

  network['output'] = DenseLayer(network['ff3'],17,nonlinearity=lasagne.nonlinearities.linear)

  return network

def main():

  env = gym.make('Humanoid-v1')
  max_steps = 250
  #load a single expert demonstration
  actions_train = np.load('/misc/data15/reco/bhattgau/Rnn/code/Code_/RL/homework/hw1/actions-train.npy')
  actions_train = np.asarray(actions_train,dtype='float32')

  demo_action = np.copy(actions_train)
  demo_action = demo_action[:49500,:]
  demo_action = np.reshape(demo_action,(-1,500,17)) 
  
  State = T.matrix(name='state',dtype='float32')
  Demo = T.tensor3(name='demo',dtype='float32')
  
  demo1 = np.asarray(random.sample(demo_action,1),dtype='float32')
  demo2 = np.asarray(random.sample(demo_action,1),dtype='float32')
  demo3 = np.asarray(random.sample(demo_action,1),dtype='float32')
  #demo4 = np.asarray(random.sample(demo_action,1),dtype='float32')
  #demo5 = np.asarray(random.sample(demo_action,1),dtype='float32')
  #demo6 = np.asarray(random.sample(demo_action,1),dtype='float32')
  #demo7 = np.asarray(random.sample(demo_action,1),dtype='float32')
  #demo8 = np.asarray(random.sample(demo_action,1),dtype='float32')
  #demo9 = np.asarray(random.sample(demo_action,1),dtype='float32')
  #demo10 = np.asarray(random.sample(demo_action,1),dtype='float32')
  
  
  network = model(Demo,State)
  
  print("Loading trained network")
  with np.load(weights) as f:
     param_values=[f['arr_%d' % i] for i in range(len(f.files))]
  lasagne.layers.set_all_param_values(network['output'],param_values)
  
  action = lasagne.layers.get_output(network['output'], deterministic=True)
  get_act = theano.function([Demo,State],action)

  returns = []
  for i in range(20):
      print('iter', i)
      obs = env.reset()
      done = False
      totalr = 0.
      steps = 0
      while not done:
          obs = np.asarray(obs,dtype='float32')
          obs = obs[None,:]
          
          noise = 0.001*np.random.standard_normal(size=(500,17)).astype('float32') 
          action1 = get_act(demo1+noise,obs)
          action2 = get_act(demo2+noise,obs)
          action3 = get_act(demo3+noise,obs)
          #action4 = get_act(demo4+noise,obs)
          #action5 = get_act(demo5,obs)
          #action6 = get_act(demo6+noise,obs)
          #action7 = get_act(demo7,obs)
          #action8 = get_act(demo8+noise,obs)
          #action9= get_act(demo9,obs)
          #action10 = get_act(demo10+noise,obs)
          
          action = (action1+action2+action3)/3.0
          #+action4+action5+action6+action7+action8+action9+action10)/10.0
          #action = np.cast(action)['float32']

          #action = np.asarray(action,dtype='float64')
          obs, r, done, _ = env.step(action)
          totalr += r
          steps += 1
          #if args.render:
              #env.render()
          if steps % 50 == 0: print("%i/%i"%(steps, max_steps))
          if steps >= max_steps:
              break
      returns.append(totalr)

  print('returns', returns)
  print('mean return', np.mean(returns))
  print('std of return', np.std(returns))

if __name__=="__main__":
  main()

