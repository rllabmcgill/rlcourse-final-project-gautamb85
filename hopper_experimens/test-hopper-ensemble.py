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

weights='/misc/scratch03/reco/bhattaga/data/i-vectors/dnn_projects/RL/cloning-priority-hopper-action/weights/s2s_rsr_minloss_valincr.npz'

def model(demonstration,state):
  
  network={}
  print("Building network ...")
  #read the demonstration
  l_in= lasagne.layers.InputLayer((None,250,3), input_var=demonstration)
  n_batch,maxlen,_ = l_in.input_var.shape

  l_encf = lasagne.layers.GRULayer(l_in, num_units=100, name='GRUEncoder', 
      mask_input=None, gradient_steps=-1,grad_clipping=100,only_return_final=False)
  l_encb = lasagne.layers.GRULayer(l_in, num_units=100, name='GRUEncoderB', 
      mask_input=None, gradient_steps=-1,grad_clipping=100,backwards=True, only_return_final=False)

  l_concat = lasagne.layers.ConcatLayer([l_encf,l_encb],axis=2)

  #collect the demonstration embeddings
  demo_emb = lasagne.layers.get_output(l_concat)

  #forward prop a minibatch of states
  network['input'] = InputLayer(shape=(None,11), input_var = state)
  batchs,_ = network['input'].input_var.shape

  network['ff1'] = DenseLayer(network['input'],100,nonlinearity=lasagne.nonlinearities.tanh)

  network['ff2'] = DenseLayer(network['ff1'],100,nonlinearity=lasagne.nonlinearities.tanh)

  #need to repeat each of the embedded states seql times
  network['repeat'] = RepeatLayer(network['ff2'],250)
  #repd = lasagne.layers.get_output(network['repeat']).eval({X:x_test})

  #Attention processing
  #easiest thing to do is concatenate the state vector with the
  #demonstration along the feature axis and then produce an
  #unnormalized score
  network['pairs'] = ConcatLayer([l_concat,network['repeat']],axis=2)
  #pout = lasagne.layers.get_output(network['pairs']).eval({X:x_test,demonstration:demo})

  network['reshape-1'] = lasagne.layers.ReshapeLayer(network['pairs'],(-1,300))

  network['attention'] = ReshapeLayer(AttentionLayer(network['reshape-1'],num_units=1,demo_input=demo_emb),(batchs,200))
  #pout = lasagne.layers.get_output(network['attention']).eval({X:x_test,demonstration:demo})

  #concatenate the attention embedding with the state embedding
  network['combine'] = ConcatLayer([network['ff2'],network['attention']])

  network['ff3'] = DenseLayer(network['combine'],200,nonlinearity=lasagne.nonlinearities.tanh)

  network['output'] = DenseLayer(network['ff3'],3,nonlinearity=lasagne.nonlinearities.linear)

  return network

def main():

  env = gym.make('Hopper-v1')
  max_steps = 500
  #load a single expert demonstration
  data = np.load('/misc/data15/reco/bhattgau/Rnn/code/Code_/RL/homework/hw1/latest-data/hopper-action-train.npy')
  data = np.asarray(data,dtype='float32')
  data = np.reshape(data,(data.shape[0],data.shape[1],3)) 
  
  State = T.matrix(name='state',dtype='float32')
  Demo = T.tensor3(name='demo',dtype='float32')
  
  network = model(Demo,State)
  
  demo1 = np.asarray(random.sample(data,1),dtype='float32')
  demo2 = np.asarray(random.sample(data,1),dtype='float32')
  demo3 = np.asarray(random.sample(data,1),dtype='float32')
  demo4 = np.asarray(random.sample(data,1),dtype='float32')
  demo5 = np.asarray(random.sample(data,1),dtype='float32')
  
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

          noise = 0.01*np.random.standard_normal(size=(3,)).astype('float32')

          action1 = get_act(demo1+noise,obs)
          action2 = get_act(demo2+noise,obs)
          action3 = get_act(demo3+noise,obs)
          action4 = get_act(demo4+noise,obs)
          action5 = get_act(demo5+noise,obs)
          
          action = (action1+action2+action3+action4+action5)/5.0
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

