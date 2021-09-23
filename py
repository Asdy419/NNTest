import numpy

import matplotlib.pyplot
#scipy.scpecial is for the sigmoid function yes
import scipy.special


class NN:
  def __init__(self,inputnodes,hiddennodes,outputnodes,learningrate):
    self.inodes = inputnodes
    self.hnodes = hiddennodes
    self.onodes = outputnodes
    #something to do with weights init
    self.wih = numpy.random.normal(0.0, pow(self.inodes,-0.5),(self.hnodes,self.inodes))
    self.who = numpy.random.normal(0.0, pow(self.hnodes,-0.5),(self.onodes,self.hnodes))
    
    self.lr = learningrate

    self.activation_function = lambda x: scipy.special.expit(x)


  def train(self,inputs_list,targets_list):
    #turn lists into 2d arary mhm
    inputs = numpy.array(inputs_list,ndmin=2).T
    targets = numpy.array(targets_list,ndmin=2).T

    #signals into hidden layers
    hidden_inputs = numpy.dot(self.wih,inputs)
    #signals coming out of the hidden layers
    hidden_outputs = self.activation_function(final_inputs)

    #calculate signals into the final output layer
    final_inputs = numpy.dot(self.who,hidden_outputs)
    #signals going out of the final output layers
    final_outputs = self.activation_function(final_inputs)

    #output layer error is the goal - actual
    output_errors = targets - final_outputs
    #hidden layer error is the output_errors, split in by weights
    hidden_errors = numpy.dot(self.who.T, output_errors)

    # update the weights for the links between hidden and output layers
    self.who += self.lr*numpy.dot((output_errors*final_outputs*(1.0-final_outputs)),numpy.transpose(hidden_outputs))

    pass

  def query(self,inputs_list):
    #turns inputs into 2d array
    inputs = numpy.array(inputs_list, ndmin = 2).T
    #calculates signals going into the hidden layer
    hidden_inputs = numpy.dot(self.wih,inputs)
      
    #calculate signals coming from hidden layer
    hidden_outputs = self.activation_function(hidden_inputs)


    #signals going into the final output layer
    final_inputs = numpy.dot(self.who,hidden_outputs)
    #calculates the signals coming out of the final output layer
    final_outputs = self.activation_function(final_inputs)

    print(final_outputs)
    return final_outputs
    

    

inputnodes = 3
outputnodes = 3
hiddennodes = 3
learningrate = 0.3
n = NN(inputnodes,hiddennodes,outputnodes,learningrate)

n.query([1.0,0.5,-1.5])


data_file = open("mnist_test.csv",'r')
data_list = data_file.readlines()
data_file.close()
