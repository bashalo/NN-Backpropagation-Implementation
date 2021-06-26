"""
Created on Sun Jun 13 19:23:49 2021
@author: Yusuf Baran

"""

import numpy as np
from numpy import random

#--------------------------------Defining Functions---------------------------#

#--- Cost Function
def cost(pre, des): 
  
    res = 1/2*((abs(des-pre))**2)
    
    if(len(res) > 1):
        res = sum(res)
        
    return res

#--- Derivative of Cost Function
def der_cost(pre, des):
    return pre-des

#--- Tanh Activation Function
def tanh(x, derv = False):
    temp = (np.exp(x) - np.exp(-x)) / (np.exp(x) + np.exp(-x))
    
    if(not derv):
        return temp
    else:
        return 1 - temp**2

#--- Sigmoid Activation Function
def sigmoid(x, derv = False):
    
    temp = 1.0 / (1.0 + np.exp(-x))
    
    if (not derv):
        return temp
    else:
        return temp * (1-temp)
    
#--- Activation Switcher
def actFuncSwitcher(funcName = 'sigmoid'):
   
    return {
        
        'sigmoid' : sigmoid,
        'tanh'    : tanh,
        
        }[funcName]

#--- Function to randomly initialize weights and biases
def create_network(ınputs, expResult, nOfHidden, hidLayout): 
    
    nOfInput = len(ınputs)
    nOfOut = len(expResult)
    # If the number of nodes in the hidden layers is not given, 
    # it is set equal to the number of nodes in the input layer.
    if(type(hidLayout) == type(None)):
        hidLayout = np.array([nOfInput+2] * nOfHidden)
        
    #--- All Node Counts (Inputs - Hiddens - Outputs)
    allNodes = np.concatenate((nOfInput, hidLayout, nOfOut), axis=None)
    
    nLoop = len(allNodes)-1 
        
    #--- Initialize Weights and Biases
    allBiases = []
    allWeights = []
    for i in range(nLoop):
        allBiases.append(random.rand(allNodes[i+1]))
        allWeights.append(random.rand(allNodes[i], allNodes[i+1]))  
        
        
    return allWeights,allBiases,allNodes
        
#--- Calculate Forward Propagation
def forward(ınputs, allWeights, allBiases, actFunc):
    
    
    actIn = []
    actOut = [] 

    actIn.append(ınputs)
    for i in range(len(allBiases)):
        #---For Inputs
        if(i==0):
            temp = np.sum((allWeights[i].T * actIn[i]).T ,axis =0) + allBiases[i]
        else:
            temp = np.sum((allWeights[i].T * actOut[i-1]).T ,axis =0) + allBiases[i]
        
       
        actIn.append(temp)
        actOut.append(actFunc(temp))
        
    actOut = [ınputs] + actOut

        
    return actIn,actOut

#--- Calculate Back Propagation
def backward(allWeights, allBiases, expResult, actIn, actOut, actFunc):
    
    # Derivatives to be transferred to the next layer   
    propDer = [x-x+1 for x in allWeights]
    # Biase Derivatives
    biaseDivs = [x-x+1 for x in allBiases]
    # Weight Derivatives
    weightDivs = [x-x+1 for x in allWeights]

    
    #--- For output layer derivatives
    for i in range(len(allWeights), 0, -1):
        
        if(i == (len(allWeights))):
            #--- Calculate cost'(prediction - desired) * sig'(actIn^1,1)
            # and multiply all weight derivatives
            derv = der_cost(actOut[-1], expResult) * (actFunc(actIn[-1], derv = True))
            
            if(len(derv) > 1):
                temp = propDer[-1]
                for k in range(temp.shape[0]):
                    temp[k] = derv                    
                propDer[-1] = temp
                weightDivs[i-1] = temp * np.tile(actOut[-2],(temp.shape[1],1)).T
            else:
                weightDivs[i-1] = derv * actOut[-2]      
          
            
            biaseDivs[-1] = derv
                  
        else:       
            #---Multiplication of derivatives, 
            #--- act'(a1*w1+a2*w2+.....+b) and
            #--- weight values from the previous layer 
            temp = np.sum(allWeights[i]*propDer[i],axis=1) * actFunc(actIn[i], derv = True)
            
            #--- Biase derivatives calculated for the current layer in the loop
            biaseDivs [i-1] = temp
            temp = np.tile(temp, (propDer[i-1].shape[0], 1))
            
            propDer[i-1] = propDer[i-1]*temp
            #--- Weight derivatives calculated for the current layer in the loop
            weightDivs[i-1] = temp * np.tile(actOut[i-1],(temp.shape[1],1)).T
                
    return weightDivs, biaseDivs
                              
#--- Update Parameters               
def update (weightDivs, biaseDivs, allWeights, allBiases, learnRate):    
    #--- İf output layer has a one nodes
    if(allWeights[-1].shape[0] > 1 and len(allBiases[-1]) == 1):
        
        allWeights[-1] = allWeights[-1].T
        
        for i in range(len(weightDivs)):            
            allWeights[i] = allWeights[i] - (learnRate * weightDivs[i])
            allBiases[i] = allBiases[i] - (learnRate * biaseDivs[i])
            
        allWeights[-1] = allWeights[-1].T     
        
    else:
        
        for i in range(len(weightDivs)):            
            allWeights[i] = allWeights[i] - (learnRate * weightDivs[i])
            allBiases[i] = allBiases[i] - (learnRate * biaseDivs[i])



    return allWeights, allBiases

#--- Training Network
#-- Training continues under two conditions.
#-- Training is stopped, whichever is provided first.
#-- In other words, if the desired error value cannot be achieved within 
#- the given iteration value, the training is stopped.

#' @param ınputs Inputs
#' @param expResult Expected Results
#' @param nOfHidden Sets the number of hidden layers if the layer structure is not given
#' @param epochs  If the first criterion is not met, the value to be checked, 
#'                that is, the maximum number of iterations
#' @param learnRate learning rate
#' @param errorThres the first criterion for the algorithm to stop
#' @param activation determines the activation function. ('tanh' or 'sigmoid')            
#' @param hidLayout It determines the structure of hidden layers.If no value is 
#'                  given, it creates 3 layers with two more nodes than the number of inputs.
#' @param verbose  sets the information messages

def trainNet (ınputs,
              expResult,
              nOfHidden = 3, 
              epochs = 1000,
              learnRate = 0.5, 
              errorThres = 1e-3, 
              activation = 'sigmoid', 
              hidLayout = None,
              verbose = False):   
    
    #--- Selected Activation Function
    actFunc = actFuncSwitcher(funcName = activation)
    #--- Creating Network
    net = create_network(ınputs, expResult, nOfHidden, hidLayout)
    allWeights = net[0]
    allBiases = net[1]
    #--- 
    ıo = forward(ınputs, allWeights, allBiases, actFunc)

  
    if(verbose): print('\n'+'\33[41m' + ' >>>>>>>>>>>>>>>>>>>> Training Starting <<<<<<<<<<<<<<<<<<<< \n' + '\033[0m')

    err = 1
    epoch = 1
    # It works until the desired error level is achieved or the number of epochs.
    while (err > errorThres and epoch <= epochs):
        
        #--- Back Propagation
        bp = backward(allWeights = allWeights,
                      allBiases= allBiases,
                      expResult = expResult, 
                      actIn = ıo[0], 
                      actOut = ıo[1],                  
                      actFunc = actFunc)  
        
        #--- Updating Parameters
        up = update(weightDivs = bp[0], 
                    biaseDivs = bp[1], 
                    allWeights = allWeights, 
                    allBiases = allBiases, 
                    learnRate = learnRate)  
        
        #--- Make Prediction with updated parameters      
        allBiases = up[1]    
        allWeights = up[0]
        ıo = forward(ınputs = ınputs, 
                     allWeights = allWeights, 
                     allBiases = allBiases, 
                     actFunc = actFunc)    
        
        prediction = ıo[1][-1]
    
        err = cost(prediction, expResult)
    
        if(verbose and (epoch % 100 == 0)): print('\33[36m' + ' -->> Epoch = %d, learning Rate = %.2f, Error = %.10f' % (epoch, learnRate, err) )
 
        epoch = epoch +1
        
    #--- Printing Results
    if(verbose):
        print('\033[0m')
        print('\n'+'\33[41m' + ' >>>>>>>>>>>>>>>>>>>>>>> Training Results <<<<<<<<<<<<<<<<<<<<<<< \n' + '\033[0m')
        print('\033[36m' + ' -->> Error : %.8f\n' %(err))
        for i in range(len(expResult)):
            print('\33[36m' + ' -->> Exp Result_%d = %.4f' % (i+1,expResult[i]) ,'\33[33m',' Prediction_%d = %.4f' % (i+1,ıo[1][-1][i]) )
                
    
    return allWeights, allBiases, actFunc, err

#-------------------------------- Example ------------------------------------#        
#--- Example  
# This example creates 15 random ınput and outputs
# The number of hidden layers is 15 and there are 20 neurons in each layer.
ınputs = random.rand(15)
expResult = random.rand(15)
nOfHidden = 15
layout = np.repeat(20, nOfHidden)

#--- Train Network
trainRes =   trainNet(ınputs,
                      expResult,                      
                      learnRate = 0.9,
                      activation = 'sigmoid', 
                      hidLayout = layout,                      
                      epochs = 50000,
                      errorThres = 1e-10,
                      verbose = True)

#--- Make Prediction    
prediction = forward(ınputs, trainRes[0], trainRes[1], trainRes[2])
               
result = prediction[1][-1]

          
            
         
