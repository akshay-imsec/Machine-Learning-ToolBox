import numpy as np

#sigmoid
def sigmoid(x,deriv=False):
    #derivative
    if(deriv==True):
        return x*(1-x)
    #sigmoid function
    return 1/(1+np.exp(-x))

#input
X = np.array([  [0,0,1],
                [0,1,1],
                [1,0,1],
                [1,1,1] ])
#output    
Y = np.array([[0,1,1,0]]).T

#randomize numbers
np.random.seed(1)

#initialize weights 
syn0 = 2*np.random.random((3,4)) - 1
syn1 = 2*np.random.random((4,1)) - 1

for iter in xrange (60000):
    
    #feed forward
    layer0 = X
    layer1 = sigmoid(np.dot (layer0,syn0))
    layer2 = sigmoid(np.dot (layer1,syn1))

    #error
    layer2_error = Y - layer2
    if iter%10000 == 0:
        print "Error: "+str(np.mean(np.abs(layer2_error)))
    layer2_delta = layer2_error * sigmoid(layer2,True)

    layer1_error = layer2_delta.dot (syn1.T)

    layer1_delta = layer1_error * sigmoid(layer1,True)

    syn1 += layer1.T.dot (layer2_delta)
    syn0 += layer0.T.dot (layer1_delta)

print "Weights after training: "
print syn0,'\n',syn1
print "Output after training: "
print layer2
