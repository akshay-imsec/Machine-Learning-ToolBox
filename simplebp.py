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
y = np.array([[0,0,1,1]]).T

#randomize numbers
np.random.seed(1)

#initialize weights 
syn0 = 2*np.random.random((3,1)) - 1

for iter in xrange(10000):

    # feed forward
    layer0 = X
    layer1 = sigmoid(np.dot(layer0,syn0))

    # error for each input
    layer1_error = y - layer1

    # multiply by slope of the sigmoid at the values in layer1
    layer1_delta = layer1_error * sigmoid(layer1,True)

    # update weights according to input matrix and delta matrix
    syn0 += np.dot(layer0.T,layer1_delta)

print "Output After Training:"
print layer1

print "Weights after Training:"
print syn0
