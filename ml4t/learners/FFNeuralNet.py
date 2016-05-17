"""Numpy implementation of a basic neural network."""

import numpy as np
import json
import random
import sys
np.seterr(invalid='ignore')

class CrossEntropyCost(object):
    """Cross-Entropy Cost function and delta for output layer."""
    @staticmethod
    def fn(a, y):
        return np.sum( np.nan_to_num( -y*np.log(a)-(1-y)*np.log(1-a) ) )

    @staticmethod
    def delta(z, a, y):
        return (a-y)

class QuadraticCost(object):
    """Quadratic Cost function and delta for output layer."""
    @staticmethod
    def fn(a, y):
        return 0.5*np.linalg.norm(a-y)**2

    @staticmethod
    def delta(z, a, y):
        return (a-y)*l_relu(z, derivative=True)

class LeakyReLU(object):
    """Leaky Rectified Linear Unit."""
    @staticmethod
    def fn(z):
        gt_zero = (z>0)
        leq_zero = (z<=0)
        return np.multiply(z, gt_zero+leq_zero*0.01)

    @staticmethod
    def prime(z):
        return np.maximum(0.01, z>0)

class Sigmoid(object):
    """Sigmoid activation."""
    @staticmethod
    def fn(z):
        return 1./(1.-np.exp(-z))

    @staticmethod
    def prime(z):
        return fn(z)*(1-fn(z))

class Network(object):

    def __init__(self, sizes, cost = CrossEntropyCost):
        self.num_layers = len(sizes)
        self.sizes = sizes
        self.default_weight_initializer()
        self.cost = cost
        
    def default_weight_initializer(self):
        self.biases = [np.random.randn(y,1) for y in self.sizes[1:]]
        self.weights = [np.random.randn(y, x)/np.sqrt(x)
                        for x, y in zip(self.sizes[:-1], self.sizes[1:])]
        self.v_biases = [np.zeros((y, 1)) for y in self.sizes[1:]]
        self.v_weights = [np.zeros((y, x))
                         for x, y in zip(self.sizes[:-1], self.sizes[1:])]
    
    def feedforward(self, a):
        """Return the output of the network if "a" is input."""
        for b, w in zip(self.biases, self.weights):
            a = l_relu(np.dot(w, a)+b)
        return a

    def SGD(self, training_data, epochs,
            mini_batch_size, eta,
            lmbda=0.0, test_data=None):
        prev_acc = 0.0
        if test_data: n_test = len(test_data)
        n = len(training_data)
        for j in xrange(epochs):
            if eta<1./1024: break
            random.shuffle(training_data)
            mini_batches = [
                training_data[k:k+mini_batch_size]
                for k in xrange(0, n, mini_batch_size)]
            for mini_batch in mini_batches:
                self.update_mini_batch(mini_batch, eta, lmbda, n)
            new_acc = self.evaluate(test_data)
            if prev_acc>new_acc:
                eta *= 0.5
            else:
                eta *= 1.1
            prev_acc = new_acc
            if test_data:
                print "Epoch {0}: {1} / {2}".format(
                    j, new_acc, n_test)
            else:
                print "Epoch {0} complete".format(j)

    def update_mini_batch(self, mini_batch, eta, lmbda, n, mu=0.9):
        nabla_b = [np.zeros(b.shape) for b in self.biases]
        nabla_w = [np.zeros(w.shape) for w in self.weights]
        for x, y in mini_batch:
            #Nesterov Momentum
            # v' = mu*v - eta*nabla_Cost( w+mu*v )
            old_weights = [w for w in self.weights]
            old_biases = [b for b in self.biases]
            self.weights = [w + mu*vw
                              for w, vw in zip(self.weights, self.v_weights)]
            self.biases = [b + mu*vb
                              for b, vb in zip(self.biases, self.v_biases)]
            #end Nesterov
            delta_nabla_b, delta_nabla_w = self.backprop(x, y)
            #Return weights to original form
            self.weights = [w for w in old_weights]
            self.biases = [b for b in old_biases]
            #end modifications
            nabla_b = [nb+dnb for nb, dnb in zip(nabla_b, delta_nabla_b)]
            nabla_w = [nw+dnw for nw, dnw in zip(nabla_w, delta_nabla_w)]
        #Momentum method
        # v' = mu*v - (eta/len(mini_batch))*nw
        # w' = w + v'
        self.v_weights = [mu*vw - (eta/len(mini_batch))*nw
                        for vw, nw in zip(self.v_weights, nabla_w)]
        self.v_biases = [mu*vb-(eta/len(mini_batch))*nb
                       for vb, nb in zip(self.v_biases, nabla_b)]
        #end Momentum
        self.weights = [(1-eta*(lmbda/n))*w + vw
                        for w, vw in zip(self.weights, self.v_weights)]
        self.biases = [b + vb
                       for b, vb in zip(self.biases, self.v_biases)]

    def backprop(self, x, y):
        nabla_b = [np.zeros(b.shape) for b in self.biases]
        nabla_w = [np.zeros(w.shape) for w in self.weights]
        #feedforward
        activation = x
        activations = [x] # list to store all the activations, layer by layer
        zs = [] # list to store all the z vectors, layer by layer
        for b, w in zip(self.biases, self.weights):
            z = np.dot(w, activation)+b
            zs.append(z)
            activation = l_relu(z)
            activations.append(activation)
        #backward pass
        delta = (self.cost).delta(zs[-1], activations[-1], y)
        nabla_b[-1] = delta
        nabla_w[-1] = np.dot(delta, activations[-2].transpose())
        for l in xrange(2, self.num_layers):
            z = zs[-l]
            sp = l_relu(z, derivative=True)
            delta = np.dot(self.weights[-l+1].transpose(), delta) * sp
            nabla_b[-l] = delta
            nabla_w[-l] = np.dot(delta, activations[-l-1].transpose())
        return (nabla_b, nabla_w)

    def evaluate(self, test_data):
        test_results = [(np.argmax(self.feedforward(x)),y)
                         for (x, y) in test_data]
        return sum(int(x == y) for (x, y) in test_results)
                        
    def cost_derivative(self, output_activations, y):
        return (output_activations-y)

    def save(self, filename):
        """Save the neural network to the file ``filename``."""
        data = {"sizes": self.sizes,
                "weights": [w.tolist() for w in self.weights],
                "biases": [b.tolist() for b in self.biases],
                "cost": str(self.cost.__name__)}
        f = open(filename, "w")
        json.dump(data, f)
        f.close()

#### Loading a Network
def load(filename):
    """Load a neural network from the file ``filename``.  Returns an
    instance of Network.
    """
    f = open(filename, "r")
    data = json.load(f)
    f.close()
    cost = getattr(sys.modules[__name__], data["cost"])
    net = Network(data["sizes"], cost=cost)
    net.weights = [np.array(w) for w in data["weights"]]
    net.biases = [np.array(b) for b in data["biases"]]
    return net

def l_relu(z, derivative=False):
    """Calculate the Rectified Linear Unit for a given input."""
    if not derivative:
        gt_zero = (z>0)
        leq_zero = (z<=0)
        return np.multiply(z, gt_zero+leq_zero*0.01)
    else:
        gt_zero = (z>0)
        return np.maximum(0.01, gt_zero)
    
##from neural_networks_and_deep_learning_master.src import mnist_loader
##training_data, validation_data, test_data = mnist_loader.load_data_wrapper()
##
##filename = "test232342.txt"
##try:
##    net = load(filename)
##except IOError:
##    feature_size = len(training_data[0][0]) #784
##    net = Network([feature_size, 30, 10], cost=QuadraticCost)
##    
##print len(training_data),
##print training_data[0][0].shape
##print "There are {} layers in this network.".format(net.num_layers)
##print "The sizes of the layers are {}.".format(net.sizes)
##net.SGD(training_data,
##        epochs = 16, #Of times through training set
##        mini_batch_size = 5,
##        eta = 0.01, #Initial Learning Rate
##        lmbda = 0.0, #Regularization parameter
##        test_data=test_data)
##
##net.save(filename)
