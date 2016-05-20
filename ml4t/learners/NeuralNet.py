import numpy as np
import json, sys

# Activation functions
class Linear(object):
        @staticmethod
        def fn(x):
                """Return the linear activation unit."""
                return x

        @staticmethod
        def prime(x):
                """Return the derivative of the linear activation unit."""
                return 1.

class Sigmoid(object):
	@staticmethod
	def fn(x):
		"""Return the sigmoid activation unit."""
		expone = np.nan_to_num(np.exp(-x))
		return np.array( 1./(1.-expone) )
	
	@staticmethod
	def prime(x):
		"""Return the derivative of the sigmoid activation unit."""
		return Sigmoid.fn(x)*(1.-Sigmoid.fn(x))
	
class Tanh(object):
        @staticmethod
        def fn(x):
		"""Return the hyperbolic tangent activation unit."""
		pos = np.nan_to_num( np.exp(x) )
                neg = np.nan_to_num( np.exp(-x) )
                return (pos-neg)/(pos+neg)
	
	@staticmethod
	def prime(x):
		"""Return the derivative of the hyperbolic tangent activation unit."""
		return (1-Tanh.fn(x)**2)
	
class ReLU(object):
	@staticmethod
	def fn(x):
		"""Return the rectified linear activation unit."""
		return np.maximum(x,0.01)
	
	@staticmethod
	def prime(x):
		"""Return the derivative of the sigmoid activation unit."""
		return np.maximum(0.01, x>0)

class Softmax(object):
	@staticmethod
	def fn(x):
		"""Return the softmax activation unit."""
		return np.nan_to_num(np.exp(x))/np.sum(np.exp(x), axis=0)
	
	@staticmethod
	def prime(x):
		"""Return the derivative of the sigmoid activation unit."""
		return np.multiply( -Softmax.fn(x),Softmax.fn(x) )

# Cost functions
class QuadraticCost(object):
	@staticmethod
	def fn(a, y):
		""" """
		return 0.5*np.linalg.norm(a-y)**2
	
	@staticmethod
	def delta(z, a, y, activation=Linear):
		""" """
		return (a-y)*activation.prime(z)

class CrossEntropyCost(object):
    """Cross-Entropy Cost function and delta for output layer."""
    @staticmethod
    def fn(a, y):
        return np.sum( np.nan_to_num( -y*np.log(a)-(1-y)*np.log(1-a) ) )

    @staticmethod
    def delta(z, a, y, activation=Linear):
        return (a-y)

class Network(object):
	def __init__(self, sizes, cost=QuadraticCost, activations=None):
		self.cost = cost
		self.sizes = sizes
		self.num_layers= len(sizes)
		self.initialize_weights()
		if activations==None:
			self.activations = [ReLU for i in sizes[1:]]
		elif not len(activations) == len(sizes[1:]):
			print "You need {} activations.".format( len(sizes[1:]) )
			exit()
		else:
                        self.activations = activations

	def initialize_weights(self):
                self.biases = [np.random.randn(1, y) for y in self.sizes[1:]]
		self.weights = [np.random.randn(x, y)/np.sqrt(x)
                                for x,y in zip(self.sizes[:-1], self.sizes[1:])]
		
	def forward(self, a):
		for act, w, b  in zip(self.activations, self.weights, self.biases):
                        z = np.dot(a, w) + b
			a = act.fn( z )
		return a
		
	def backprop(self, x, y):
		n_w = [np.zeros(w.shape) for w in self.weights]
		n_b = [np.zeros(b.shape) for b in self.biases]
		
		# Feed-Forward Pass
		z_s = []
		a_s = [x]
		a = x
		for w, b, act in zip(self.weights, self.biases, self.activations):
			z = np.dot(a, w) + b
			z_s.append(z)
			a = act.fn(z)
			a_s.append(a)
		
		# Feed-Backward Pass
		delta = (self.cost).delta(z_s[-1], a_s[-1], y, self.activations[-1])
		n_b[-1] = delta.mean(axis=0)
		n_b[-1] = n_b[-1].reshape(1, delta.shape[1])
                n_w[-1] = np.dot(a_s[-2].transpose(), delta)
                for l in xrange(2,self.num_layers):
                        z = z_s[-l]
                        sp = self.activations[-l].prime(z)
                        delta = np.dot(delta, self.weights[-l+1].transpose())
                        delta *= sp
                        n_b[-l] = delta.mean(axis=0)
                        n_b[-l] = n_b[-l].reshape(1, delta.shape[1])
                        a = np.array(a_s[-l-1])
                        n_w[-l] = np.dot(a.transpose(), delta)
		return (n_b, n_w)
	
	def sgd(self, trainX, trainY, iterations=10000):
                eta = 0.000001
                # for adjusting learning rate (eta)
                old_costs = []
                # for stopping criteria
                running_cost = []
                # for reset criteria
                init_cost = QuadraticCost.fn(self.forward(trainX), trainY)
                for _ in range(iterations):
                        n_b, n_w = self.backprop(trainX, trainY)
                        self.weights = [w - eta*dnw for w, dnw in zip(self.weights, n_w)]
                        self.biases = [b - eta*dnb for b, dnb in zip(self.biases, n_b)]
                        new_cost = QuadraticCost.fn(self.forward(trainX), trainY)
                        # adjust learning rate (eta)
                        eta = self.adjust_eta(eta, old_costs, new_cost)
                        old_costs.append(new_cost)
                        # Assess learning
                        if _%(iterations/10)==0:
                                print "Cost at iter {}: {}".format(_,new_cost)
                                print "learning rate: {}\t".format(eta)
                        # Assess stopping criteria
                        if len(running_cost)>=100:
                                if np.std(running_cost)<1e-6 and np.mean(running_cost)>new_cost: break
                                running_cost.pop(0)
                        running_cost.append(new_cost)
                        # Assess reset criteria
                        if new_cost > 2*init_cost: self.initialize_weights()
                        

        def adjust_eta(self, eta, old_costs, new_cost):
                len_reached = len(old_costs)>=10
                if len(old_costs)<1:
                        return eta
                elif np.mean(old_costs)<new_cost:
                        if len_reached:
                                old_costs.pop(0)
                        return eta*0.5 + 1e-6
                else:
                        if len_reached:
                                old_costs.pop(0)
                        return eta*1.1

	def save(self, filename):
                """Save the neural network to the file ``filename``."""
                data = {"sizes": self.sizes,
                        "weights": [w.tolist() for w in self.weights],
                        "biases": [b.tolist() for b in self.biases],
                        "cost": str(self.cost.__name__),
                        "activations": [str(act.__name__) for act in self.activations]}
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
        activations = [getattr(sys.modules[__name__], act) for act in data["activations"]]
        cost = getattr(sys.modules[__name__], data["cost"])
        net = Network(data["sizes"], cost=cost, activations=activations)
        net.weights = [np.array(w) for w in data["weights"]]
        net.biases = [np.array(b) for b in data["biases"]]
        return net


trainX=np.array(
        [[0.],
         [1.],
         [2.],
         [3.],
         [4.]]
        )

trainY=np.array(
        [[0., 0., 0.],
         [1., 2., 3.],
         [4., 4., 6.],
         [9., 6., 9.],
         [16., 8., 12.]]
        )

trainY=np.array(
        [[1., 0., 0.],
         [0., 1., 0.],
         [1., 0., 0.],
         [0., 1., 0.],
         [1., 0., 0.]]
        )
acts = [ReLU,ReLU]

feature_size = trainX.shape[1]
output_size = trainY.shape[1]
net = Network([feature_size,
               100,
               output_size],cost=CrossEntropyCost,
              activations=acts)
print net.forward(trainX)
bs, ws = net.backprop(trainX, trainY)
net.sgd(trainX, trainY, iterations=100000)
print np.round(net.forward(trainX))

