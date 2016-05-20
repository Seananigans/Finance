import numpy as np

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
		return np.array( 1./(1.-np.exp(-x)) )
	
	@staticmethod
	def prime(x):
		"""Return the derivative of the sigmoid activation unit."""
		return Sigmoid.fn(x)*(1.-Sigmoid.fn(x))
		
class ReLU(object):
	@staticmethod
	def fn(x):
		"""Return the rectified linear activation unit."""
		gt_zero = (z>0)
		leq_zero = (z<=0)
		return np.multiply(z, gt_zero+leq_zero*0.01)
	
	@staticmethod
	def prime(x):
		"""Return the derivative of the sigmoid activation unit."""
		return np.maximum(0.01, x>0)

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
	
class Network(object):
	def __init__(self, sizes, cost=QuadraticCost, activations=None):
		self.biases = [np.random.randn(1, y) for y in sizes[1:]]
		self.weights = [np.random.randn(x, y)/np.sqrt(x)
                                for x,y in zip(sizes[:-1], sizes[1:])]
		self.cost = cost
		self.num_layers= len(sizes)
		if activations==None:
			self.activations = [Linear for i in sizes[1:]]
		elif not len(activations) == len(sizes[1:]):
			print "You need {} activations.".format( len(sizes[1:]) )
			exit()
		else:
                        self.activations = activations
	
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
		n_b[-1] = delta
                n_w[-1] = np.dot(a_s[-2].transpose(), delta)
                for l in xrange(2,self.num_layers):
                        z = z_s[-l]
                        sp = self.activations[-l].prime(z)
                        delta = np.dot(delta, self.weights[-l+1].transpose())
                        delta *= sp
                        n_b[-l] = delta
                        a = np.array(a_s[-l-1])
                        n_w[-l] = np.dot(a.transpose(), delta)
		return (n_b, n_w)
	
	def sgd(self):
		pass
	


trainX=[[0.0, 1.0],
        [3.2, 2.4],
        [1.2, 3.1],
        [5.4, 4.3],
        [2.3, 2.4]]

trainY=[[0.0],
        [3.2],
        [1.2],
        [5.4],
        [2.3]]
acts = [Sigmoid,Sigmoid]
net = Network([2,200,1],activations=acts)
print net.forward(trainX)
bs, ws = net.backprop(trainX, trainY)
print [b.shape for b in bs]
print [b.shape for b in net.biases]
print [b.shape for b in ws]
print [b.shape for b in net.weights]
