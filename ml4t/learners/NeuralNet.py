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
		return 1

class Sigmoid(object):
	@staticmethod
	def fn(x):
		"""Return the sigmoid activation unit."""
		return 1./(1.-np.exp(-x))
	
	@staticmethod
	def prime(x):
		"""Return the derivative of the sigmoid activation unit."""
		return fn(x)*(1.-fn(x))
		
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
	def fn(a,y):
		""" """
		return 0.5*np.linalg.norm(a-y)**2
	
	@staticmethod
	def delta(a, y, activation=Linear.prime):
		""" """
		return (a-y)*activation(z)
		(a-y)*activation.prime(z)
	
class Network(object):
	def __init__(self, sizes, cost=QuadraticCost, activations=None):
		self.weights = [np.random.randn(x, y)/np.sqrt(y) for x,y in zip(sizes[:-1], sizes[1:])]
		self.biases = [np.random.randn(1, y) for y in sizes[1:]]
		self.cost = cost
		if activations==None:
			self.activations = [Linear.fn for i in sizes[1:]]
		elif not len(activations) == len(sizes[1:]):
			print "You need {} activations.".format( len(sizes[1:]) )
			exit()
	
	def forward(self, a):
		for act, w, b  in zip(self.activations, self.weights, self.biases):
			a = act( np.dot(a, w) + b )
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
			a = act(z)
			a_s.append(a)
			
		# Feed-Backward Pass
		delta = (self.cost).delta(zs[-1], activations[-1], y)
		n_b[-1] = delta
        n_w[-1] = np.dot(delta, a_s[-2].transpose())
		pass
	
	def sgd(self):
		pass
	


trainX=[[0.0, 1.0],
		[3.2, 2.4],
		[1.2, 3.1],
		[5.4, 4.3],
		[2.3, 2.4]]
net = Network([2,200,1])
print net.forward(trainX)