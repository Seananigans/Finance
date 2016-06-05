import numpy as np
from NeuralNet import *

def clear():
	if os.name == 'nt':
		os.system('cls')
	else:
		os.system('clear')



class StackedAE(object):
	def __init__(self, sizes, cost=QuadraticCost, lmbda=0.0, dropout=1.0):
                rev_sizes = sizes[::-1]
                stacked_sizes = [[i,j,i] for i,j in zip(sizes,sizes[1:])] + \
                                [[i,j,i] for i,j in zip(rev_sizes,rev_sizes[1:])]
                self.auto_encoders = [Network(sizes=i, cost=cost,
                                              lmbda=lmbda, dropout=dropout,
                                              activations=[ReLU for i in range(2)])
                                      for i in stacked_sizes]
                [net.initialize_weights() for net in self.auto_encoders]
		self.num_layers= len(sizes)

        def train(self, dataX, iterations=1e5):
                a = dataX
                for net in self.auto_encoders:
                        net.grad_descent(a, a, iterations=iterations)
                        a = self.step(a, net)

        def step(self, dataX, net):
                z = np.dot(dataX, net.weights[0]) + net.biases[0]
                a = net.activations[0].fn(z)
                return a

        def forward(self, dataX):
                a = dataX
                for net in self.auto_encoders:
                        a = self.step(a, net)
                return a

        def decode(self):
                a = dataX
                for net in self.auto_encoders[len(self.auto_encoders)/2:]:
                        a = self.step(a, net)
                return a

        def encode(self):
                a = dataX
                for net in self.auto_encoders[:len(self.auto_encoders)/2]:
                        a = self.step(a, net)
                return a
        
	def save(self, filename):
				"""Save the neural network to the file ``filename``."""
				data = {"sizes": self.sizes,
						"lmbda": self.lmbda,
						"dropout": self.dropout,
						"cost": str(self.cost.__name__),
						"activations": [str(act.__name__) for act in self.activations],
						"weights": [w.tolist() for w in self.weights],
						"biases": [b.tolist() for b in self.biases]}
				f = open(filename, "w")
				json.dump(data, f)
				f.close()

	def display_error(self, iter, cost, iterations):
		'''#Inform the user of the current error as iterations increase.'''
		print "Training {0:.5g}% Complete: Error = {1:.5g}\r".format(iter*100.0/iterations, cost)
		sys.stdout.write("\033[F") #back to previous line
		sys.stdout.write("\033[K") #clear line

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
		net = Network(
					sizes = data["sizes"], 
					cost = cost, 
					activations = activations,
					lmbda = data["lmbda"],
					dropout = data["dropout"])
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

feature_size = trainX.shape[1]
# Create test network
net = StackedAE([feature_size,2],
                cost=QuadraticCost,
                dropout=1.)
# Learn from data
net.train(trainX,  iterations=50000)
# Produce test output
pred = net.forward(trainX)
print pred

