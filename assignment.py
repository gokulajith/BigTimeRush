import sys
import numpy as np
import torch as torch
import torch.nn as nn
import torch.nn.functional as F
import preprocess
import dgl
import networkx as nx


class Model(nn.Module):

	def __init__(self):
		super(Model, self).__init__()

		# Initialize hyperparameters
		self.raw_features = 10

		self.batch_size = 10
		# Initialize trainable parameters
		self.liftingLayer = nn.Linear(10, 300)
		self.MP1 = MPLayer(300, 300)
		self.MP2 = MPLayer(300, 300)
		self.MP3 = MPLayer(300, 300)
		self.readOutLayer = nn.Linear(300, 31)

		self.optimizer = torch.optim.Adam(self.parameters(), lr=0.0005)

	def forward(self, g):
		"""
		Responsible for computing the forward pass of your network. Analagous to
		"call" methods from previous assignments.

		1) The batched graph that you pass in hasn't had its node features lifted yet.
			Pop them, run them through your lifting layer. Don't apply an activation function. 
		2) After the node features for the graph have been lifted, run them through
				the mp layers (ReLUing the result returned from each).
		3) After ReLUing the result of your final mp layer, feed it through the
		   readout function in order to get logits.

		:param g: The DGL graph you wish to run inference on.
		:return: logits tensor of size (batch_size, 2)
		"""
		all_node_features = g.ndata.pop('features')
		lifted_features = self.liftingLayer(all_node_features)
		lifted_features = torch.relu(self.MP1.forward(g, lifted_features))
		lifted_features = torch.relu(self.MP2.forward(g, lifted_features))
		lifted_features = torch.relu(self.MP3.forward(g, lifted_features))
		logits = self.readout(g, lifted_features)
		return logits

	def readout(self, g, node_feats):
		""" 
		Responsible for reducing the dimensionality of the graph to
		num_classes, and summing the node features in order to return logits.

		Set your node features to be the output of your readout layer on node_feats,
		then use dgl.sum_nodes to return logits. 

		:param g: The batched DGL graph
		:param node_feats: The features at each node in the graph.
		:return: logits tensor of size (batch_size, 31)
		"""
		node_feats = self.readOutLayer(node_feats)
		g.ndata['features'] = node_feats
		logits = dgl.sum_nodes(g, 'features')
		return logits

	def accuracy_function(self, logits, labels):
		"""
		Computes the accuracy across a batch of logits and labels.

		:param logits: a 2-D np array of size (batch_size, 2)
		:param labels: a 1-D np array of size (batch_size) 
									(1 for if the play is active against cancer, else 0).
		:return: mean accuracy over batch.
		"""
		newlabels = label_converter(labels)
		for i in range(len(logits)):
			print("guess:", np.argmax(i))
			print("correct:", newlabels[i])
		return 0

class MPLayer(nn.Module):
	"""
	A PyTorch module designed to represent a single round of message passing.
	This should be instantiated in your Model class several times.
	"""

	def __init__(self, in_feats, out_feats):
		"""
		Init method for the MPLayer. You should make a layer that will be
		applied to all nodes as a final transformation when you've finished
		message passing that maps the features from size in_feats to out_feats (in case
		you want to change the dimensionality of your node vectors at between steps of message
		passing). You should also make another layer to be used in computing
		your messages.

		:param in_feats: The size of vectors at each node of your graph when you begin
		message passing for this round.
		:param out_feats: The size of vectors that you'd like to have at each of your
		nodes when you end message passing for this round.
		"""
		super(MPLayer, self).__init__()
		self.finalTransformation = nn.Linear(in_feats, out_feats)
		self.computeLayer = nn.Linear(in_feats, in_feats)


	def forward(self, g, node_feats):
		"""
		Responsible for computing the forward pass of your network. Analagous to
		"call" methods from previous assignments.

		1) You should reassign g's ndata field to be the node features that were popped off
			in the previous layer (node_feats).
		2) Trigger message passing and aggregation on g using the send and recv functions.
		3) Pop the node features, and then feed it through a linear layer and return.

		You can assign/retrieve node data by accessing the graph's ndata field with some attribute
		that you'd like to save the features under in the graph (e.g g.ndata["h"] = node_feats)

		:param g: The batched DGL graph you wish to run inference on.
		:param node_feats: Beginning node features for your graph. should be a torch tensor (float) of shape
		(number_atoms_batched_graph, in_feats).
		:return: node_features of size (number_atoms_batched_graph, out_feats)
		"""
		g.ndata['features'] = node_feats
		g.send(g.edges(), self.message)
		g.recv(g.nodes(), self.reduce)
		all_node_features = g.ndata.pop('features')
		final = self.finalTransformation(all_node_features)
		return final

	def message(self, edges):
		"""
		A function to be passed to g.send. This function, when called on a group of
		edges, should compute a message for each one of them. Each message will then be sent
		to the edge's "dst" node's mailbox.

		The particular rule to compute the message should be familiar. The message from
		node n1 with node feature v1 to n2 should be ReLU(f(v1)), where f is a feed-forward layer.

		:param edges: All the DGL edges in the batched DGL graph.
		:return: A map from some string (you choose) to all the messages
		computed for each edge. These messages can then be retrieved at
		each destination node's mailbox (e.g destination_node.mailbox["string_you_chose"]) 
		once DGL distributes them with the send function.
		"""
		return {'message': torch.relu(self.computeLayer(edges.src['features']))}


	def reduce(self, nodes):
		"""
		A function to be passed to g.recv. This function, when called on a group of nodes,
		should aggregate (ie. sum) all the messages received in their mailboxes from message passing.
		DGL will then save these new features in each node under the attribute you set (see the return).

		The messages in each node can be accessed like:
		nodes.mailbox['string_you_chose_in_message']

		:param nodes: All the DGL nodes in the batched DGL Graph.
		:return: A Map from string to the summed messages for each node.
		The string should be the same attribute you've been using to
		access ndata this whole time. The node data at this
		attribute will be updated to the summed messages by DGL.
		"""
		return {'features': torch.sum(nodes.mailbox['message'], dim=1)}

def build_graph(play):
	"""
	Constructs a DGL graph out of a play from the train/test data.

	:param play: a play object (see play.py for more info)
	:return: A DGL Graph with the same number of nodes as atoms in the play, edges connecting them,
	         and node features applied.
	"""
	# TODO: Initialize a DGL Graph
	graph = dgl.DGLGraph()
	# TODO: Call the graph's add_nodes method with the number of nodes in the play.
	graph.add_nodes(len(play.nodes))
	# TODO: Turn play's nodes into a tensor, and set it to be the data of this graph.
	tensor = torch.from_numpy(play.nodes)
	graph.ndata['features'] = tensor
	# TODO: Construct a tuple of src and dst nodes from the list of edges in plays.
	#      e.g if the edges of the play looked like [(1,2), (3,4), (5,6)] return
	#      (1,3,5) and (2,4,6).
	src = []
	dst = []
	for tuple in play.edges:
		src.append(tuple[0])
		dst.append(tuple[1])
	graph.add_edges(src, dst)
	graph.add_edges(dst, src)
	return graph

def train(model, train_data):
	"""
	Trains your model given the training data.

	For each batch of plays in train data...
		1) Make dgl graphs for each of the plays in your batch; collect them in a list.
		2) call dgl.batch to turn your list of graphs into a batched graph.
		3) Turn the labels of each of the plays in your batch into a 1-D tensor of size
		   batch_size  
		4) Pass this graph to the Model's forward pass. Run the resulting logits
				and the labels of the play batch through nn.CrossEntropyLoss.
		3) Zero the gradient of your optimizer.
		4) Do backprop on your loss.
		5) Take a step with the optimizer.

	Note that nn.CrossEntropyLoss expects LOGITS, not probabilities. It contains
	a softmax layer on its own. Your model won't train well if you pass it probabilities.

	:param model: Model class representing your MPNN.
	:param train_data: A 1-D list of play objects, representing all the plays
	in the training set from get_data
	:return: nothing.
	"""

	for i in range(int(len(train_data) / model.batch_size)):
		offset = i * model.batch_size
		graphs = []
		labels = []
		for m in range(offset, offset+model.batch_size):
			G = build_graph(train_data[m])
			graphs.append(G)
			labels.append(train_data[m].label)

		labels_torch = torch.from_numpy(np.array(label_converter(labels)))
		batch = dgl.batch(graphs)
		logits = F.log_softmax(model(batch), 1)
		l = F.nll_loss(logits, labels_torch)
		model.optimizer.zero_grad()
		l.backward()
		model.optimizer.step()


def test(model, test_data):
	"""
	Testing function for our model.

	Batch the plays in test_data, feed them into your model as described in train.
	After you have the logits: turn them back into numpy arrays, compare the accuracy to the labels,
	and keep a running sum.

	:param model: Model class representing your MPNN.
	:param test_data: A 1-D list of play objects, representing all the plays in your
	testing set from get_data.
	:return: total accuracy over the test set (between 0 and 1)
	"""
	tot_acc = 0
	num_batches = 0
	for i in range(int(len(test_data) / model.batch_size)):
		num_batches += 1
		offset = i * model.batch_size
		graphs = []
		labels = []
		for m in range(offset, offset + model.batch_size):
			graphs += [build_graph(test_data[m])]
			labels += [int(test_data[m].label)]
		labels = np.array(labels)
		batch = dgl.batch(graphs)
		logits = F.log_softmax(model(batch))
		logits = logits.detach().numpy()
		acc = model.accuracy_function(logits, labels)
		tot_acc += acc
	return tot_acc/num_batches

def label_converter(labels):
	newLabels = []
	for i in labels:
		label = i
		if label < -15:
			label = -15
		if label > 15:
			label = 15
		label += 15
		newLabels += [label]
	return newLabels

def main():
	# TODO: Return the training and testing data from get_data
	trainData, testData = preprocess.get_data('data/train.csv')
	print("finished preprocess")
	# TODO: Instantiate model
	model = Model()
	# TODO: Train and test for up to 15 epochs.
	for i in range(15):
		train(model, trainData)
		print("finished training epoch", i)
		acc = test(model, testData)
		print("accuracy epoch", i, "is", acc)


if __name__ == '__main__':
	main()
