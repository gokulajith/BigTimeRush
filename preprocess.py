import numpy as np
import sys
from random import shuffle
import csv
from play import Play

def read_csv(file_name):
	plays = []
	data = []
	with open(file_name) as csv_file:
		csv_reader = csv.reader(csv_file, delimiter=',')
		line_count = 0
		for row in csv_reader:
			if line_count == 0:
				line_count += 1
			else:
				line_count += 1
				data += [row]
				if line_count == 2201:
					break

	play_count = 0
	cur_play = []
	for i in range(len(data)):
		if ((i+1) % 22) == 0:
			play_count += 1
			cur_play += [data[i]]
			plays += [cur_play]
			cur_play = []
		else:
			cur_play += [data[i]]
	return plays

def get_plays(data):
	plays = []
	for row in data:
		nodes = []
		edges = []
		label = row[0][31]
		for i in range(len(row)):
			node = []
			if row[i][2] == "away":
				node.append(0)
			else:
				node.append(1)
			node.append(float(row[i][3])) # x position
			node.append(float(row[i][4])) # y position
			node.append(float(row[i][5])) # speed (yards per second)
			node.append(float(row[i][6])) # acceleration (yards per second^2)
			node.append(float(row[i][7])) # distance traveled since snap
			node.append(float(row[i][8])) # orientation
			node.append(float(row[i][9])) # direction
			node.append(float(row[i][14])) # yardline
			node.append(float(row[i][33])) # weight
			nodes += [node]
		label = np.array(label).astype(np.long)
		plays += [Play(np.array(nodes, dtype=np.float32), edges, label)]
	return plays

def get_data(file_name):
	"""
	Loads the NCI dataset from an sdf file.

	After getting back a list of all the molecules in the .sdf file,
	there's a little more preprocessing to do. First, you need to one hot
	encode the nodes of the molecule to be a 2d numpy array of shape
	(num_atoms, 119) of type np.float32 (see play.py for more details).
	After the nodes field has been taken care of, shuffle the list of
	molecules, and return a train/test split of 0.9/0.1.

	:param file_name: string, name of data file
	:return: train_data, test_data. Two lists of shuffled molecules that have had their
	nodes turned into a 2d numpy matrix, and of split 0.9 to 0.1.
	"""
	np.set_printoptions(threshold=sys.maxsize)
	test_fraction = 0.1
	plays = get_plays(read_csv(file_name))
	np.random.shuffle(plays)
	test_length = int(np.floor(len(plays) * test_fraction))
	test = plays[:test_length]
	train = plays[test_length:]
	return train, test
