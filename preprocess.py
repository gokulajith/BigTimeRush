import numpy as np
import sys
from random import shuffle
import csv
from play import Play

defense_positions = {'SS', 'DE', 'ILB', 'FS', 'CB', 'DT', 'OLB', 'NT', 'MLB'}

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
	ball_carriers = []
	for row in data:
		nodes = []
		edges = []
		label = row[0][31]
		offense = []
		defense = []
		num_nodes = 0
		for i in range(len(row)):
			node = []

			if row[i][10] == row[i][23]:
				ball_carriers.append(i)

			if row[i][36] in defense_positions:
				defense.append(num_nodes)
			else:
				offense.append(num_nodes)

			if row[i][2] == "away":
				node.append(0)
			else:
				node.append(1)
			node.append(float(row[i][3])) # x position
			node.append(float(row[i][4])) # y position
			node.append(float(row[i][5])) # speed (yards per second)
			node.append(float(row[i][6])) # acceleration (yards per second^2)
			node.append(float(row[i][7])) # distance traveled since snap
			try :
				node.append(float(row[i][8])) # orientation
			except ValueError:
				node.append(0)
			try:
				node.append(float(row[i][9])) # direction
			except ValueError:
				node.append(0)
			####COMMENT OUT
			node.append(float(row[i][14])) # yardline
			node.append(float(row[i][33])) # weight
			num_nodes += 1
			nodes += [node]

		for o in offense:
			for d in defense:
				edges.append([o, d])
		label = np.array(label).astype(np.long)
		plays += [Play(np.array(nodes, dtype=np.float32), edges, label)]

	return plays, np.array(ball_carriers)

def get_data(file_name):
	"""
	"""
	np.set_printoptions(threshold=sys.maxsize)
	test_fraction = 0.1
	plays, ball_carriers = get_plays(read_csv(file_name))
	# np.random.shuffle(plays)
	test_length = int(np.floor(len(plays) * test_fraction))
	test = plays[:test_length]
	train = plays[test_length:]
	test_ball_carriers = ball_carriers[:test_length]
	train_ball_carriers = ball_carriers[test_length:]
	return train, test, train_ball_carriers, test_ball_carriers

def get_convolution_data(file_name):
	## return train, train_labels, test, test_labels
	np.set_printoptions(threshold=sys.maxsize)
	test_fraction = 0.1
	plays, ball_carriers = get_plays(read_csv(file_name))

	test_length = int(np.floor(len(plays) * test_fraction))
	test = plays[:test_length]
	train = plays[test_length:]
	train, train_labels = convolution_play_converter(train)
	test, test_labels = convolution_play_converter(test)


	return train, train_labels, test, test_labels

def convolution_play_converter(plays):
	"""
	This function converts play data into matrices for use in
	the convolution model
	:param plays:
	:return:
	"""

	result = []
	labels = []
	for i in range(0,len(plays)):
		offense = []
		defense = []
		play = plays[i]
		play_features = play.nodes
		labels.append(play.label)

		for j in range(0, len(play_features)):
			if (j < 11):

				defense.append(play_features[j])
			else:

				offense.append(play_features[j])
		defen = np.float32(defense)
		off = np.float32(offense)

		result.append((defen, off))

	inputs = np.float32(result)
	labels = np.float32(labels)
	return inputs,labels

