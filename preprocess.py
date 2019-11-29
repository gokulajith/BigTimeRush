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
		offense = []
		defense = []
		num_nodes = 0
		for i in range(len(row)):
			node = []

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
			node.append(float(row[i][8])) # orientation
			node.append(float(row[i][9])) # direction
			node.append(float(row[i][14])) # yardline
			node.append(float(row[i][33])) # weight
			num_nodes += 1
			nodes += [node]

		for o in offense:
			for d in defense:
				edges.append([o, d])
		label = np.array(label).astype(np.long)
		plays += [Play(np.array(nodes, dtype=np.float32), edges, label)]
	return plays

def get_data(file_name):
	"""
	"""
	np.set_printoptions(threshold=sys.maxsize)
	test_fraction = 0.1
	plays = get_plays(read_csv(file_name))
	np.random.shuffle(plays)
	test_length = int(np.floor(len(plays) * test_fraction))
	test = plays[:test_length]
	train = plays[test_length:]
	return train, test
