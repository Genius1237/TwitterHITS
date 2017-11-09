import numpy as np
import scipy.sparse as sparse
import time
import pickle
from igraph import *
from dataset_fetcher import ListToMatrixConverter

debug = False

class HITS():
	def __init__(self, link_matrix, users, index_id_map, is_sparse=False):
		"""
		Initializes an instance of HITS

		Args:
			link_matrix: The link matrix
			users: Details of all users
			index_id_map: Dictionary representing a map from link matrix index
			to user id 
			is_sparse: True if the links matrix is a sparse matrix
		"""
		self.__is_sparse = is_sparse
		self.__link_matrix = link_matrix
		self.__link_matrix_tr = self.__link_matrix.transpose()
		self.__n = self.__link_matrix.shape[0]
		self.__hubs = np.ones((self.__n,1))
		self.__auths = np.ones((self.__n,1))
		self.__size = 30
		self.__adj_graph = Graph.Adjacency((link_matrix[0:self.__size, 0:self.__size]>0).tolist())
		self.__names = [users[index_id_map[i]]['screen_name'] for i in range(0,self.__size)]

	def calc_scores(self):
		"""Calculates hubbiness and authority
		"""
		epsilon = 0.001
		epsilon_matrix = epsilon * (np.ones((self.__n,1)).all())

		if self.__is_sparse:
			while True:
				hubs_old = self.__hubs

				self.__auths = self.__link_matrix_tr * hubs_old
				max_score = self.__auths.max(axis=0)
				if max_score != 0:
					self.__auths = self.__auths / max_score

				self.__hubs = self.__link_matrix * self.__auths
				max_score = self.__hubs.max(axis=0)
				if max_score != 0:
					self.__hubs = self.__hubs / max_score

				if ((abs(self.__hubs - hubs_old)) < epsilon_matrix).all():
					break

		else:
			while True:
				hubs_old = self.__hubs

				self.__auths = np.matmul(self.__link_matrix_tr, hubs_old)
				max_score = self.__auths.max(axis=0)
				if max_score != 0:
					self.__auths = self.__auths / max_score

				self.__hubs = np.matmul(self.__link_matrix, self.__auths)
				max_score = self.__hubs.max(axis=0)
				if max_score != 0:
					self.__hubs = self.__hubs / max_score

				#self.plot_graph(self.__hubs,self.__adj_graph,self.__names,0)
				#self.plot_graph(self.__auths,self.__adj_graph,self.__names,1)

				if ((abs(self.__hubs - hubs_old)) < epsilon_matrix).all():
					break

	def get_hubs(self):
		return self.__hubs

	def get_auths(self):
		return self.__auths

	def get_names(self):
		return self.__names

	def get_sample_adj_matrix(self):
		sample_matrix = self.__adj_graph[0:self.__size, 0:self.__size]
		return Graph.Adjacency(sample_matrix)

	def plot_graph(self,x,g,names,c):
		"""Plots the graph
		"""
		g.vs["name"] = names
		g.vs["attr"] = ["%.3f" % k for k in x]

		array_min = 0
		if x.min(axis=0) < 0.001:
			array_min = 0.001
		else:
			array_min = x.min(axis=0)

		###layout###
		layout = g.layout("kk")
		visual_style = {}
		visual_style["vertex_size"] = [(x[i]/array_min)*0.3 if x[i]>=0.001 else 10 for i in range(0,min(self.__size,len(x)))]
		visual_style["vertex_label"] = [(g.vs["name"][i],float(g.vs["attr"][i])) for i in range(0,min(self.__size,len(x)))]
		color_dict = {"0":"red" , "1":"yellow"}
		g.vs["color"] = color_dict[str(c)]
		visual_style["edge_arrow_size"]=2
		visual_style["vertex_label_size"]=35
		visual_style["layout"] = layout
		visual_style["bbox"] = (3200, 2200)
		visual_style["margin"] = 250
		visual_style["edge_width"] = 4
		plot(g, **visual_style)


class DatasetReader():
	def __init__(self):
		"""Initializes an instance of DatasetReader
		"""
		pass

	def read_users(self, users_path):
		"""Return the dictionary object (stored in a file) containing details of
		all users

		Args:
			users_path: Path to the file where info of all users is stored
		"""
		with open(users_path, mode='rb') as f:
			users = pickle.load(f)
		return users

	def read_map(self, map_path):
		"""Return the dictionary object (stored in a file) that represents a map
		from the link matrix index to user id

		Args:
			map_path: Path to the file where the map is stored
		"""
		with open(map_path, mode='rb') as f:
			index_id_map = pickle.load(f)
		return index_id_map

	def read_link_matrix(self, link_matrix_path, is_sparse=False):
		"""Return the array (stored in a file) that represents the link matrix

		Args:
			link_matrix_path: Path to the file where the link matrix is stored
			is_sparse: True if the link matrix is stored as a sparse matrix
		"""
		with open(link_matrix_path, mode='rb') as f:
			if is_sparse:
				link_matrix = sparse.load_npz(link_matrix_path)
			else:
				link_matrix = np.load(f)
		return link_matrix

def main():
	sparse = False

	users_path = '../data/users'
	map_path = '../data/map'
	link_matrix_path = '../data/link_matrix'

	# Load the stored data into objects
	r = DatasetReader()
	users = r.read_users(users_path)
	index_id_map = r.read_map(map_path)
	link_matrix = r.read_link_matrix(link_matrix_path, is_sparse=sparse)

	if debug:
		np.set_printoptions(threshold=np.inf)
		print('users\n', users, '\n')
		print('map\n', index_id_map, '\n')
		print('link_matrix\n', link_matrix.todense(), '\n')

	h = HITS(link_matrix,users,index_id_map,is_sparse=sparse)
	h.calc_scores()
	print(h.get_auths())
	print(h.get_hubs())
	x = h.get_hubs()
	y = h.get_sample_adj_matrix()
	z = h.get_names()
	h.plot_graph(x,y,z)

if __name__ == '__main__':
	main()
