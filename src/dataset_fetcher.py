import tweepy
import queue
import time
import pickle
import numpy as np
import scipy.sparse as sparse

debug = True

class DatasetFetcher():
	def __init__(self, key, secret):
		auth = tweepy.AppAuthHandler(key, secret)
		self._api = tweepy.API(auth)
		self._visited = None
		self._graph = None

	def _print_api_rem(self):
		temp = self._api.rate_limit_status()
		print('friends endpoint remaining: ', temp['resources']['friends']['/friends/list']['remaining'])
		print('followers endpoint remaining: ', temp['resources']['followers']['/followers/list']['remaining'])

	def get_dataset(self, seed_user, limit, limit_on='explored'):
		"""\
			seed_user is the id/screen_name/name
			of the user to start the bfs with
			limit_on can be either explored or visited
		"""
		self._print_api_rem()
		# three possible states -
		# unvisited, visited but not explored, explored

		# each key-value pair is of the form
		# id: {'name': '', 'screen_name': ''}
		# servers two purposes -
		#   ids in this are those that are visited
		#   stores user info corresponding to each id
		self._visited = {}

		# each key-value pair is of the form
		# id: {'friends': [], 'followers': []}
		# set of ids in graph equal to set of ids in visited
		self._graph = {}

		# ids that have been visited (and hence their info is in visited dict)
		# but not yet explored
		boundary = queue.Queue()

		# initialise
		seed_user = self._api.get_user(seed_user)
		self._visited[seed_user.id] = {
			'name': seed_user.name,
			'screen_name': seed_user.screen_name
		}
		self._graph[seed_user.id] = {
			'friends': [],
			'followers': []
		}
		boundary.put(seed_user.id)

		# get the graph
		limited_var_val = 0
		while limited_var_val < limit and not boundary.empty():
			user_id = boundary.get()
			if debug:
				print('\nselected: ', self._visited[user_id]['screen_name'])

			rate_limit_info = self._api.rate_limit_status()['resources']['friends']['/friends/list']
			while rate_limit_info['remaining'] == 0:
				self._print_api_rem()
				time.sleep(max(rate_limit_info['reset'] - (int)(time.time()), 1))
			
			friends = self._api.friends(user_id)
			if debug:
				print('friends: ')
			for i in friends:
				if i.id not in self._visited:
					self._visited[i.id] = {
						'name': i.name,
						'screen_name': i.screen_name
					}
					self._graph[i.id] = {
						'friends': [],
						'followers': []
					}
					boundary.put(i.id)
				if debug:
					print(self._visited[i.id]['screen_name'], end=' ')
				self._graph[user_id]['friends'].append(i.id)

			rate_limit_info = self._api.rate_limit_status()['resources']['followers']['/followers/list']
			while rate_limit_info['remaining'] == 0:
				time.sleep(max(rate_limit_info['reset'] - (int)(time.time()), 1))
			followers = self._api.followers(user_id)
			if debug:
				print('\nfollowers: ')
			for i in followers:
				if i.id not in self._visited:
					self._visited[i.id] = {
						'name': i.name,
						'screen_name': i.screen_name
					}
					self._graph[i.id] = {
						'friends': [],
						'followers': []
					}
					boundary.put(i.id)
				if debug:
					print(self._visited[i.id]['screen_name'], end=' ')
				self._graph[user_id]['followers'].append(i.id)

			if limit_on == 'explored':
				limited_var_val = len(self._visited) - boundary.qsize()
			else:
				limited_var_val = len(self._visited)
			if debug:
				print('\nlen(visited): ', len(self._visited), 'qsize: ', boundary.qsize())

		if debug:
			for i in self._graph:
				print(i, self._graph[i]['friends'], self._graph[i]['followers'], sep=', ', end='\n')

	def save_dataset(self, users_path, adj_list_path):
		if users_path != '':
			with open(users_path, mode='wb') as f:
				pickle.dump(self._visited, f)
				if debug:
					print('\nusers\n', self._visited, '\n')
		if adj_list_path != '':
			with open(adj_list_path, mode='wb') as f:
				pickle.dump(self._graph, f)
				if debug:
					print('\ngraph\n', self._graph, '\n')



class ListToMatrixConverter():
	def __init__(self, adj_list_path):
		with open(adj_list_path, 'rb') as f:
			self._adj_list = pickle.load(f)
		self._link_matrix = None
		self._index_id_map = None

	def convert(self):
		# put contents of self._adj_list in a matrix
		size = len(self._adj_list)
		self._link_matrix = np.zeros((size, size), dtype=np.int)

		# create map to save some time
		id_index_map = {}
		index = 0
		for user_id in self._adj_list:
			id_index_map[user_id] = index
			index += 1

		for user_id in self._adj_list:
			for friend_id in self._adj_list[user_id]['friends']:
				self._link_matrix[id_index_map[user_id], id_index_map[friend_id]] = 1
			for follower_id in self._adj_list[user_id]['followers']:
				self._link_matrix[id_index_map[follower_id], id_index_map[user_id]] = 1

		self._index_id_map = {}
		for i in id_index_map:
			self._index_id_map[id_index_map[i]] = i

	def save(self, map_path, link_matrix_path, use_sparse=False):
		if map_path != '':
			with open(map_path, 'wb') as f:
				pickle.dump(self._index_id_map, f)
			if debug:
				print('map\n', self._index_id_map, '\n')

		if link_matrix_path != '':
			with open(link_matrix_path, mode='wb') as f:
				if debug:
					print('link_matrix\n', self._link_matrix, '\n')
				if use_sparse:
					sparse.save_npz(f, sparse.csr_matrix(self._link_matrix))
				else:
					np.save(f, self._link_matrix)


def main():
	sparse = False

	if debug:
		np.set_printoptions(threshold=np.inf)
	key = 'j5idDIRvUfwI1213Nr14Drh33'
	secret = 'jOw1Dgt8dJlu4rPh3GeoGofnIV5VKLkZ8fOQqYk1zUsaSMJnVl'
	seed_user = 'n1khl'

	users_path = '../data/users'
	adj_list_path = '../data/adj_list'
	map_path = '../data/map'
	link_matrix_path = '../data/link_matrix'

	app = DatasetFetcher(key, secret)
	app.get_dataset(seed_user, 3, limit_on='explored')
	app.save_dataset(users_path, adj_list_path)

	# adjacency list created by dataset fetcher is used to generate the link matrix
	# which is then saved
	c = ListToMatrixConverter(adj_list_path)
	c.convert()
	c.save(map_path, link_matrix_path, use_sparse=sparse)

if __name__ == '__main__':
	main()