import tweepy
import queue
import time
import pickle
import numpy as np
import scipy.sparse as sparse

debug = False
log = True

class DatasetFetcher():
	def __init__(self, key, secret, log_path):
		auth = tweepy.AppAuthHandler(key, secret)
		self._api = tweepy.API(auth)
		self._visited = None
		self._graph = None

	def _print_api_rem(self):
		try:
			temp = self._api.rate_limit_status()
		except tweepy.RateLimitError:
			print('Print API limit reached')
		except Exception as e:
			print('Print API limit exception: ', e)
		else:
			print('Friends endpoint remaining: ', temp['resources']['friends']['/friends/list']['remaining'])
			print('Followers endpoint remaining: ', temp['resources']['followers']['/followers/list']['remaining'])

	def _handle_limit(self, cursor, friends_or_followers):
		while True:
			try:
				yield cursor.next()
			except tweepy.RateLimitError:
				try:
					reset_time = self._api.rate_limit_status()['resources'][friends_or_followers]['/' + friends_or_followers + '/list']['reset']
				except tweepy.RateLimitError:
					print('Sleeping for ', 15 * 60, ' seconds')
					time.sleep(15 * 60)
				except Exception as e:
					print('Unexpected exception thrown: ', e)
					print('Sleeping for ', 15 * 60, ' seconds')
					time.sleep(15 * 60)
				else:
					print('Sleeping for ', max(reset_time - time.time() + 1, 1), ' seconds')
					time.sleep(max(reset_time - time.time() + 1, 1))


	def get_dataset(self, seed_user, friends_limit, followers_limit, limit_on, limit, live_save, users_path, adj_list_path):
		"""\
			seed_user is the id/screen_name/name
			of the user to start the bfs with

			friends_limit is the number of friends to consider for each user

			followers_limit is the number of followers to consider for each user

			limit_on can be either explored or visited

			limit is the limit corresponding to limit_on
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
		live_save_suffix = 0
		while limited_var_val < limit and not boundary.empty():
			print()
			self._print_api_rem()
			user_id = boundary.get()
			if debug or log:
				print('Selected:', self._visited[user_id]['screen_name'])
  
			# friends
			if log:
				print('Finding friends..')
			cnt = 0
			for friend in self._handle_limit(tweepy.Cursor(self._api.friends, user_id=user_id).items(friends_limit), 'friends'):
				cnt += 1
				if friend.id not in self._visited:
					self._visited[friend.id] = {
						'name': friend.name,
						'screen_name': friend.screen_name
					}
					self._graph[friend.id] = {
						'friends': [],
						'followers': []
					}
					boundary.put(friend.id)
				self._graph[user_id]['friends'].append(friend.id)
				if debug:
					print(self._visited[friend.id]['screen_name'], end=' ')
			if log:
				print('Found', cnt, 'friends')

			# followers
			if log:
				print('Finding followers..')
			cnt = 0
			for follower in self._handle_limit(tweepy.Cursor(self._api.followers, user_id=user_id).items(followers_limit), 'followers'):
				cnt += 1
				if follower.id not in self._visited:
					self._visited[follower.id] = {
						'name': follower.name,
						'screen_name': follower.screen_name
					}
					self._graph[follower.id] = {
						'friends': [],
						'followers': []
					}
					boundary.put(follower.id)
				self._graph[user_id]['followers'].append(follower.id)
				if debug:
					print(self._visited[follower.id]['screen_name'], end=' ')
			if log:
				print('Found', cnt, 'followers')

			if limit_on == 'explored':
				limited_var_val = len(self._visited) - boundary.qsize()
			else:
				limited_var_val = len(self._visited)
			if debug:
				print('\nlen(visited): ', len(self._visited), 'qsize: ', boundary.qsize())

			if log:
				print('Latest save suffix: ', live_save_suffix % 2)
			if live_save:
				self.save_dataset(users_path + str(live_save_suffix % 2), adj_list_path + str(live_save_suffix % 2))
			live_save_suffix += 1

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
	seed_user = 'genius1238'

	log_path = 'logs.txt'

	users_path = '../data/users'
	adj_list_path = '../data/adj_list'
	map_path = '../data/map'
	link_matrix_path = '../data/link_matrix'
	
	users_temp_path = '../data/temp/users_'
	adj_list_temp_path = '../data/temp/adj_list_'

	friends_limit = 200
	followers_limit = 200
	limit = 40

	app = DatasetFetcher(key, secret, log_path)
	if log:
		print('Obtaining dataset..')
	app.get_dataset(seed_user, friends_limit, followers_limit, 'explored', limit, True, users_temp_path, adj_list_temp_path)
	if log:
		print('Dataset obtained!')
	app.save_dataset(users_path, adj_list_path)

	# adjacency list created by dataset fetcher is used to generate the link matrix
	# which is then saved
	c = ListToMatrixConverter(adj_list_path)
	c.convert()
	c.save(map_path, link_matrix_path, use_sparse=sparse)

if __name__ == '__main__':
	main()