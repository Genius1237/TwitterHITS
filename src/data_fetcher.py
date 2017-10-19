import tweepy
import queue
import time
import pickle
import numpy as np
from collections import OrderedDict

class App():
	def __init__(self, key, secret):
		auth = tweepy.AppAuthHandler(key, secret)
		self._api = tweepy.API(auth)

	def get_info(self, seed_user, num_users, limit_on='explored'):
		"""\
			seed_user is the id of screen_name of name
			of the user to start the bfs with
			limit_on can be either explored or visited
		"""
		temp = self._api.rate_limit_status()
		print('friends: ', temp['resources']['friends']['/friends/list']['remaining'])
		print('followers: ', temp['resources']['followers']['/followers/list']['remaining'])
		# three possible states -
		# unvisited, visited but not explored, explored

		# each key-value pair is of the form
		# id: {'name': '', 'screen_name': ''}
		visited = OrderedDict()

		# each key-value pair is of the form
		# id: {'friends': [], 'followers': []}
		# set of ids in graph equal to set of ids in visited
		graph = {}

		# ids that have been visited (and hence their info is in visited dict)
		# but not yet explored
		boundary = queue.Queue()

		# initialise
		seed_user = self._api.get_user(seed_user)
		visited[seed_user.id] = {
			'name': seed_user.name,
			'screen_name': seed_user.screen_name
		}
		graph[seed_user.id] = {
			'friends': [],
			'followers': []
		}
		boundary.put(seed_user.id)

		# get the graph
		limited_var_val = 0
		while limited_var_val < num_users and not boundary.empty():
			user_id = boundary.get()

			rate_limit_info = self._api.rate_limit_status()['resources']['friends']['/friends/list']
			while rate_limit_info['remaining'] == 0:
				time.sleep(max(rate_limit_info['reset'] - (int)(time.time()), 1))
			friends = self._api.friends(user_id)
			for i in friends:
				if i.id not in visited:
					visited[i.id] = {
						'name': i.name,
						'screen_name': i.screen_name
					}
					graph[i.id] = {
						'friends': [],
						'followers': []
					}
				graph[user_id]['friends'].append(i.id)
				boundary.put(i.id)

			rate_limit_info = self._api.rate_limit_status()['resources']['followers']['/followers/list']
			while rate_limit_info['remaining'] == 0:
				time.sleep(max(rate_limit_info['reset'] - (int)(time.time()), 1))
			followers = self._api.followers(user_id)
			for i in followers:
				if i.id not in visited:
					visited[i.id] = {
						'name': i.name,
						'screen_name': i.screen_name
					}
					graph[i.id] = {
						'friends': [],
						'followers': []
					}
				graph[user_id]['followers'].append(i.id)
				boundary.put(i.id)

			if limit_on == 'explored':
				limited_var_val = len(visited) - boundary.qsize()
			else:
				limited_var_val = len(visited)
			print('limited_var_val: ', limited_var_val)

		# put contents of graph in a matrix
		for i in graph:
			print(i, i['friends'], i['followers'], sep=', ', end='\n')

	def save_info(self, users_path, link_matrix_path):
		with open(users_path) as f:
			pickle.dump(self._explored, f)
		with open(link_matrix_path) as f:
			np.save(f, self._link_matrix)

key = "j5idDIRvUfwI1213Nr14Drh33"
secret = "jOw1Dgt8dJlu4rPh3GeoGofnIV5VKLkZ8fOQqYk1zUsaSMJnVl"
app = App(key, secret)
seed_user = 'n1khl'
app.get_info(seed_user, 3, limit_on='explored')
#app.save_info()