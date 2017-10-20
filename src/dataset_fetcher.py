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

    def get_dataset(self, seed_user, limit, limit_on='explored'):
        """\
            seed_user is the id/screen_name/name
            of the user to start the bfs with
            limit_on can be either explored or visited
        """
        temp = self._api.rate_limit_status()
        print('friends endpoint remaining: ', temp['resources']['friends']['/friends/list']['remaining'])
        print('followers endpoint remaining: ', temp['resources']['followers']['/followers/list']['remaining'])
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
        graph = {}

        # ids that have been visited (and hence their info is in visited dict)
        # but not yet explored
        boundary = queue.Queue()

        # initialise
        seed_user = self._api.get_user(seed_user)
        self._visited[seed_user.id] = {
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
        while limited_var_val < limit and not boundary.empty():
            user_id = boundary.get()
            if debug:
                np.set_printoptions(threshold=np.inf)
                print('\nselected: ', self._visited[user_id]['screen_name'])

            rate_limit_info = self._api.rate_limit_status()['resources']['friends']['/friends/list']
            while rate_limit_info['remaining'] == 0:
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
                    graph[i.id] = {
                        'friends': [],
                        'followers': []
                    }
                    boundary.put(i.id)
                if debug:
                    print(self._visited[i.id]['screen_name'], end=' ')
                graph[user_id]['friends'].append(i.id)

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
                    graph[i.id] = {
                        'friends': [],
                        'followers': []
                    }
                    boundary.put(i.id)
                if debug:
                    print(self._visited[i.id]['screen_name'], end=' ')
                graph[user_id]['followers'].append(i.id)

            if limit_on == 'explored':
                limited_var_val = len(self._visited) - boundary.qsize()
            else:
                limited_var_val = len(self._visited)
            if debug:
                print('\nlen(visited): ', len(self._visited), 'qsize: ', boundary.qsize())

        if debug:
            for i in graph:
                print(i, graph[i]['friends'], graph[i]['followers'], sep=', ', end='\n')

        # put contents of graph in a matrix
        self._link_matrix = np.zeros((len(self._visited), len(self._visited)), dtype=np.int)

        # create map to save some time
        id_index_map = {}
        index = 0
        for user_id in graph:
            id_index_map[user_id] = index
            index += 1

        for user_id in graph:
            for friend_id in graph[user_id]['friends']:
                self._link_matrix[id_index_map[user_id], id_index_map[friend_id]] = 1
            for follower_id in graph[user_id]['followers']:
                self._link_matrix[id_index_map[follower_id], id_index_map[user_id]] = 1

        self._index_id_map = {}
        for i in id_index_map:
            self._index_id_map[id_index_map[i]] = i

    def save_dataset(self, users_path, map_path, link_matrix_path, use_sparse=False):
        with open(users_path, mode='wb') as f:
            pickle.dump(self._visited, f)
            if debug:
                print('\nusers\n', self._visited, '\n')

        with open(map_path, mode='wb') as f:
            pickle.dump(self._index_id_map, f)
            if debug:
                print('map\n', self._index_id_map, '\n')

        with open(link_matrix_path, mode='wb') as f:
            if debug:
                print('link_matrix\n', self._link_matrix, '\n')
            if use_sparse:
                sparse.save_npz(f, sparse.csr_matrix(self._link_matrix))
            else:
                np.save(f, self._link_matrix)

if __name__ == '__main__':
    key = "j5idDIRvUfwI1213Nr14Drh33"
    secret = "jOw1Dgt8dJlu4rPh3GeoGofnIV5VKLkZ8fOQqYk1zUsaSMJnVl"
    seed_user = 'n1khl'

    app = DatasetFetcher(key, secret)
    app.get_dataset(seed_user, 3, limit_on='explored')
    users_path = '../data/users'
    map_path = '../data/map'
    link_matrix_path = '../data/link_matrix'
    app.save_dataset(users_path, map_path, link_matrix_path, use_sparse=False)