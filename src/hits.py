import numpy as np
import scipy.sparse as sparse
import time
import pickle

debug = True

class HITS():
    def __init__(self, link_matrix, is_sparse=True):
        self.__is_sparse = is_sparse
        self.__link_matrix = link_matrix
        self.__link_matrix_tr = self.__link_matrix.transpose()
        self.__n = self.__link_matrix.shape[0]
        self.__hubs = np.ones((self.__n,1))
        self.__auths = np.ones((self.__n,1))

    def calc_scores(self):
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

                if ((abs(self.__hubs - hubs_old)) < epsilon_matrix).all():
                    break

    def get_hubs(self):
        return self.__hubs

    def get_auths(self):
        return self.__auths

class DatasetReader():
    def __init__(self):
        pass
    
    def read_users(self, users_path):
        with open(users_path, mode='rb') as f:
            users = pickle.load(f)
        return users
    
    def read_map(self, map_path):
        with open(map_path, mode='rb') as f:
            id_index_map = pickle.load(f)
        return id_index_map
    
    def read_link_matrix(self, link_matrix_path, is_sparse=True):
        with open(link_matrix_path, mode='rb') as f:
            if is_sparse:
                link_matrix = sparse.load_npz(link_matrix_path)
            else: 
                link_matrix = np.load(f)
        return link_matrix

    def gen_sample_input(self, size):
        def func(i):
            if i < 0.5:
                return 0
            else:
                return 1
        a = np.random.rand(size, size)
        vfunc = np.vectorize(func)
        #return [[0,1,1,1,0], [1,0,0,1,0], [0,0,0,0,1], [0,1,1,0,0], [0,0,0,0,0]]
        return vfunc(a)

def main():
    r = DatasetReader()
    users_path = 'users'
    map_path = 'map'
    link_matrix_path = 'link_matrix'
    users = r.read_users(users_path)
    id_index_map = r.read_map(map_path)
    link_matrix = r.read_link_matrix(link_matrix_path, is_sparse=True)

    if debug:
        print('users\n', users, '\n')
        print('map\n', id_index_map, '\n')
        print('link_matrix\n', link_matrix.todense(), '\n')

    h = HITS(link_matrix, is_sparse=True)
    h.calc_scores()
    print(h.get_auths())
    print(h.get_hubs())
    
if __name__ == '__main__':
    main()