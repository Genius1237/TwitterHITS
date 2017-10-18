import numpy as np
from scipy.sparse import csc_matrix

class HITS():
    def __init__(self, link_matrix, use_sparse=True):
        self.__use_sparse = use_sparse
        if self.__use_sparse:
            self.__link_matrix = csc_matrix(link_matrix)
        else:
            self.__link_matrix = np.array(link_matrix)
        self.__link_matrix_tr = self.__link_matrix.transpose()
        self.__n = self.__link_matrix.shape[0]
        self.__hubs = np.ones((self.__n,1))
        self.__auths = np.ones((self.__n,1))

    def calc_scores(self):
        epsilon = 0.001
        epsilon_matrix = epsilon * (np.ones((self.__n,1)).all())

        if self.__use_sparse:
            while True:
                hubs_old = self.__hubs

                self.__auths = self.__link_matrix_tr * hubs_old
                self.__auths = self.__auths / self.__auths.max(axis=0)

                self.__hubs = self.__link_matrix * self.__auths
                self.__hubs = self.__hubs / self.__hubs.max(axis=0)

                if ((abs(self.__hubs - hubs_old)) < epsilon_matrix).all():
                    break
        else:
            while True:
                hubs_old = self.__hubs

                self.__auths = np.matmul(self.__link_matrix_tr, hubs_old)
                self.__auths = self.__auths / self.__auths.max(axis=0)

                self.__hubs = np.matmul(self.__link_matrix, self.__auths)
                self.__hubs = self.__hubs / self.__hubs.max(axis=0)

                if ((abs(self.__hubs - hubs_old)) < epsilon_matrix).all():
                    break

    def get_hubs(self):
        return self.__hubs

    def get_auths(self):
        return self.__auths

def main():
    a =[[0,1,1,1,0], [1,0,0,1,0], [0,0,0,0,1], [0,1,1,0,0], [0,0,0,0,0]]
    p = HITS(a, use_sparse=True)
    p.calc_scores()
    print(p.get_hubs())
    print(p.get_auths())

if __name__ == '__main__':
    main()