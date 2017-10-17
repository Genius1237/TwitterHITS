import numpy
import scipy

class HITS():
    def __init__(self, link_matrix):
        self.__link_matrix = matrix
        self.__link_matrix_tr = link_matrix.transpose()
        self.__n = self.__link_matrix.shape[0]
        self.__hubs = numpy.ones((self.__n,1))
        self.__auths = numpy.ones((self.__n,1))

    def calc_scores(self):
        epsilon = 0.001
        epsilon_matrix = epsilon * (numpy.ones((self.__n,1)).all())

        while True:
            hubs_old= self.__hubs

            self.__auths = numpy.matmul(self.__link_matrix_tr, hubs_old)
            self.__auths = self.__auths / self.__auths.max(axis=0)

            self.__hubs = numpy.matmul(self.__link_matrix, self.__auths)
            self.__hubs = self.__hubs / self.__hubs.max(axis=0)

            if ((abs(self.__hubs - hubs_old))< epsilon_matrix).all():
                break

    def get_hubs(self):
        return self.__hubs

    def get_auths(self):
        return self.__auths

def main():
    a=[[0,1,1,1,0],[1,0,0,1,0],[0,0,0,0,1],[0,1,1,0,0],[0,0,0,0,0]]
    b= numpy.array(a)
    p = HITS(b)
    p.calcRank()
    print(p.getHubs())
    print(p.getAuth())

if __name__ == '__main__':
	main()