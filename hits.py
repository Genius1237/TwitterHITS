import numpy
import scipy

class HITS():
    def __init__(self,matrix,trmatrix):
        self.__matrix = matrix
        self.__trmatrix = trmatrix
        self.__n= self.__matrix.shape[0]
        self.__hubs= numpy.ones((self.__n,1))
        self.__auth= numpy.ones((self.__n,1))

    def calcRank(self):
        epsilon = 0.001
        iter=0
        while True:
            iter+=1

            hubs_old= self.__hubs
            self.__auth= numpy.matmul(self.__trmatrix, hubs_old)
            self.__auth= self.__auth/self.__auth.max(axis=0)

            self.__hubs= numpy.matmul(self.__matrix, self.__auth)
            self.__hubs= self.__hubs/self.__hubs.max(axis=0)

            if ((abs(self.__hubs - hubs_old))< epsilon*numpy.ones((self.__n,1))).all():
                break
        #print(iter)

    def getHubs(self):
        return self.__hubs

    def getAuth(self):
        return self.__auth

def main():
    a=[[0,1,1,1,0],[1,0,0,1,0],[0,0,0,0,1],[0,1,1,0,0],[0,0,0,0,0]]
    b= numpy.array(a)
    c= b.transpose()
    p = HITS(b,c)
    p.calcRank()
    print(p.getHubs())
    print(p.getAuth())

if __name__ == '__main__':
	main()
