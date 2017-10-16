import numpy

class PageRank():

	def __init__(self,matrix):
		self.__matrix=matrix
		self.__n=self.__matrix.shape[0]
		self.__rank=(1/self.__n)*numpy.ones((self.__n,1))

	def calcRanks(self):
		e=0.001
		b=1
		iter=0
		while True:
			iter+=1
			rankold=self.__rank
			self.__rank=(b*numpy.matmul(self.__matrix,self.__rank))#+(1-b)
			if ((abs(self.__rank-rankold))<e*numpy.ones((self.__n,1))).all():
				break

		print(iter)


	def getRanks(self):
		return self.__rank

def main():
	a=[[0.5,0.5,0],[0.5,0,1],[0,0.5,0]]
	b=numpy.array(a)
	p=PageRank(b)
	p.calcRanks()
	print(p.getRanks())

if __name__ == '__main__':
	main()