import pickle
import numpy as np

np.set_printoptions(threshold=np.nan)

with open('users', 'rb') as f:
	users = pickle.load(f)

with open('map', 'rb') as f:
	mapp = pickle.load(f)

with open('adj_list', 'rb') as f:
	adj_list = pickle.load(f)

with open('link_matrix', 'rb') as f:
	link_matrix = np.load(f)

print('------------------------ users ------------------------')
print(len(users))
print(users)

print('------------------------ map ------------------------')
print(len(mapp))
print(mapp)

print('------------------------ adj_list ------------------------')
print(len(adj_list))
print(adj_list)

print('------------------------ link_matrix ------------------------')
print(link_matrix.shape)
print(link_matrix)

