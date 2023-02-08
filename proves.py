import numpy as np

n_objectives = 3
n_actions = 6
n_cells = 56
Q = np.zeros([n_cells, n_cells, n_cells, n_actions, n_objectives]) 


print(Q.shape[4])
print(Q.shape[:-2])
print(Q.shape[:-2] + (Q.shape[4],))
print(Q.shape[-1])

V = np.zeros(Q.shape[:-2] + (Q.shape[4],))

print(V.shape)
