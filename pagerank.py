import numpy as np

def pagerank1(M, eps=1.0e-8, d=0.85):
    N = M.shape[1]
    v = np.random.rand(N, 1)
    v = v / np.linalg.norm(v, 1)
    last_v = np.zeros((N, 1)) 
    M_hat = (d * M) + (((1 - d) / N) * np.ones((N, N)))
    
    while np.linalg.norm(v - last_v, 2) > eps:
        last_v = v
        v = np.matmul(M_hat, v)
    return v
    
def pagerank2(M, d=0.85):
    N = M.shape[1]   
    q=(1-d/N)*np.ones((N,1)) 
    P=np.linalg.inv(np.identity(N)-d*M)
    Q=np.matmul(P,q)
    return Q/np.linalg.norm(Q, 1)    

M = np.array([[0.000,0.0,0.5,0.0,0,0.0,0,0.0],
              [1/3.0,0.0,0.0,0.5,0,0.0,0,0.0],
              [1/3.0,0.0,0.0,0.0,0,0.0,0,0.0],
              [1/3.0,0.5,0.5,0.0,0,0.0,0,0.0],
              [0.000,0.5,0.0,0.0,0,0.5,0,0.0],
              [0.000,0.0,0.0,0.0,0,0.0,1,0.5],
              [0.000,0.0,0.0,0.5,1,0.0,0,0.5],
              [0.000,0.0,0.0,0.0,0,0.5,0,0.0]])
v1 = pagerank1(M, 0.001, 0.85)
print (v1)
#print sum(v)

v2 = pagerank2(M,0.85)
print (v2)
#print sum(v2)
