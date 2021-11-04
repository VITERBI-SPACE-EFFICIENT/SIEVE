from Viterbi import Sieve
import numpy as np 
import random
import networkx as nx 


def create_B(n_observables = 100, n_states = 100, sd = 1): 
    
    ''' create matrix of uniform emission probabilities '''
    
    np.random.seed(sd) 
        
    B = np.full((n_states,n_observables), float(np.random.rand(1)))
    
    B = B/B.sum(axis=1)[:,None]
    
    return B 


if __name__ == '__main__': 
        
        # seed for data generation 
        sd = 1
        # number of observable symbols 
        n_observables = 50
        # number of states 
        K = 100
        random.seed(sd)
        T = 12 
        # vector of observations 
        y = [random.randint(0,n_observables-1) for _ in range(T)] 
        # number of states 
        n_states = 250 
    
        # generate random DAG graph - sample edges with probability one 
        G=nx.gnp_random_graph(n_states,0.9,directed=True)
        DAG = nx.DiGraph([(u,v,{'weight':random.uniform(0,1)}) for (u,v) in G.edges() if u<v])        
        A = nx.to_numpy_array(DAG)
        A = A/A.sum(axis=1)
        A = np.nan_to_num(A)
        
        # generate emission probabilities 
        B = create_B(n_states = n_states, sd = sd) 
        
        # uniform initial probabilities 
        pi = np.full(n_states, 1 / n_states)
  
        print("Starting Vanilla Viterbi .. \n ") 
        
        vit = Sieve(pi, A, B, y)
        x, t1, t2 = vit.viterbi() 
        
        print("Vanilla Viterbi done .. \n" + "path: " + str(x)) 
        
        print("Starting Sieve .. \n " + "path edges: ") 
        
        vit = Sieve(pi, A, B, y)
        indices = [x for x in range(n_states)] 
        pi = np.full(n_states, 1 / n_states)
        vit.initial_state = None 
              
        out = vit.sieve_dag(indices, A, B, y, Pi =pi, K=n_states, root = True)
        
        print("Sieve done .. \n" ) 
