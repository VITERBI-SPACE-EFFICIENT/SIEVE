import numpy as np
from collections import defaultdict
from math import floor 

class Sieve:

    '''
    Decoding using SIEVE or the standard Viterbi algorithm 
    '''

    def __init__(self, pi, A, B, y): 
        
        '''
        pi: optional, (K,)
            Initial state probabilities: Pi[i] is the probability x[0] == i. If
            None, uniform initial distribution is assumed (Pi[:] == 1/K)
        y : array (T,)
            Observation state sequence. int dtype.
        A : array (K, K)
            State transition matrix. 
        B : array (K, |O|)
            Emission matrix.
        '''
        
        self.Pi = pi 
        self.A = A 
        self.K = self.A.shape[0]
        self.B = B 
        self.y = y
        self.mp_path = [] #middlepath 
        self.path = [] #sieve 
    

    def viterbi(self):
        """
        Return the MAP estimate of state trajectory of Hidden Markov Model.
    
        Parameters
        None 
    
        Returns
        -------
        x : array (T,)
            Maximum a posteriori probability estimate of hidden state trajectory,
            conditioned on observation sequence y under the model parameters A, B,
            Pi. In other words, the optimal Viterbi path. 
        T1: array (K, T)
            the probability of the most likely path so far (DP table)
        T2: array (K, T)
            the x_j-1 of the most likely path so far (DP table)
        """
        
       
        T = len(self.y)
        T1 = np.zeros((self.K, T))
        T2 = np.zeros((self.K, T))

    
        # Initilaize the tracking tables from first observation
        T1[:, 0] = np.log( self.Pi ) + np.log( self.B[:, self.y[0]] ) 
        T2[:, 0] = 0
        
        
        # Iterate throught the observations updating the tracking tables
        for j in range(1, T): 
                                   
            for i in range(self.K): 
                                    
                
                T1[i, j] = np.max(T1[:, j - 1] + np.log(self.A[:,i]) + np.log( self.B[i, self.y[j]]) )
                 
                T2[i, j] = int(np.argmax(T1[:, j - 1] + np.log( self.A[:,i] )+ np.log(  self.B[i, self.y[j]])) )
                
                              
        # Build the output, optimal model trajectory by Backtracking 
        x = np.zeros(T, dtype=int)
        
        x[-1] = int(np.argmax(T1[:, T - 1]))

        for i in reversed(range(1, T)):
          
            x[i - 1] = T2[x[i], i]
        
        return x, T1, T2
    
    
    
    
    
    def viterbi_checkpoint_subroutine(self, y, initial_probabilities, final_state, this_step):
        """
        Standard Viterbi in a subset of data 
    
        Parameters
        None 
    
        Returns
        -------
        x : array (T,)
            Maximum a posteriori probability estimate of hidden state trajectory,
            conditioned on observation sequence y under the model parameters A, B,
            Pi. In other words, the optimal Viterbi path. 
        T1: array (K, T)
            the probability of the most likely path so far (DP table)
        T2: array (K, T)
            the x_j-1 of the most likely path so far (DP table)
        """
        
       
        T_sub = len(y)
        T1_sub = np.zeros((self.K, T_sub))
        T2_sub = np.zeros((self.K, T_sub))

        # Initilaize the tracking tables from first observation
        T1_sub[:, 0] = initial_probabilities
        T2_sub[:, 0] = 0
        
        # Iterate throught the observations updating the tracking tables
        for j in range(1, T_sub): 
            for i in range(self.K): 
                                    
                T1_sub[i, j] = np.max(T1_sub[:, j - 1] + np.log(self.A[:,i]) + np.log( self.B[i, y[j]]) )
                T2_sub[i, j] = int(np.argmax(T1_sub[:, j - 1] + np.log( self.A[:,i] )+ np.log(  self.B[i, y[j]])) )
                
        # Build the output, optimal model trajectory by Backtracking 
        x = np.zeros(T_sub, dtype=int)
        
        if final_state == None: 
            x[-1] = int(np.argmax(T1_sub[:, T_sub - 1]))
        else: 
            x[-1] = final_state 

        for i in reversed(range(1, T_sub)):
          
            x[i - 1] = T2_sub[x[i], i]
        
        if final_state==None:
            return x
        else: 
            return x[:-1]
    
    
    
    
    def viterbi_checkpoint(self, step = None):
        """
        Return the MAP estimate of state trajectory of Hidden Markov Model, using checkpoint approach to reduce space complexity.
    
        Parameters
        None 
    
        Returns
        -------
        x : array (T,)
            Maximum a posteriori probability estimate of hidden state trajectory,
            conditioned on observation sequence y under the model parameters A, B,
            Pi. In other words, the optimal Viterbi path. 
        T1: array (K, T)
            the probability of the most likely path so far (DP table)
        T2: array (K, T)
            the x_j-1 of the most likely path so far (DP table)
        """
        
       
        T = len(self.y)
        
        # default value is sqrt(T) 
        if step == None: 
            step = floor(np.sqrt(T))

    
        # Initilaize the tracking tables from first observation
        T1_previous = np.log( self.Pi ) + np.log( self.B[:, self.y[0]] ) 
        
        checkpoints = [x for x in range(0, T, step)]
        cnt_checks = 0 
        T1 = np.zeros((self.K, len(checkpoints)))
        T1[:,0] =  T1_previous 
       
        # Iterate throught the observations updating the tracking tables
        for j in range(1, T): 
            
            T1_current = []
            for i in range(self.K): 
                T1_current.append(np.max(T1_previous + np.log(self.A[:,i]) + np.log( self.B[i, self.y[j]]) ) )
            T1_previous = T1_current 
            
            if j in checkpoints: 
                cnt_checks+=1 
                T1[:,cnt_checks] =  T1_current 
            
            
        # Build the output, optimal model trajectory by Backtracking 
        path = []
        
        # go through the checkpoints one by one 
        final_state = None 
        for i_check in reversed(range(len(checkpoints))): 
            
            #initial_state = T2[:,i_check]
            initial_probabilities = T1[:,i_check]
            
            if final_state == None: 
                this_step = T - max(checkpoints) 
            else: 
                this_step = step 
            
            y = self.y[ (checkpoints[i_check]) : (checkpoints[i_check]+this_step+1) ]
                        
            this_path = self.viterbi_checkpoint_subroutine(y, initial_probabilities, final_state, this_step)
                        
            path.append(list(this_path))
                        
            final_state = this_path[0] 
        
        # return reversed entire path 
        return  [item for sublist in path[::-1] for item in sublist]

    
    
    
          
    def BFS_ancestors(self, source, indices, b):
 
        '''
        Perform single-source BFS traversal of ancestors up to b hops 
        
        Parameters: 
            source: starting node 
            indices: array (K) sequence of states 
            b: number of hops 
        '''
        
        
        # Mark all the vertices as not visited
        visited = set() 
 
        # Create a queue for BFS
        queue = []
 
        # Mark the source node as 
        # visited and enqueue it
        queue.append(source)
        queue.append("null") # for level 
        #visited.add(source)
        
        level = 0 
        
        A_t = self.A.T
        
        output_set = set() 
 
        while queue and level < b:
 
            # Dequeue a vertex from 
            # queue and print it
            s = queue.pop(0)
            
            if s == "null": 
                level += 1 # you increase one level everytime you encounter a null 
                queue.append("null") 
                
            else: 
                # Get all adjacent vertices of the
                # dequeued vertex s. If a adjacent
                # has not been visited, then mark it
                # visited and enqueue it
                for state_idx in  indices: 
                    
                    
                    i = A_t[s][state_idx]
                    
                    if i > 0: 
                        
                        if state_idx not in visited: 
                            output_set.add(state_idx) 
                            
                            queue.append(state_idx)
                            visited.add(state_idx)
     
        
        return output_set 
    
    
    def BFS_descendants(self, source, indices,   b):
 
        ''''
        Perform single-source BFS traversal of descendants up to b hops 
        
        Parameters: 
            source: starting node 
            indices: array (K) sequence of states 
            b: number of hops 
        '''
        
        
        # Mark all the vertices as not visited
        visited = set() 
         
        # Create a queue for BFS
        queue = []
 
        # Mark the source node as 
        # visited and enqueue it
        queue.append(source)
        queue.append("null") # for level 
        #visited.add(source)
        
        level = 0 
 
        output_set = set() 
        
        while queue and level < b:
 
            # Dequeue a vertex from 
            # queue and print it
            s = queue.pop(0)
            
            if s == "null": 
                level += 1 # you increase one level everytime you encounter a null 
                queue.append("null") 
                
            else: 
                # Get all adjacent vertices of the dequeued vertex s. If a adjacent
                # has not been visited, then mark it visited and enqueue it
                for state_idx in indices:
                    
                    i = self.A[s][state_idx]
                    
                
                    if i > 0: 
                        
                        if state_idx not in visited:
                            output_set.add(state_idx)
                            
                            queue.append(state_idx)
                            visited.add(state_idx)
     
        return output_set 
    
    
    
    def BFS_ancestors_middlepath(self, source, indices, b):
 
        ''''
        Perform single-source BFS traversal of ancestors up to b hops 
        for SIEVE-Middlepath
        
        Parameters: 
            source: starting node 
            indices: array (K) sequence of states 
            b: number of hops ¨
        '''
        
        
        # Mark all the vertices as not visited
        visited = set() 

        # Create a queue for BFS
        queue = []
 
        # Mark the source node as 
        # visited and enqueue it
        queue.append(source)
        queue.append("null") # for level 
        #visited.add(source)
        
        level = 0 
        
        A_t = self.A.T
 
        while queue and level < b:
 
            # Dequeue a vertex from 
            # queue and print it
            s = queue.pop(0)
            
            if s == "null": 
                level += 1 # you increase one level everytime you encounter a null 
                queue.append("null") 
                
            else: 
                # Get all adjacent vertices of the
                # dequeued vertex s. If a adjacent
                # has not been visited, then mark it
                # visited and enqueue it
                for state_idx in  indices: 
                    
                    
                    ''' check this ''' 
                    
                    i = A_t[s][state_idx]
                    
                    #state_idx = indices[idx]
                    
                    #print("state_idx "+ str(state_idxindicesindicesindicesindicesindicesindices))
                    #print("i " + str(i)) 
                    
                    if i > 0: 
                        
                        if state_idx not in visited: 
                            queue.append(state_idx)
                            visited.add(state_idx)
     
        
        return visited
        
    
    
    def BFS_descendants_middlepath(self, source, indices,  b):
 
        ''''
        Perform single-source BFS traversal of ancestors up to b hops 
        for SIEVE-Middlepath
        
        Parameters: 
            source: starting node 
            indices: array (K) sequence of states 
            b: number of hops ¨
        '''
        
        
        
        # Mark all the vertices as not visited
        visited = set() 
         
        # Create a queue for BFS
        queue = []
 
        # Mark the source node as 
        # visited and enqueue it
        queue.append(source)
        queue.append("null") # for level 
        #visited.add(source)
        
        level = 0 
 
        while queue and level < b:
 
            # Dequeue a vertex from 
            # queue and print it
            s = queue.pop(0)
            
            if s == "null": 
                level += 1 # you increase one level everytime you encounter a null 
                queue.append("null") 
                
            else: 
                # Get all adjacent vertices of the
                # dequeued vertex s. If a adjacent
                # has not been visited, then mark it
                # visited and enqueue it
                for state_idx in indices:
                    
                    i = self.A[s][state_idx]
                                                            
                    if i > 0: 
                        
                        if state_idx not in visited:
                            queue.append(state_idx)
                            visited.add(state_idx)
     
        return visited
        
        
    
    
    
    def viterbi_preprocessing_descendants_pruning_root(self, indices, b, K): 
        
        ''' 
        Implement preprocessing to count descendants  
        necessary to find the median pairs. 
        Perform one BFS to search the b-hop neighbourhood of each state amd updates number of ancestors and descendants 
        
        Parameters: 
        indices: sequence of states 
        b: number of hops to be explored 
        K: number of states 
        
        ''' 
        
        
        self.b_hop_descendants = defaultdict(int)
        
        self.b_hop_descendants_nodes = defaultdict(set)
        
        for source in range(K): 
            
            output_set = self.BFS_descendants(source, indices, b)
            
            self.b_hop_descendants[source] = len( output_set )
        
            
        
    def viterbi_preprocessing_ancestors_pruning_root(self, indices, b, K): 
        
        ''' 
        Implement preprocessing to count ancestors 
        necessary to find the median pairs. 
        Perform one BFS to search the b-hop neighbourhood of each state amd updates number of ancestors and descendants 
        
        Parameters: 
        indices: sequence of states 
        b: number of hops to be explored 
        K: number of states 
        
        ''' 

        self.b_hop_ancestors = defaultdict(int)
        
        self.b_hop_ancestors_nodes = defaultdict(set)
    
        for source in range(K): 
            
            output_set = self.BFS_ancestors(source, indices,  b)
            
            self.b_hop_ancestors[source] = len( output_set )
            
            
    
   
    
        
    
    def sieve(self, indices, A, B, y, Pi = None, K = None,  last = None): 
        """
        Return the MAP estimate of state trajectory of Hidden Markov Model.
        Implements the space efficient divide and conquer algorithm 
        
        
        Parameters
        ----------
        indices : array (T,)
             state sequence.
        A : array (K, K)
            State transition matrix. See HiddenMarkovModel.state_transition  for
            details.
        B : array (K, M)
            Emission matrix. See HiddenMarkovModel.emission for details.
        y: array (T,) observation sequence    
        Pi: optional, (K,)
            Initial state probabilities: Pi[i] is the probability x[0] == i. 
        K: optional, number of states 
        last: optional (int) 
              State id of last state 
        
    
        -------
        print the inorder Viterbi path to standard output 
        """
        
        
        T = len(y)        
        
        if K == None: 
            K = A.shape[0]
   
        if K == 1: 
            this_out = [] 
            for i in range(T): 
                this_out.append(int(indices[0]))  
            print(  this_out  ) 
            
  
        if K > 1: 
                                    
            if self.initial_state!=None: 
                Pi = np.array([0 if it!=self.initial_state else 1 for it in indices]) # we start from the 
            Pi = Pi if Pi is not None else np.full(K, 1 / K)

            # Initialize previous values arrays            
            T1 = np.log( Pi ) + np.log( B[:, y[0]]  )      
            previous_n = [-1 for _ in range(K)]
            previous_medians = [-1 for _ in range(K)]
            previous_medians_value = [np.float("inf") for _ in range(K)]
                     
            # Iterate throught the observations updating the tracking tables
            for j in range(1, T): 
                                
                # Initialize arrays of current values         
                new_t1 = [] 
                new_n = [-1 for _ in range(K)]
                new_medians = [-1 for _ in range(K)]
                new_median_values = [np.float("inf")  for _ in range(K)]
                                
                for i in range(K): 
                 
                     
                    tmp = T1 + np.log(A[:,i]) + np.log( B[i, y[j]] )
                    this_t1 = np.max( tmp )                                        
                    maximizer = np.argmax(tmp)
                    state_maximizer = indices[maximizer]             
                    new_t1.append(this_t1)                         
                    state_i = indices[i]
                  
                    prev_median_value = previous_medians_value[maximizer] 
                    
                    n_ancestors = self.b_hop_ancestors[state_maximizer]
                    n_descendants = self.b_hop_descendants[state_i]
                    
                    this_pair_to_compare = max(n_ancestors, n_descendants)
                
                    if this_pair_to_compare < prev_median_value: 
                        
                        this_median_value = this_pair_to_compare
                        new_median_values[i] = this_median_value
                        new_medians[i] = (state_maximizer , state_i) 
                        new_n[i] = j
                    
                    else:
                        if previous_medians[maximizer]!=-1: 
                            new_medians[i] = previous_medians[maximizer]
                            new_n[i] = previous_n[maximizer]
                            new_median_values[i] = previous_medians_value[maximizer] 
                    
             
                # update arrays of previous values 
                previous_n = new_n
                previous_medians = new_medians 
                previous_medians_value = new_median_values
                T1 = new_t1
            

            if last == None: 
                last = np.argmax(T1)
                
            # extract median pair 
            x_a, x_b =  new_medians[last]
         
            N_left = int(new_n[last])         
            y_left = y[:N_left] 
            
                      
            if len(y_left) >1: 
               
                
                ancestors_source = self.BFS_ancestors( x_a , indices,  N_left-1)
                             
                states_left_indices =  sorted( list(ancestors_source.union({x_a})) ) 
                
                index_x_a = states_left_indices.index(x_a)
                                
                A_left = self.A[states_left_indices, :][:, states_left_indices]
                B_left = self.B[states_left_indices, :]
                K_left = len(states_left_indices) 
             
                self.sieve(states_left_indices, A_left, B_left, y_left, Pi = None, K = K_left, last = index_x_a)
                             
            
            N_right = T - N_left 
            y_right = y[-N_right:]
            
            #inorder print of median pairs 
            self.path.append(new_medians[last])
            
            if len(y_right) >1: 
            
              
                nodes_to_consider = self.BFS_descendants( x_b , indices,  N_right-1)
                states_right_indices = sorted( list(nodes_to_consider.union({x_b})) ) 
                
                A_right = self.A[states_right_indices, :][:, states_right_indices]
                B_right = self.B[states_right_indices, :]
                K_right = len(states_right_indices) 
            
                self.initial_state = x_b                                         
                self.sieve(states_right_indices, A_right, B_right, y_right, Pi = None, K = K_right )  # append to the right 
                        
        
                  
    
    
    
    
    def sieve_middlepath(self, indices, A, B, y, Pi = None, K = None,  last = None): 
        """
        Return the MAP estimate of state trajectory of Hidden Markov Model.
        Implements the space efficient divide and conquer algorithm in the case of middle pairs that  
        split the observation sequence rather than the state space 
        
        
        Parameters
        ----------
        indices : array (T,)
             state sequence.
        A : array (K, K)
            State transition matrix. See HiddenMarkovModel.state_transition  for
            details.
        B : array (K, M)
            Emission matrix. See HiddenMarkovModel.emission for details.
        y: array (T,) observation sequence    
        Pi: optional, (K,)
            Initial state probabilities: Pi[i] is the probability x[0] == i. 
        K: optional, number of states 
        last: optional (int) 
              State id of last state 
        
    
        -------
        print the inorder Viterbi path to standard output 
        """
        
        
        T = len(y)        
        
        
        if K == None: 
            K = A.shape[0]
   
        if K == 1: 
            this_out = [] 
            for i in range(T): 
                this_out.append(int(indices[0]))  
            print(  this_out  ) 
            
  
        if K > 1: 
                                    
            if self.initial_state!=None: 
                Pi = np.array([0 if it!=self.initial_state else 1 for it in indices]) # we start from the 
            Pi = Pi if Pi is not None else np.full(K, 1 / K)

            # Initialize previous values arrays            
            T1 = np.log( Pi ) + np.log( B[:, y[0]]  )    
            previous_medians = [-1 for _ in range(K)]
          
            # Iterate throught the observations updating the tracking tables
            for j in range(1, T): 
                
                # Initialize arrays of current values         
                new_t1 = [] 
                new_medians = [-1 for _ in range(K)]
                                                
                for i in range(K): 
                 
                     
                    tmp = T1 + np.log(A[:,i]) + np.log( B[i, y[j]] )
                    this_t1 = np.max( tmp )                                        
                    maximizer = np.argmax(tmp)
                    state_maximizer = indices[maximizer]             
                    new_t1.append(this_t1)                         
                    state_i = indices[i]
                     
            
                    if j==floor(T/2): 
                        
                        new_medians[i] = (state_maximizer , state_i) 
                   
                    elif j>floor(T/2):
                        
                        new_medians[i] = previous_medians[maximizer]
                    
             
                # update arrays of previous values 
                previous_medians = new_medians 
                T1 = new_t1
            

            if last == None: 
                last = np.argmax(T1)
                
            # extract median pair 
            x_a, x_b =  new_medians[last]
         
            N_left = floor(T/2)  
            y_left = y[:N_left] 
            
                      
            if len(y_left) >1: 
               
                
                ancestors_source = self.BFS_ancestors_middlepath( x_a , indices,  N_left-1)
                             
                states_left_indices =  sorted( list(ancestors_source.union({x_a})) ) 
                
                index_x_a = states_left_indices.index(x_a)
                                
                A_left = self.A[states_left_indices, :][:, states_left_indices]
                B_left = self.B[states_left_indices, :]
                K_left = len(states_left_indices) 
             
                self.sieve_middlepath(states_left_indices, A_left, B_left, y_left, Pi = None, K = K_left, last = index_x_a)
                             
            
            N_right = T - N_left 
            y_right = y[-N_right:]
            
            
            if len(y_right) <= 1 and len(y_left) <=1 and len(self.mp_path) < len(self.y)-2: 
                self.mp_path.append( -1, )
            else:
                #inorder print of median pairs 
                self.mp_path.append(new_medians[last])
            
            if len(y_right) >1: 
            
              
                nodes_to_consider = self.BFS_descendants_middlepath( x_b , indices,  N_right-1)
                states_right_indices = sorted( list(nodes_to_consider.union({x_b})) ) 
                
                A_right = self.A[states_right_indices, :][:, states_right_indices]
                B_right = self.B[states_right_indices, :]
                K_right = len(states_right_indices) 
            
                self.initial_state = x_b                                         
                self.sieve_middlepath(states_right_indices, A_right, B_right, y_right, Pi = None, K = K_right )  # append to the right 
                
    
    
    def pretty_print_path(self, path): 
        ''' print sieve output ''' 
        if len(path) == 0: 
            raise ValueError("You must call sieve first")

        output_path = [] 
        
        output_path.append(path[0][0])
        output_path.append(path[0][1])
        i = 1
        while len(output_path) <= len(path): 
            if path[i] == -1 : 
                output_path.append(path[i+1][0])
                output_path.append(path[i+1][1])
                i+=1 
            else: 
                output_path.append(path[i][1])
            i+=1 

        print("Path " + "|" + ",".join(list(map(str, output_path))) + "|")

        
    def viterbi_preprocessing_ancestors_pruning_dag(self, indices, y): 
        ''' implement preprocessing to count ancestors and descendants
        necessary to find the median pairs.
        Assume that the states are ordered from the root to the leaves''' 
        
        # now x contains all the hidden states of the viterbi path 
        # for the central nodes 
        
        A_transpose = self.A.T 
     
        b_hop_ancestors = defaultdict(int)
        
        b_hop_ancestors_nodes_tmp = defaultdict(lambda: defaultdict(set))
        
        b_hop_ancestors_nodes =  defaultdict(set)
        
        visited = set() 
        
        while True: 
                        
            for state_u in indices: 
                
                if state_u not in visited:
                
                    to_visit_u = set() 
                    
                    for state_v in indices: 
                        
                        if A_transpose[state_u][state_v] > 0: 
                            
                            to_visit_u.add(state_v)
                            
              #      print("not entered " + str(state_u))
                    
              #      print("to visit " + str(to_visit_u))
                    
                    to_visit_u = to_visit_u.difference({state_u})
                    
                    if len( to_visit_u.difference(visited)  ) == 0: 
                       
                        # we can visit node u 
                        visited.add(state_u)
                        
              #          print("entered " + str(state_u))
                        
              #          print(len(visited))
                        
                        for neig in to_visit_u: 
                            
                           # self.b_hop_ancestors[state_u][1] +=  1
                            
                            b_hop_ancestors_nodes_tmp[state_u][1].add(neig) # 1 hop
                            
                           # for k,v in self.b_hop_ancestors[neig]: 
                            
                           #     self.b_hop_ancestors[state_u][1+k] += v 
                            
                            for k,v in b_hop_ancestors_nodes_tmp[neig].items(): 
                                #self.b_hop_ancestors[state_u][1+k] +=  len(v) 
                                b_hop_ancestors_nodes_tmp[state_u][1+k].update(v) 
                                            
                        
                
                        for b in range(len(y)): 
                            b_hop_ancestors_nodes[state_u].update(  b_hop_ancestors_nodes_tmp[state_u][b] )                    
                        b_hop_ancestors[state_u] = len( b_hop_ancestors_nodes[state_u] )
                    
                    
                                    
                        if len(visited) == len(indices): 
                            break 
                        
            if len(visited) == len(indices): 
               break        
        
        return b_hop_ancestors
        
        
        
    def  viterbi_preprocessing_descendants_pruning_dag(self, indices, y): 
        
        ''' implement preprocessing to count ancestors and descendants
        necessary to find the median pairs 
        Assume that the nodes are sorted from the root the leaves ''' 
        
        # now x contains all the hidden states of the viterbi path 
        # for the central nodes 
        
        
        b_hop_descendants = defaultdict(int)
        
        b_hop_descendants_nodes_tmp = defaultdict(lambda: defaultdict(set))
        
        b_hop_descendants_nodes = defaultdict(set)
        
        visited = set() 
        
        while True: 
                        
            for state_u in indices: 
                
                if state_u not in visited:
                
                    to_visit_u = set() 
                    
                    for state_v in indices: 
                        
                        if self.A[state_u][state_v] > 0: 
                            
                            to_visit_u.add(state_v)
                          
                    
                    to_visit_u = to_visit_u.difference({state_u})
                            
                    if len( to_visit_u.difference(visited)  ) == 0: 
                        
                        # we can visit node u 
                        visited.add(state_u)
                        
                        for neig in to_visit_u: 
                            
                            b_hop_descendants_nodes_tmp[state_u][1].add(neig) # 1 hop
                           
                            for k,v in b_hop_descendants_nodes_tmp[neig].items(): 
                                b_hop_descendants_nodes_tmp[state_u][1+k].update(v) 
                                            
                                    
                        for b in range(len(y)): 
                            b_hop_descendants_nodes[state_u].update(  b_hop_descendants_nodes_tmp[state_u][b] )                    
                        b_hop_descendants[state_u] = len( b_hop_descendants_nodes[state_u] )
                
                            
                                
                        if len(visited) == len(indices): 
                            break 
                    
            if len(visited) == len(indices): 
               break            
        
    
        return b_hop_descendants  
            

        
    def sieve_dag(self, indices, A, B, y, Pi = None, K = None, root = False, last = None): 
        """
        Return the MAP estimate of state trajectory of Hidden Markov Model.
        Implements the space efficient divide and conquer algorithm in the simplified case of 
        DAG transitions, recomputing the counts of ancestors and descendants at every iteration 
        
        
        Parameters
        ----------
        indices : array (T,)
             state sequence.
        A : array (K, K)
            State transition matrix. See HiddenMarkovModel.state_transition  for
            details.
        B : array (K, M)
            Emission matrix. See HiddenMarkovModel.emission for details.
        y: array (T,) observation sequence    
        K: optional, number of states 
        Pi: optional, (K,)
            Initial state probabilities: Pi[i] is the probability x[0] == i. 
    
        -------
        print the inorder Viterbi path to standard output 
        """
        # Initialize the priors with default (uniform dist) if not given by caller
        T = len(y)

        if K == None: 
            K = A.shape[0]
    
        if K == 1: 
            
            this_out = [] 
            for i in range(T): 
                this_out.append(int(indices[0]))  
            
            print(  this_out  ) 
            
  
        if K > 1: 
            
            # recompute ancestors and descendants at any iteration 
        
            b_hop_descendants = self.viterbi_preprocessing_descendants_pruning_dag(indices, y) 
            
            b_hop_ancestors = self.viterbi_preprocessing_ancestors_pruning_dag(indices, y)             
                                    
            if self.initial_state!=None: 
                Pi = np.array([0 if it!=self.initial_state else 1 for it in indices]) # we start from the 
            Pi = Pi if Pi is not None else np.full(K, 1 / K)
                                   
            
            # intiialize arrays of previous values 
            
            T1 = np.log( Pi ) + np.log( B[:, y[0]]  )      
            
            previous_n = [-1 for _ in range(K)]
            previous_medians = [-1 for _ in range(K)]
            previous_medians_value = [np.float("inf") for _ in range(K)]
        
            for j in range(1, T): 
            
                # initialize arrays of current values 
                
                new_t1 = [] 
                new_n = [-1 for _ in range(K)]
                new_medians = [-1 for _ in range(K)]
                new_median_values = [np.float("inf")  for _ in range(K)]
                                
                for i in range(K): 
                 
                     
                    tmp = T1 + np.log(A[:,i]) + np.log( B[i, y[j]] )
                    this_t1 = np.max( tmp )
                    maximizer = np.argmax(tmp)
                    state_maximizer = indices[maximizer]
                    new_t1.append(this_t1)                         
                    state_i = indices[i]
                     
                    prev_median = previous_medians[maximizer] 
                    prev_median_value = previous_medians_value[maximizer] 
                    
                    n_ancestors = b_hop_ancestors[state_maximizer]
                    n_descendants = b_hop_descendants[state_i]
                    
                    this_pair_to_compare = max(n_ancestors, n_descendants)

                     # update median 
                    if this_pair_to_compare < prev_median_value: 
                        
                        new_median_values[i] = this_pair_to_compare
                        new_medians[i] = (state_maximizer , state_i) 
                        new_n[i] = j
                    
                    
                    else:
                        if previous_medians[maximizer]!=-1: 
                            new_medians[i] = previous_medians[maximizer]
                            new_n[i] = previous_n[maximizer]
                            new_median_values[i] = previous_medians_value[maximizer] 
                    
                    
                # update arrays of previous values
                previous_n = new_n
                previous_medians = new_medians                
                previous_medians_value = new_median_values
                T1 = new_t1
            

            if last == None: 
                last = np.argmax(T1)
                
             
            x_a, x_b =  new_medians[last]
        
            N_left = int(new_n[last])
            N_right = T - N_left
            y_left = y[:N_left] 
         
            if len(y_left) >1: 
               
                ancestors_source = self.BFS_ancestors( x_a , indices,  N_left-1)
               
                states_left_indices =  sorted( list(ancestors_source.union({x_a})) ) 
                index_x_a = states_left_indices.index(x_a)
                                
                A_left = self.A[states_left_indices, :][:, states_left_indices]
                B_left = self.B[states_left_indices, :]
                K_left = len(states_left_indices) # - 1 
                
            
                out = self.sieve_dag(states_left_indices, A_left, B_left, y_left, Pi = None, K = K_left, last = index_x_a)
                  
            
            
               
            N_right = T - N_left 
            y_right = y[-N_right:]
            
            print(new_medians[last])
                                                  
            if len(y_right) >1: 
            
                nodes_to_consider = self.BFS_descendants( x_b , indices, N_right-1)                
                states_right_indices = sorted( list(nodes_to_consider.union({x_b})) ) 
                
                A_right = self.A[states_right_indices, :][:, states_right_indices]
                B_right = self.B[states_right_indices, :]
                K_right = len(states_right_indices) 
                
                
                
                self.initial_state = x_b                                         
                self.sieve_dag(states_right_indices, A_right, B_right, y_right, Pi = None, K = K_right ) # append to the right 
                        
                
                  
        return [] 
        
    
    
    


    
    