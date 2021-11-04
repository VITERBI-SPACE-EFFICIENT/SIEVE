import numpy as np
from collections import defaultdict
import copy 
from scipy.stats import multivariate_normal


class SIEVE_forced_alignment:
    def __init__(self, pi, A_out, A_in, dict_id2state, dict_source2labels, allmeans, allvariances, allprobabilities,  y): 
        self.Pi = pi 
        self.A_out = A_out 
        self.A_in = A_in 
        self.y = y
        self.initial_state = None 
        self.dict_states = dict_id2state
        self.n_components = 39
        self.means_dict = allmeans
        self.vars_dict = allvariances
        self.weights_dict = allprobabilities
        self.dict_labels = dict_source2labels
        self.T = len(self.y)
       
        
        
        def evaluate_GMM(self, state, obs): 
        
            ''' evaluate emission probabilities with multivariate GMM 
            params: 
                state: hidden state of interest 
                obs: observation of interest 
                
            return 
                density (log) 
            
            ''' 
            
            st = state[1:-1]
            means = self.means_dict[st] 
            variances = self.vars_dict[st] 
            weights = [float(x[:-2]) for x in self.weights_dict[st]] 
            density = 0 
            for i in range(len(weights)): 
                var = multivariate_normal(means[i], cov=np.diag(variances[i]))
                density += (np.log(weights[i]) + np.log(var.pdf(obs)))
                
            return density
        
        
        
        def viterbi_preprocessing_descendants_forced_alignment(self, allstates, b): 
        
            ''' 
            Compute b hop descendants for the forced aligment data 
            
            params: 
                allstates: list of states 
                b : number of hops 
                
            return: 
                None 
            
            ''' 
         
            
            self.b_hop_descendants = defaultdict(int)  
            
            
            for source in allstates: 
                
                # we need to do BFS from this node to get all the b hop descendants 
               
             #   print(source)
                
                visited = set() 
                visited_emitting = dict() 
                
            
                to_be_mantained = set() 
                # Create a queue for BFS
                queue = []
         
                # Mark the source node as 
               
                # visited and enqueue it
                queue.append(source)
               # queue.append("null") # for level 
                if "s_e" in visited_emitting: 
                    visited_emitting[source] = 1
                else: 
                    visited_emitting[source] = 0 
                #visited.add(source)
                
                level = 0 
                #A_t = self.A_in 
                while queue: # and level < b:
         
                    # Dequeue a vertex from 
                    # queue and print it
                    s = queue.pop(0)
                   
                    
                    if visited_emitting[s] < b: 
                    
                        for tup  in self.A_out[s]:    
                             
                            node_id = tup[0]
                            
                            if node_id not in visited: 
                    #             if root: 
                                self.b_hop_descendants[source]+=1
                     #            else: 
                     #                to_be_mantained.add(node_id)
                                 
                                if "s_e" in node_id:
                                    visited_emitting[node_id] = visited_emitting[s] + 1 
                                else:
                                    visited_emitting[node_id] = visited_emitting[s]
                                
                                
                                queue.append(node_id)
                                visited.add(node_id)
          
                    
          
                
    
        
    def viterbi_preprocessing_ancestors_forced_alignment(self, allstates, b): 
        
        '''
        Compute b hop ancestors for the forced aligment data 
        
        params: 
            allstates: list of states 
            b : number of hops 
            
        return: 
            None 
        
         ''' 
        
        
        self.b_hop_ancestors = defaultdict(int)  
        
        for source in allstates: 
            
     
            visited = set() 
            visited_emitting = dict() 
            
        
            to_be_mantained = set() 
            # Create a queue for BFS
            queue = []
     
            queue.append(source)
            if "s_e" in visited_emitting: 
                visited_emitting[source] = 1
            else: 
                visited_emitting[source] = 0 
            
            level = 0 
            while queue: 
                s = queue.pop(0)

                if visited_emitting[s] < b: 
                
                    for tup  in self.A_in[s]:    
                         
                        node_id = tup[0]
                        
                        if node_id not in visited: 
                            self.b_hop_ancestors[source]+=1
                           
                            if "s_e" in node_id:
                                visited_emitting[node_id] = visited_emitting[s] + 1 
                            else:
                                visited_emitting[node_id] = visited_emitting[s]
                            
                            
                            queue.append(node_id)
                            visited.add(node_id)
      
    


    def sieve(self, indices, y, Pi = None, K = None, root = False, last = None): 
        """
        this version does not assume the states are stored 
        Return the MAP estimate of state trajectory of Hidden Markov Model.
        Implements the space efficient divide and conquer algorithm 
        
        
        Parameters
        ----------
        y : array (T,)
            Observation state sequence. int dtype.
        A : array (K, K)
            State transition matrix. See HiddenMarkovModel.state_transition  for
            details.
        B : array (K, M)
            Emission matrix. See HiddenMarkovModel.emission for details.
        Pi: optional, (K,)
            Initial state probabilities: Pi[i] is the probability x[0] == i. If
            None, uniform initial distribution is assumed (Pi[:] == 1/K).
    
        Returns
        -------
        x : array (T,)
            Maximum a posteriori probability estimate of hidden state trajectory,
            conditioned on observation sequence y under the model parameters A, B,
            Pi.
        T1: array (K, T)
            the probability of the most likely path so far
        T2: array (K, T)
            the x_j-1 of the most likely path so far
        """
        
      
        T = len(y)
        indices_set = set(copy.deepcopy(indices))
                
        if K == None: 
            K = len(indices)
        
    
        if K == 1: 
            
            this_out = [] 
            for i in range(T): 
                this_out.append(int(indices[0]))  
            
            print(  this_out  ) 
            
  
        if K > 1: 
                        
            if self.initial_state!=None: 
                Pi = np.array([0 if it!=self.initial_state else 1 for it in indices]) # we start from the 
                                                                            # initial states for sure 
        
            Pi = Pi if Pi is not None else np.full(K, 1 / K)
            
            ordered_indices = [] 
            cnt_j = 0
            map_indices2position = dict()
            for j in self.ordering_alignement: 
                if j in indices_set: 
                    ordered_indices.append(j) 
                    map_indices2position[j] = cnt_j
                    cnt_j+=1 
                    
            indices = copy.deepcopy(ordered_indices) 
            del ordered_indices
        
            T1 = np.zeros((len(indices))) 
            for h in range(len(indices)):
                node_i = indices[h]
                if "s_e" in node_i: 
                    this_state = self.dict_states[ node_i ]
                    T1[h] = Pi[h] + self.evaluate_GMM(this_state, self.y[0]) 
                else: 
                    T1[h] =  Pi[h] 
                
            previous_n = [-1 for _ in range(K)]
            previous_medians = [-1 for _ in range(K)]
            previous_medians_value = [np.float("inf") for _ in range(K)]
            new_t1 = copy.deepcopy(T1)
                        
            for j in range(1, T): 
                                
                new_n = [-1 for _ in range(len(indices))]
                new_medians = [-1 for _ in range(len(indices))]
                new_median_values = [np.float("inf")  for _ in range(len(indices))]

                for i in range(len(indices)): 
                    
                    node_i = indices[i]
                    
                    if "s_e" in node_i: 
                      
                        this_t1 = float("-inf")
                        node_maximizer = node_i
                        maximizer = map_indices2position[node_i] 
                        emission_prob = self.evaluate_GMM(self.dict_states[ node_i ], y[j]) 
                        
                        for node_h in self.A_in[ node_i ]: 
                            
                            h = node_h[0] 
                            if h in indices_set and h!=node_i: 
                                h_mapped = map_indices2position[h] 
                            
                                if "s_e" in node_h: 
                                
                                    prob = float( node_h[1] )
                                    h_mapped_t1 = T1[h_mapped] + prob + emission_prob
                                
                                else:
                                    
                                    prob = float( node_h[1] )
                                    h_mapped_t1 = new_t1[h_mapped] + prob + emission_prob
                                
                                
                                if h_mapped_t1  > this_t1 : 
                                    this_t1 = h_mapped_t1 
                                    maximizer = h_mapped
                                    node_maximizer = h
                                    
                                
                         
                            
                    else:
                        
                        this_t1 = float("-inf")
                        node_maximizer = node_i
                        maximizer = map_indices2position[node_i] 
                   
                        for node_h in self.A_in[ node_i ]: 
                            
                            h = node_h[0] 
                            if h in indices_set and h!=node_i:
                                
                                h_mapped = map_indices2position[h] 
                                 
                                if "s_e" in node_h: 
                                    prob = float( node_h[1] )                       
                                    h_mapped_t1 =  T1[h_mapped] + prob  
                                    
                                else: 
                                    prob = float( node_h[1] )                        
                                    h_mapped_t1 =  new_t1[h_mapped] + prob  
                                                                                            
                                if h_mapped_t1 > this_t1 : 
                                    this_t1 = h_mapped_t1 
                                    maximizer = h_mapped
                                    node_maximizer = h
                                
                    
               
                    new_t1[i] = this_t1 
                  
                    
                    this_pair_to_compare = max(self.b_hop_ancestors[node_maximizer], self.b_hop_descendants[node_i])
 
                    if this_pair_to_compare < previous_medians_value[maximizer] : 
                        new_median_values[i] = this_pair_to_compare
                        new_medians[i] = (node_maximizer , node_i) 
                        new_n[i] = j
                    
                    
                    else:
                        if previous_medians[maximizer]!=-1:
                            new_medians[i] = previous_medians[maximizer]
                            new_n[i] = previous_n[maximizer]
                            new_median_values[i] = previous_medians_value[maximizer] 
                    
                    
                previous_n = new_n
                previous_medians = new_medians 
                previous_medians_value = new_median_values
                T1 = new_t1
              
                
            
            if last == None: 
                last = np.argmax(T1)
            else:
                last = last 
                
#                                     
            if root: 
                root = False 
            
            try: 
                x_a, x_b =  new_medians[last]
                
            except: 
                return [] 
            
            N_left = int(new_n[last])
        
            N_right = T - N_left
                        
            y_left = y[:N_left] 
            
            if len(y_left) >1: 
         
                b_hop_ancestors_nodes_x_a = self.single_node_ancestors_alignment( x_a , N_left )
                states_left_indices =  sorted( list(b_hop_ancestors_nodes_x_a.union({x_a})) ) # basically indicdes is the list of node string ids (not numeric)
                index_x_a = states_left_indices.index(x_a)
                K_left = len(states_left_indices) # - 1 
                out = self.sieve(states_left_indices, y_left, Pi = None, K = K_left, last = index_x_a)
                                  
                
            #inorder print of median pairs 
            print(str(new_medians[last]))
            
            y_right = y[-N_right:]
                                                               
            if len(y_right) >1: 
                            
                b_hop_descendants_nodes_x_b = self.single_node_descendant_alignment(x_b , N_right )
                nodes_to_consider =  b_hop_descendants_nodes_x_b
                states_right_indices = sorted( list(nodes_to_consider.union({x_b})) ) 
                K_right = len(states_right_indices) 
                self.initial_state = x_b                                         
                out = self.sieve(states_right_indices, y_right, Pi = None, K = K_right )  # append to the right 
                        
                              
        return None